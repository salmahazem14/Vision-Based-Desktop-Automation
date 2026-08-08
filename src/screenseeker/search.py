"""ScreenSeekeR: recursive, planner-guided, best-first visual search.

Core idea (arXiv:2504.07981): grounders fail on full high-res screens because
targets are tiny; they succeed on small crops. So don't ground the whole screen
-- recursively narrow to the region that likely holds the target, then ground.

    visual_search(instruction, viewport, depth):
        if depth >= D_max or crop small enough:
            return verify(ground(crop))              # base case == ReGround-ish
        regions   = planner.position_inference(crop) # re-plan every level
        votes     = [grounder.ground(crop, r) for r in regions]
        ranked    = rank(dilate/score/nms/sort)      # scoring.py
        for region in ranked:                        # best-first DFS
            found, box = visual_search(child_of(region), depth+1)
            if found: return True, box               # early return up the stack
        return False, None                            # exhausted -> backtrack

The three-way Result-Check verdict controls how far we backtrack:
    is_target        -> done
    target_elsewhere -> right neighborhood: try siblings under this parent
    target_not_found -> wrong neighborhood: unwind to the parent
"""
from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import Optional, Callable

from .models.base import Box, Prediction, Grounder, Planner, Verdict
from .scoring import rank_candidates
from .transforms import Transform
from .config import GroundingConfig

log = logging.getLogger(__name__)

# A CropView describes the current sub-image in ORIGINAL pixels: (w, h, ox, oy).
# In production `image_provider(crop)` returns real pixels for the model; here we
# keep the lightweight tuple so the same engine runs against the mock.
CropView = tuple[float, float, float, float]


@dataclass
class GroundResult:
    found: bool
    box: Optional[Box]              # in ORIGINAL pixels, ready to click
    confidence: float = 0.0
    reason: str = ""


class ScreenSeekeR:
    def __init__(self, grounder: Grounder, planner: Planner,
                 cfg: GroundingConfig,
                 image_provider: Optional[Callable[[Box], object]] = None):
        self.grounder = grounder
        self.planner = planner
        self.cfg = cfg
        # image_provider maps a crop Box (original px) -> the object the models
        # consume (real pixels in prod; a CropView tuple for the mock).
        self.image_provider = image_provider

    # -- public entry -------------------------------------------------------
    def locate(self, full_w: float, full_h: float, instruction: str) -> GroundResult:
        """Find `instruction` in the full screenshot. Returns click-ready box.

        Fast path: try ONE full-image grounding first. If it is confident and the
        planner verifies it, we are done in 1-2 API calls -- which is the common
        case for a clear desktop icon. Only fall back to the (expensive) recursive
        ScreenSeekeR search when the quick pass is unconfident or unverified.
        """
        root = Box(0, 0, full_w, full_h)

        # --- fast single-shot path ---
        # NOTE: a general VLM's self-reported confidence is unreliable (often 0 even
        # when the box is correct), so we do NOT gate on it. The planner's verify()
        # is the source of truth: if it confirms the box is the target, we click.
        try:
            view = self._view(root)
            pred = self.grounder.ground(view, instruction)
            abs_box = self._local_to_original(pred.box, root)
            if self.planner.verify(view, abs_box, instruction) == Verdict.IS_TARGET:
                log.info("grounded '%s' via fast path at %s (conf=%.2f)",
                         instruction, abs_box.center, pred.confidence)
                return GroundResult(True, abs_box, max(pred.confidence, 0.5), "is_target_fast")
        except Exception as e:
            log.warning("fast path failed (%s); falling back to recursive search", e)

        # --- recursive ScreenSeekeR fallback ---
        res = self._search(root, instruction, depth=0, full_w=full_w, full_h=full_h)
        if res.found:
            log.info("grounded '%s' at %s (conf=%.2f)", instruction, res.box.center, res.confidence)
        else:
            log.warning("grounding FAILED for '%s': %s", instruction, res.reason)
        return res

    # -- recursion ----------------------------------------------------------
    def _view(self, crop: Box) -> object:
        if self.image_provider is not None:
            return self.image_provider(crop)
        # default lightweight view (used by the mock models)
        return (crop.width, crop.height, crop.x1, crop.y1)

    def _search(self, crop: Box, instruction: str, depth: int,
                full_w: float, full_h: float) -> GroundResult:
        crop = crop.clamp(full_w, full_h)
        small_enough = max(crop.width, crop.height) <= self.cfg.ground_threshold_px

        # --- base case: crop is small enough (or depth exhausted) -> ground+verify
        if depth >= self.cfg.d_max or small_enough:
            return self._ground_and_verify(crop, instruction, full_w, full_h)

        # --- inductive case: plan -> ground candidates -> rank -> DFS
        view = self._view(crop)
        regions = self.planner.position_inference(view, instruction)
        if not regions:
            # planner gave nothing -> don't dead-end; ground this crop directly
            log.info("planner proposed no regions; grounding crop directly")
            return self._ground_and_verify(crop, instruction, full_w, full_h)

        votes: list[Box] = []
        for r in regions:
            rview = self._view(r.box)
            pred = self.grounder.ground(rview, instruction)
            votes.append(self._local_to_original(pred.box, r.box))

        ranked = rank_candidates(
            votes, full_w, full_h,
            sigma=self.cfg.sigma, dilate_scale=self.cfg.dilate_scale,
            nms_iou=self.cfg.nms_iou, top_k=self.cfg.top_k_candidates,
        )

        for region in ranked:                       # best-first
            child = self._search(region, instruction, depth + 1, full_w, full_h)
            if child.found:
                return child                          # is_target bubbled up
            if child.reason == "not_found_backtrack":
                continue                              # wrong area: try next sibling
            # elsewhere: also continue to siblings (already local)
        return GroundResult(False, None, reason="not_found_backtrack")

    # -- base-case grounding + Result-Check --------------------------------
    def _ground_and_verify(self, crop: Box, instruction: str,
                           full_w: float, full_h: float) -> GroundResult:
        # verify() is the gate (self-reported confidence is unreliable). Ground,
        # verify, and accept the first box the planner confirms is the target.
        best_box: Optional[Box] = None
        best_conf = 0.0
        last_verdict = Verdict.TARGET_NOT_FOUND
        for _ in range(self.cfg.max_ground_retries + 1):
            view = self._view(crop)
            pred = self.grounder.ground(view, instruction)
            abs_box = self._local_to_original(pred.box, crop)
            verdict = self.planner.verify(view, abs_box, instruction)
            if verdict == Verdict.IS_TARGET:
                return GroundResult(True, abs_box, max(pred.confidence, 0.5), "is_target")
            if best_box is None or pred.confidence > best_conf:
                best_box, best_conf, last_verdict = abs_box, pred.confidence, verdict

        # nothing verified as the target -> report the backtracking signal
        if last_verdict == Verdict.TARGET_ELSEWHERE:
            return GroundResult(False, None, best_conf, "elsewhere")
        return GroundResult(False, None, best_conf, "not_found_backtrack")

    @staticmethod
    def _local_to_original(local_box: Box, crop: Box,
                           resized_w: float | None = None,
                           resized_h: float | None = None) -> Box:
        """Map a grounder box from the crop's LOCAL frame back to original pixels.

        The grounder sees a sub-image cropped at `crop` (original px) and possibly
        resized to (resized_w, resized_h). Default (no resize) => offset-only map,
        which is what both the mock and the native-resolution HF backend need.
        A resize-aware backend passes the resized dims to compose scale as well.
        """
        rw = resized_w if resized_w is not None else crop.width
        rh = resized_h if resized_h is not None else crop.height
        t = Transform.identity().compose_crop(crop, rw, rh)
        return t.box_to_original(local_box)


def reground(grounder: Grounder, planner: Planner, cfg: GroundingConfig,
             full_w: float, full_h: float, instruction: str,
             image_provider: Optional[Callable[[Box], object]] = None) -> GroundResult:
    """ReGround: the depth-1 degenerate case, built first as a milestone.

    Ground once on the full screen, crop `reground_crop_px` around the guess,
    re-ground on the enlarged crop. Reuses the engine with d_max clamped to 1.
    """
    import copy
    c = copy.copy(cfg)
    c.d_max = 1
    engine = ScreenSeekeR(grounder, planner, c, image_provider)
    return engine.locate(full_w, full_h, instruction)
