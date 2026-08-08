"""Deterministic mock Grounder and Planner.

Used for unit tests and for running the engine end-to-end without GPU models.
The mock grounder "knows" a ground-truth target and returns noisy predictions
that get more accurate as the crop shrinks around the target -- reproducing the
paper's central finding (small crops -> better grounding) so the search logic
can be exercised realistically.
"""
from __future__ import annotations
import random
from typing import Sequence
from .base import Box, Prediction, PlanRegion, Verdict


class MockGrounder:
    def __init__(self, target: Box, full_w: float, full_h: float, seed: int = 0,
                 accurate_frac: float = 0.15):
        self.target = target
        self.full_w = full_w
        self.full_h = full_h
        self.rng = random.Random(seed)
        # crops whose area <= accurate_frac of full image ground accurately
        self.accurate_area = accurate_frac * full_w * full_h

    def ground(self, image, instruction: str) -> Prediction:
        """`image` here is a (w, h, ox, oy) tuple describing the current crop in
        original pixels (the mock stands in for real pixel data)."""
        w, h, ox, oy = image
        tcx, tcy = self.target.center
        crop_area = w * h
        if crop_area <= self.accurate_area:
            # small crop: land inside the target (local coords)
            lx = self.rng.uniform(self.target.x1, self.target.x2) - ox
            ly = self.rng.uniform(self.target.y1, self.target.y2) - oy
            conf = 0.9
        else:
            # big crop: noisy guess biased toward the target
            noise = 0.25 * max(w, h)
            lx = (tcx - ox) + self.rng.uniform(-noise, noise)
            ly = (tcy - oy) + self.rng.uniform(-noise, noise)
            conf = 0.4
        return Prediction(Box(lx - 15, ly - 15, lx + 15, ly + 15), confidence=conf)


class MockPlanner:
    def __init__(self, target: Box):
        self.target = target

    def position_inference(self, image, instruction: str) -> Sequence[PlanRegion]:
        w, h, ox, oy = image
        tcx, tcy = self.target.center
        # one good region around the target + one distractor
        good = PlanRegion(Box(tcx - w * 0.2, tcy - h * 0.2, tcx + w * 0.2, tcy + h * 0.2),
                          "near the target")
        bad = PlanRegion(Box(ox, oy, ox + w * 0.3, oy + h * 0.3), "distractor corner")
        return [good, bad]

    def verify(self, image, box: Box, instruction: str) -> Verdict:
        cx, cy = box.center
        if self.target.contains(cx, cy):
            return Verdict.IS_TARGET
        # if the box is near the target, say elsewhere; else not found
        tcx, tcy = self.target.center
        near = abs(cx - tcx) < self.target.width * 3 and abs(cy - tcy) < self.target.height * 3
        return Verdict.TARGET_ELSEWHERE if near else Verdict.TARGET_NOT_FOUND
