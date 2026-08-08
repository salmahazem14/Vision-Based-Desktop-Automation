"""Hybrid popup handler + per-post recovery.

Detection is cheap-first: OCR + keyword match on the small, KNOWN set of Notepad
dialogs; only on low-confidence / no-match do we fall back to the VLM to classify
by meaning. Known dialogs are dismissed by policy (ground the button, click it).
UNKNOWN dialogs are screenshotted, summarized to the user, safely dismissed with
Esc, and -- crucially -- the failure is isolated to the current post so the batch
survives, with a circuit breaker for systemic failure.
"""
from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Callable

log = logging.getLogger(__name__)


class PopupType(str, Enum):
    NONE = "none"
    SAVE_CONFIRMATION = "save_confirmation"
    OVERWRITE_PROMPT = "overwrite_prompt"
    PATH_ERROR = "path_error"
    UNKNOWN = "unknown"


# Policy for KNOWN dialogs: dismiss them with the KEYBOARD, not vision.
# On each of these Windows dialogs the desired button is the DEFAULT (focused)
# one, so a single Enter confirms it instantly and deterministically -- no slow
# Gemini grounding, no race. (Save / Yes / OK are all the default button.)
POPUP_KEYS: dict[PopupType, tuple[str, ...]] = {
    PopupType.SAVE_CONFIRMATION: ("enter",),   # default = Save
    PopupType.OVERWRITE_PROMPT: ("enter",),    # default = Yes (replace)
    PopupType.PATH_ERROR: ("enter",),          # default = OK
}
# Kept for reference / the vision fallback: how each button would be described.
POPUP_POLICY: dict[PopupType, str] = {
    PopupType.SAVE_CONFIRMATION: "the Save button",
    PopupType.OVERWRITE_PROMPT: "the Yes button",
    PopupType.PATH_ERROR: "the OK button",
}

# OCR fast-path: substrings that identify each known dialog (case-insensitive).
OCR_KEYWORDS: dict[PopupType, tuple[str, ...]] = {
    PopupType.SAVE_CONFIRMATION: ("save changes", "do you want to save"),
    PopupType.OVERWRITE_PROMPT: ("already exists", "replace it", "confirm save as"),
    PopupType.PATH_ERROR: ("not valid", "cannot find", "path does not exist"),
}


@dataclass
class Blocker:
    type: PopupType
    summary: str = ""
    confidence: float = 0.0


class UnknownBlocker(Exception):
    def __init__(self, screenshot_path: Path, summary: str):
        super().__init__(summary)
        self.screenshot_path = screenshot_path
        self.summary = summary


# --- detection ------------------------------------------------------------
def classify_via_ocr(image) -> Optional[Blocker]:
    """Fast, free, deterministic. Returns a Blocker for a known dialog, or None
    if nothing recognized (caller then tries the VLM)."""
    try:
        import pytesseract
        from PIL import Image
        text = pytesseract.image_to_string(Image.fromarray(image)).lower()
    except Exception as e:                       # OCR unavailable
        log.debug("OCR unavailable: %s", e)
        return None
    for ptype, keys in OCR_KEYWORDS.items():
        if any(k in text for k in keys):
            return Blocker(ptype, summary=f"OCR matched {ptype.value}", confidence=0.9)
    return None


def classify_via_vlm(image, planner) -> Blocker:
    """Fallback: let the planner read the dialog and classify by meaning."""
    from .prompts import POPUP_CLASSIFY
    try:
        result = planner.classify_popup(image, POPUP_CLASSIFY)  # -> {"type","summary"}
        ptype = PopupType(result.get("type", "unknown"))
        return Blocker(ptype, summary=result.get("summary", ""), confidence=0.6)
    except Exception as e:
        log.debug("VLM classify failed: %s", e)
        return Blocker(PopupType.UNKNOWN, summary="classifier error", confidence=0.0)


# --- handling -------------------------------------------------------------
class PopupHandler:
    def __init__(self, grounder_locate: Callable, planner, screenshot_fn,
                 click_fn, press_fn, debug_dir: Path, min_conf: float = 0.3):
        self.locate = grounder_locate      # (w,h,instruction) -> GroundResult
        self.planner = planner
        self.screenshot = screenshot_fn
        self.click = click_fn
        self.press = press_fn
        self.debug_dir = debug_dir
        self.min_conf = min_conf
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        self.unknown_events: list[dict] = []   # log of unknown popups + how we handled them

    def detect(self, image) -> Blocker:
        blk = classify_via_ocr(image)          # cheap first pass
        if blk is not None:
            return blk
        if self.planner is not None and hasattr(self.planner, "classify_popup"):
            return classify_via_vlm(image, self.planner)  # meaning-based fallback
        return Blocker(PopupType.NONE)

    def ensure_clear(self, max_tries: int = 3, post_id: int | None = None) -> None:
        """Before continuing, make sure nothing is blocking. Raises UnknownBlocker
        (caught per-post by the orchestrator) when an unrecognized dialog persists.
        """
        for _ in range(max_tries):
            image = self.screenshot()
            blk = self.detect(image)
            if blk.type == PopupType.NONE:
                return
            if blk.type == PopupType.UNKNOWN:
                # Unknown dialog: screenshot it, let the VLM CHOOSE the safest button
                # that closes it gracefully (Cancel/No/Close/OK -- never a destructive
                # action), click it, verify it cleared, and log the whole event.
                path = self._save_debug(image, post_id)
                h, w = image.shape[0], image.shape[1]
                dismiss_desc = (
                    "the single button that most SAFELY closes or dismisses this dialog "
                    "without confirming a destructive or irreversible action. Prefer, in "
                    "order: Cancel, No, Close, OK, or the window's X close button. NEVER "
                    "choose Delete, Remove, Shut Down, Restart, Format, Yes-to-delete, or "
                    "any button that changes or loses data."
                )
                action, coords, conf = "none", None, 0.0
                try:
                    res = self.locate(w, h, dismiss_desc)     # VLM -> coordinates
                    if res.found and res.confidence >= self.min_conf:
                        self.click(*res.box.center)
                        action, coords, conf = "vlm_click", res.box.center, res.confidence
                        time.sleep(0.4)
                except Exception as e:
                    log.warning("VLM dismiss grounding failed: %s", e)

                cleared = self.detect(self.screenshot()).type == PopupType.NONE
                if not cleared:                               # fallback: Esc
                    self.press("esc")
                    time.sleep(0.4)
                    cleared = self.detect(self.screenshot()).type == PopupType.NONE
                    if cleared and action == "none":
                        action = "esc_fallback"

                event = {"post_id": post_id, "summary": blk.summary,
                         "screenshot": str(path), "action": action,
                         "coords": coords, "confidence": conf, "cleared": cleared}
                self.unknown_events.append(event)
                log.warning("UNKNOWN popup handled: %s", event)

                if cleared:
                    return                                    # gracefully closed, continue
                raise UnknownBlocker(path, blk.summary or "unrecognized dialog")
            # known dialog -> KEYBOARD (instant, deterministic; no slow grounding)
            keys = POPUP_KEYS.get(blk.type)
            if keys:
                log.info("known dialog %s -> pressing %s", blk.type.value, keys)
                self.press(*keys)
            else:
                self.press("esc")
            time.sleep(0.4)
        # tried max_tries and still blocked
        image = self.screenshot()
        raise UnknownBlocker(self._save_debug(image, post_id), "popup would not clear")

    def _save_debug(self, image, post_id: int | None) -> Path:
        from PIL import Image
        ts = int(time.time())
        name = f"debug_post{post_id}_{ts}.png" if post_id else f"debug_{ts}.png"
        path = self.debug_dir / name
        try:
            Image.fromarray(image).save(path)
        except Exception:
            pass
        return path


@dataclass
class RunReport:
    saved: list[int] = field(default_factory=list)
    skipped: list[tuple[int, str]] = field(default_factory=list)   # (id, reason)
    aborted: bool = False
    abort_reason: str = ""

    def summary(self) -> str:
        lines = [f"Saved {len(self.saved)} posts: {self.saved}"]
        if self.skipped:
            lines.append(f"Skipped {len(self.skipped)}: " +
                         ", ".join(f"post_{i}({why})" for i, why in self.skipped))
        if self.aborted:
            lines.append(f"ABORTED: {self.abort_reason}")
        return "\n".join(lines)
