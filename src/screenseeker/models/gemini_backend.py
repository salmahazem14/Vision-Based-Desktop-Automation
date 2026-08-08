"""Gemini backend: ONE model, BOTH roles (Grounder + Planner).

Uses Gemini 2.5 Flash via its OpenAI-compatible endpoint, so we reuse the openai
client already in the deps. A single free Google AI Studio key covers the whole
pipeline -- no GPU, no second provider.

    export GEMINI_API_KEY=...        # from https://aistudio.google.com

Gemini returns bounding boxes natively; per Google's convention boxes come back as
[ymin, xmin, ymax, xmax] normalized to 0-1000, which we convert to absolute pixels.
"""
from __future__ import annotations
import base64
import io
import json
import logging
import os
from typing import Sequence

from .base import Box, Prediction, PlanRegion, Verdict

log = logging.getLogger(__name__)

GEMINI_OPENAI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"


def _encode(image) -> str:
    """numpy RGB array -> data URL."""
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class GeminiBackend:
    """Implements the Grounder AND Planner protocols against one Gemini model."""

    def __init__(self, model: str = "gemini-flash-latest", api_key: str | None = None):
        from openai import OpenAI
        from pathlib import Path
        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            for base in [Path.cwd(), *Path.cwd().parents[:4]]:
                envf = base / ".env"
                if envf.is_file():
                    for line in envf.read_text(encoding="utf-8").splitlines():
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, _, v = line.partition("=")
                            if k.strip() in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
                                key = v.strip().strip('"').strip("'")
                    break
        if not key:
            raise RuntimeError("Set GEMINI_API_KEY (env var or .env file).")
        self.client = OpenAI(api_key=key, base_url=GEMINI_OPENAI_BASE,
                             timeout=30.0, max_retries=2)   # never hang on a read
        self.model = model
        
    # -- shared call --------------------------------------------------------
    def _ask_json(self, image, prompt: str) -> dict:
        resp = self.client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": _encode(image)}},
            ]}],
        )
        raw = resp.choices[0].message.content
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            log.warning("Gemini returned non-JSON: %r", raw)
            return {}

    @staticmethod
    def _norm_box_to_px(b, w: int, h: int) -> Box:
        """Gemini boxes are [ymin, xmin, ymax, xmax] in 0-1000 normalized coords."""
        if len(b) != 4:
            return Box(w / 2 - 10, h / 2 - 10, w / 2 + 10, h / 2 + 10)
        ymin, xmin, ymax, xmax = b
        x1, x2 = xmin / 1000 * w, xmax / 1000 * w
        y1, y2 = ymin / 1000 * h, ymax / 1000 * h
        return Box(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

    # -- Grounder role ------------------------------------------------------
    def ground(self, image, instruction: str) -> Prediction:
        h, w = image.shape[:2]
        prompt = (
            "Return the bounding box of this UI element: "
            f"{instruction}. "
            'Respond as STRICT JSON {"box_2d": [ymin, xmin, ymax, xmax], '
            '"confidence": 0.0-1.0} with coordinates normalized to 0-1000. '
            'If not present, set confidence to 0.'
        )
        data = self._ask_json(image, prompt)
        box = self._norm_box_to_px(data.get("box_2d", []), w, h)
        conf = float(data.get("confidence", 0.5))
        return Prediction(box, confidence=conf)

    # -- Planner role -------------------------------------------------------
    def position_inference(self, image, instruction: str) -> Sequence[PlanRegion]:
        from ..prompts import POSITION_INFERENCE
        h, w = image.shape[:2]
        # Gemini uses 0-1000 boxes; ask in that convention then convert.
        prompt = POSITION_INFERENCE.format(width=w, height=h, instruction=instruction) + (
            '\nUse box coords normalized to 0-1000 as [ymin,xmin,ymax,xmax].'
        )
        data = self._ask_json(image, prompt)
        regions: list[PlanRegion] = []
        for r in data.get("regions", []):
            b = r.get("box")
            if b and len(b) == 4:
                regions.append(PlanRegion(self._norm_box_to_px(b, w, h),
                                          r.get("rationale", "")))
        return regions

    def verify(self, image, box: Box, instruction: str) -> Verdict:
        h, w = image.shape[:2]
        prompt = (
            "In this screenshot, an element is at pixel box "
            f"[{int(box.x1)},{int(box.y1)},{int(box.x2)},{int(box.y2)}]. "
            f"Does it match: {instruction}? "
            'Reply STRICT JSON {"result": "is_target|target_elsewhere|target_not_found"}. '
            "is_target = it matches; target_elsewhere = wrong element but the target "
            "IS visible somewhere in this image; target_not_found = target not in view."
        )
        data = self._ask_json(image, prompt)
        try:
            return Verdict(data.get("result", "target_not_found"))
        except ValueError:
            return Verdict.TARGET_NOT_FOUND

    # -- popup classify (hybrid handler fallback) ---------------------------
    def classify_popup(self, image, prompt: str) -> dict:
        return self._ask_json(image, prompt)
