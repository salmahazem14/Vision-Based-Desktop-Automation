"""Real planner adapter: GPT-4o (or any vision-LLM) via the OpenAI client.

Implements position_inference + verify (the two paper prompts) and classify_popup
(the hybrid handler's VLM fallback). Sends the image as a base64 data URL.
"""
from __future__ import annotations
import base64
import io
import json
from typing import Sequence
from .base import Box, PlanRegion, Verdict


def _encode(image) -> str:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class OpenAIPlanner:
    def __init__(self, model: str = "gpt-4o"):
        from openai import OpenAI
        self.client = OpenAI()
        self.model = model

    def _ask(self, image, prompt: str) -> dict:
        resp = self.client.chat.completions.create(
            model=self.model,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": _encode(image)}},
            ]}],
        )
        return json.loads(resp.choices[0].message.content)

    def position_inference(self, image, instruction: str) -> Sequence[PlanRegion]:
        from ..prompts import POSITION_INFERENCE
        h, w = image.shape[:2]
        data = self._ask(image, POSITION_INFERENCE.format(width=w, height=h,
                                                          instruction=instruction))
        regions = []
        for r in data.get("regions", []):
            b = r.get("box")
            if b and len(b) == 4:
                regions.append(PlanRegion(Box(*b), r.get("rationale", "")))
        return regions

    def verify(self, image, box: Box, instruction: str) -> Verdict:
        from ..prompts import RESULT_CHECK
        # caller is expected to have drawn the red box; we pass the instruction
        data = self._ask(image, RESULT_CHECK.format(instruction=instruction))
        try:
            return Verdict(data.get("result", "target_not_found"))
        except ValueError:
            return Verdict.TARGET_NOT_FOUND

    def classify_popup(self, image, prompt: str) -> dict:
        return self._ask(image, prompt)
