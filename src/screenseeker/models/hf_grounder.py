"""Real grounder adapter: OS-Atlas-7B / UGround-7B via HuggingFace Transformers.

Runs on the GPU machine (~16GB VRAM). Fill in the model-specific prompt format
and output parsing for whichever checkpoint you load; the contract is just:
    ground(image_rgb_ndarray, instruction) -> Prediction(Box in LOCAL crop px)
"""
from __future__ import annotations
import re
from .base import Box, Prediction


class HFGrounder:
    def __init__(self, model_id: str = "OS-Copilot/OS-Atlas-Base-7B",
                 device: str = "cuda"):
        from transformers import AutoModelForCausalLM, AutoProcessor  # lazy
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype="auto", device_map=device, trust_remote_code=True,
        )
        self.device = device

    def ground(self, image, instruction: str) -> Prediction:
        h, w = image.shape[:2]
        prompt = (
            "In this UI screenshot, locate: "
            f"{instruction}. Respond with the bounding box as <box>x1,y1,x2,y2</box> "
            "in pixel coordinates."
        )
        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=64)
        text = self.processor.batch_decode(out, skip_special_tokens=True)[0]
        box = self._parse_box(text, w, h)
        return Prediction(box, confidence=0.8)

    @staticmethod
    def _parse_box(text: str, w: int, h: int) -> Box:
        nums = re.findall(r"-?\d+\.?\d*", text)
        if len(nums) >= 4:
            x1, y1, x2, y2 = (float(n) for n in nums[:4])
            # some checkpoints emit 0-1000 normalized coords; rescale if so
            if max(x1, y1, x2, y2) <= 1000 and w > 1000:
                x1, x2 = x1 / 1000 * w, x2 / 1000 * w
                y1, y2 = y1 / 1000 * h, y2 / 1000 * h
            return Box(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        return Box(w / 2 - 10, h / 2 - 10, w / 2 + 10, h / 2 + 10)  # fallback center
