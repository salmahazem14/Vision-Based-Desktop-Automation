"""Local evaluation: center-in-box accuracy with position/size/theme slices.

Replaces the paper's full ScreenSpot-Pro reproduction with a small, task-right
harness over your OWN hand-labeled desktop screenshots. Same metric as the paper
(a prediction is correct iff its center point lands inside the ground-truth box),
sliced by the variables the bonuses care about, and compared across methods
(bare -> ReGround -> full ScreenSeekeR) so you can show the search actually helps.

Dataset format: a JSON list of samples:
  [{"image": "assets/eval/tl_small_light.png",
    "gt_box": [x1,y1,x2,y2],
    "position": "top-left", "size": "small", "theme": "light"}]
"""
from __future__ import annotations
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .models.base import Box


@dataclass
class Sample:
    image: str
    gt_box: Box
    position: str = ""
    size: str = ""
    theme: str = ""


def load_dataset(path: str | Path) -> list[Sample]:
    data = json.loads(Path(path).read_text())
    out = []
    for d in data:
        out.append(Sample(
            image=d["image"], gt_box=Box(*d["gt_box"]),
            position=d.get("position", ""), size=d.get("size", ""),
            theme=d.get("theme", ""),
        ))
    return out


def center_in_box(pred_center: tuple[float, float], gt: Box) -> bool:
    """The paper's metric: correct iff predicted center point falls inside GT box."""
    return gt.contains(*pred_center)


def evaluate(locate_fn: Callable[[Sample], tuple[float, float]],
             dataset: list[Sample]) -> dict:
    """`locate_fn(sample) -> predicted (x,y)`. Returns overall + sliced accuracy.

    Reports raw counts too, because with ~10-20 samples each hit swings the % a
    lot -- treat as indicative hit-rate, not a tight benchmark.
    """
    overall_hits = 0
    by = {"position": defaultdict(lambda: [0, 0]),
          "size": defaultdict(lambda: [0, 0]),
          "theme": defaultdict(lambda: [0, 0])}
    for s in dataset:
        pred = locate_fn(s)
        hit = center_in_box(pred, s.gt_box)
        overall_hits += int(hit)
        for dim in ("position", "size", "theme"):
            key = getattr(s, dim)
            if key:
                by[dim][key][0] += int(hit)
                by[dim][key][1] += 1

    def fmt(bucket):
        return {k: {"acc": h / n if n else 0.0, "hits": h, "n": n}
                for k, (h, n) in bucket.items()}

    n = len(dataset)
    return {
        "overall": {"acc": overall_hits / n if n else 0.0, "hits": overall_hits, "n": n},
        "by_position": fmt(by["position"]),
        "by_size": fmt(by["size"]),
        "by_theme": fmt(by["theme"]),
    }


def compare_methods(methods: dict[str, Callable[[Sample], tuple[float, float]]],
                    dataset: list[Sample]) -> dict[str, dict]:
    """Run several grounding methods over the same set -> the 18.9/40/48-style
    table, adapted to your local data."""
    return {name: evaluate(fn, dataset) for name, fn in methods.items()}


def annotate(image_path: str | Path, pred_center: tuple[float, float],
             gt_box: Box | None, out_path: str | Path) -> None:
    """Draw the predicted click-point (and optional GT box) on a screenshot for
    the deliverable's annotated evidence images."""
    import cv2
    import numpy as np
    from PIL import Image
    img = np.array(Image.open(image_path).convert("RGB"))
    px, py = int(pred_center[0]), int(pred_center[1])
    if gt_box is not None:
        cv2.rectangle(img, (int(gt_box.x1), int(gt_box.y1)),
                      (int(gt_box.x2), int(gt_box.y2)), (0, 200, 0), 2)
    cv2.drawMarker(img, (px, py), (255, 0, 0), cv2.MARKER_CROSS, 28, 3)
    cv2.circle(img, (px, py), 14, (255, 0, 0), 2)
    Image.fromarray(img).save(out_path)
