"""Candidate scoring: dilation, Eq.1 Gaussian centrality, NMS, and sorting.

This is the ScreenSeekeR machinery that turns a bag of noisy grounder "votes"
into a ranked list of regions to search. Pure functions, fully unit-tested.
"""
from __future__ import annotations
import math
from typing import Sequence
from .models.base import Box


def dilate(box: Box, scale: float, img_w: float, img_h: float) -> Box:
    """Grow a (often tiny) grounder box about its center by `scale`, clamped."""
    cx, cy = box.center
    half_w = max(box.width, 1.0) * scale / 2.0
    half_h = max(box.height, 1.0) * scale / 2.0
    return Box(cx - half_w, cy - half_h, cx + half_w, cy + half_h).clamp(img_w, img_h)


def iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a.x1, b.x1), max(a.y1, b.y1)
    ix2, iy2 = min(a.x2, b.x2), min(a.y2, b.y2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def centrality_score(candidate: Box, votes: Sequence[Box], sigma: float = 0.3) -> float:
    """Eq.1 from the paper.

    Each vote box's center is normalized into the candidate's local frame; a 2D
    Gaussian centered at (0.5, 0.5) scores it: ~1 at the candidate center, ~0 at
    the corner, 0 outside. Scores summed over all votes. This rewards candidates
    whose votes cluster centrally and mitigates bias toward large boxes.
    """
    total = 0.0
    cw = candidate.width or 1.0
    ch = candidate.height or 1.0
    two_sigma_sq = 2.0 * sigma * sigma
    for v in votes:
        vx, vy = v.center
        xp = (vx - candidate.x1) / cw
        yp = (vy - candidate.y1) / ch
        if not (0.0 <= xp <= 1.0 and 0.0 <= yp <= 1.0):
            continue  # vote center outside candidate -> contributes 0
        d2 = (xp - 0.5) ** 2 + (yp - 0.5) ** 2
        total += math.exp(-d2 / two_sigma_sq)
    return total


def nms(scored: Sequence[tuple[Box, float]], iou_thresh: float = 0.5) -> list[tuple[Box, float]]:
    """Greedy non-maximum suppression: keep highest-scored boxes, drop overlaps."""
    order = sorted(scored, key=lambda t: t[1], reverse=True)
    kept: list[tuple[Box, float]] = []
    for box, score in order:
        if all(iou(box, kbox) < iou_thresh for kbox, _ in kept):
            kept.append((box, score))
    return kept


def rank_candidates(
    votes: Sequence[Box],
    img_w: float,
    img_h: float,
    sigma: float = 0.3,
    dilate_scale: float = 2.5,
    nms_iou: float = 0.5,
    top_k: int | None = None,
) -> list[Box]:
    """Full pipeline: dilate votes -> score by centrality -> NMS -> sort desc.

    Returns candidate search regions in best-first order.
    """
    if not votes:
        return []
    candidates = [dilate(v, dilate_scale, img_w, img_h) for v in votes]
    scored = [(c, centrality_score(c, votes, sigma)) for c in candidates]
    deduped = nms(scored, nms_iou)
    deduped.sort(key=lambda t: t[1], reverse=True)
    ordered = [box for box, _ in deduped]
    return ordered[:top_k] if top_k else ordered
