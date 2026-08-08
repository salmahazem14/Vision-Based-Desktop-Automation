"""Coordinate remapping across recursion levels.

THE gotcha the paper omits: every crop has its own pixel frame. A grounder
prediction is in the cropped sub-image's coordinates; we must map it back to the
ORIGINAL screenshot before we can click it. We track a composed affine transform
(offset + scale) per recursion level.
"""
from __future__ import annotations
from dataclasses import dataclass
from .models.base import Box


@dataclass(frozen=True)
class Transform:
    """Maps points from a cropped-and-resized sub-image back to original pixels.

    A crop takes region [ox, oy, ox+cw, oy+ch] of the parent and (optionally)
    resizes it to (rw, rh) for the model. To invert:
        original_x = ox + (local_x / rw) * cw
        original_y = oy + (local_y / rh) * ch
    """
    ox: float = 0.0   # crop origin x in original pixels
    oy: float = 0.0   # crop origin y in original pixels
    sx: float = 1.0   # original-pixels-per-local-pixel in x  (cw / rw)
    sy: float = 1.0   # original-pixels-per-local-pixel in y  (ch / rh)

    @staticmethod
    def identity() -> "Transform":
        return Transform()

    def point_to_original(self, x: float, y: float) -> tuple[float, float]:
        return (self.ox + x * self.sx, self.oy + y * self.sy)

    def box_to_original(self, b: Box) -> Box:
        x1, y1 = self.point_to_original(b.x1, b.y1)
        x2, y2 = self.point_to_original(b.x2, b.y2)
        return Box(x1, y1, x2, y2)

    def compose_crop(self, crop: Box, resized_w: float, resized_h: float) -> "Transform":
        """Return the transform for a child crop taken (in ORIGINAL pixels) at
        `crop` and resized to (resized_w, resized_h) before grounding.

        Because `crop` is already in original coords, the child's origin/scale
        compose directly with this transform's mapping of that origin.
        """
        # Map the crop origin (given in original px) through the current transform
        # is unnecessary here: crops in this engine are computed in original px,
        # so the child maps local->original in one step.
        cw = max(crop.width, 1.0)
        ch = max(crop.height, 1.0)
        return Transform(
            ox=crop.x1,
            oy=crop.y1,
            sx=cw / max(resized_w, 1.0),
            sy=ch / max(resized_h, 1.0),
        )
