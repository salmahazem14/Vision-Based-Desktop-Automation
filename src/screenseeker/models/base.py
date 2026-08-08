"""Core data types and the two model interfaces (Grounder + Planner).

Keeping these abstract is what lets the same search engine run against a mock
(for tests, and for this sandbox) or the real OS-Atlas / GPT-4o backends.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, Sequence, Optional


@dataclass(frozen=True)
class Box:
    """Axis-aligned box in absolute pixel coords of the ORIGINAL screenshot."""
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    def contains(self, x: float, y: float) -> bool:
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def clamp(self, w: float, h: float) -> "Box":
        """Clamp to image bounds [0,w] x [0,h]."""
        return Box(
            max(0.0, min(self.x1, w)), max(0.0, min(self.y1, h)),
            max(0.0, min(self.x2, w)), max(0.0, min(self.y2, h)),
        )


@dataclass(frozen=True)
class Prediction:
    """A grounder output: a box plus the model's confidence in [0,1]."""
    box: Box
    confidence: float = 1.0

    @property
    def point(self) -> tuple[float, float]:
        return self.box.center


class Verdict(str, Enum):
    """Result-Check outcomes (paper, Table 8)."""
    IS_TARGET = "is_target"
    TARGET_ELSEWHERE = "target_elsewhere"   # here-ish: search siblings
    TARGET_NOT_FOUND = "target_not_found"   # wrong area: back out to parent


@dataclass(frozen=True)
class PlanRegion:
    """A region the planner proposes to search within, in absolute pixels."""
    box: Box
    rationale: str = ""


class Grounder(Protocol):
    """Pixel localizer. Good at coordinates, bad on full screens."""
    def ground(self, image, instruction: str) -> Prediction: ...


class Planner(Protocol):
    """Strong vision-LLM. Good at layout/reasoning, bad at exact pixels."""
    def position_inference(self, image, instruction: str) -> Sequence[PlanRegion]: ...
    def verify(self, image, box: Box, instruction: str) -> Verdict: ...
