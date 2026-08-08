"""End-to-end control-flow tests for the ScreenSeekeR engine using mock models.

These exercise the recursion, scoring integration, three-way backtracking, and
coordinate handling WITHOUT any GPU model -- proving the logic is correct even
though the real grounder runs elsewhere.
"""
from screenseeker.models.base import Box, Verdict
from screenseeker.models.mock import MockGrounder, MockPlanner
from screenseeker.config import GroundingConfig
from screenseeker.search import ScreenSeekeR, reground


FULL_W, FULL_H = 1920, 1080


def _engine(target: Box, **overrides):
    cfg = GroundingConfig(**overrides)
    g = MockGrounder(target, FULL_W, FULL_H, seed=1)
    p = MockPlanner(target)
    return ScreenSeekeR(g, p, cfg)


def test_finds_icon_top_left():
    target = Box(40, 30, 88, 78)          # small icon, top-left
    res = _engine(target).locate(FULL_W, FULL_H, "the Notepad icon")
    assert res.found
    assert target.contains(*res.box.center)   # center-in-box == success


def test_finds_icon_bottom_right():
    target = Box(1840, 980, 1888, 1028)
    res = _engine(target).locate(FULL_W, FULL_H, "the Notepad icon")
    assert res.found and target.contains(*res.box.center)


def test_finds_icon_center():
    target = Box(936, 516, 984, 564)
    res = _engine(target).locate(FULL_W, FULL_H, "the Notepad icon")
    assert res.found and target.contains(*res.box.center)


def test_reground_depth1_also_finds():
    target = Box(200, 200, 248, 248)
    cfg = GroundingConfig()
    g = MockGrounder(target, FULL_W, FULL_H, seed=2)
    p = MockPlanner(target)
    res = reground(g, p, cfg, FULL_W, FULL_H, "the Notepad icon")
    assert res.found and target.contains(*res.box.center)


def test_returns_original_pixel_coords():
    target = Box(1500, 800, 1548, 848)
    res = _engine(target).locate(FULL_W, FULL_H, "the Notepad icon")
    cx, cy = res.box.center
    assert 0 <= cx <= FULL_W and 0 <= cy <= FULL_H   # mapped back to full frame


def test_verify_three_way():
    target = Box(100, 100, 150, 150)
    p = MockPlanner(target)
    view = (200, 200, 0, 0)
    assert p.verify(view, Box(110, 110, 140, 140), "x") == Verdict.IS_TARGET
    assert p.verify(view, Box(160, 160, 190, 190), "x") == Verdict.TARGET_ELSEWHERE
    assert p.verify(view, Box(1000, 1000, 1010, 1010), "x") == Verdict.TARGET_NOT_FOUND
