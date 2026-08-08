"""Unit tests for the scoring module with KNOWN point positions."""
import math
from screenseeker.models.base import Box
from screenseeker.scoring import dilate, iou, centrality_score, nms, rank_candidates


def test_dilate_grows_and_centers():
    b = Box(100, 100, 110, 110)          # 10x10
    d = dilate(b, scale=3.0, img_w=1000, img_h=1000)
    assert d.center == b.center          # center preserved
    assert d.width == 30 and d.height == 30


def test_dilate_clamps_to_bounds():
    b = Box(5, 5, 7, 7)
    d = dilate(b, scale=20.0, img_w=1000, img_h=1000)
    assert d.x1 >= 0 and d.y1 >= 0       # cannot go negative


def test_iou_identical_is_one():
    b = Box(0, 0, 10, 10)
    assert math.isclose(iou(b, b), 1.0)


def test_iou_disjoint_is_zero():
    assert iou(Box(0, 0, 10, 10), Box(50, 50, 60, 60)) == 0.0


def test_centrality_center_beats_corner():
    cand = Box(0, 0, 100, 100)
    center_vote = [Box(48, 48, 52, 52)]   # near candidate center
    corner_vote = [Box(2, 2, 6, 6)]       # near candidate corner
    assert centrality_score(cand, center_vote) > centrality_score(cand, corner_vote)


def test_centrality_outside_contributes_zero():
    cand = Box(0, 0, 100, 100)
    outside = [Box(200, 200, 210, 210)]   # center outside candidate
    assert centrality_score(cand, outside) == 0.0


def test_centrality_sums_over_votes():
    cand = Box(0, 0, 100, 100)
    one = [Box(48, 48, 52, 52)]
    two = [Box(48, 48, 52, 52), Box(50, 50, 54, 54)]
    assert centrality_score(cand, two) > centrality_score(cand, one)


def test_nms_drops_overlaps_keeps_best():
    a = (Box(0, 0, 10, 10), 5.0)
    b = (Box(1, 1, 11, 11), 3.0)          # overlaps a heavily -> dropped
    c = (Box(90, 90, 100, 100), 1.0)      # disjoint -> kept
    kept = nms([a, b, c], iou_thresh=0.5)
    boxes = [box for box, _ in kept]
    assert a[0] in boxes and c[0] in boxes and b[0] not in boxes


def test_rank_orders_by_centrality_and_dedups():
    # two clustered votes near (500,500), one lone vote near (50,50)
    votes = [Box(495, 495, 505, 505), Box(498, 498, 508, 508), Box(45, 45, 55, 55)]
    ranked = rank_candidates(votes, 1000, 1000, top_k=5)
    assert ranked                              # non-empty
    # the top candidate should sit near the dense cluster
    cx, cy = ranked[0].center
    assert 400 < cx < 600 and 400 < cy < 600


def test_rank_empty_votes():
    assert rank_candidates([], 1000, 1000) == []
