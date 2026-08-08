"""Round-trip tests for coordinate remapping across crop levels."""
import math
from screenseeker.models.base import Box
from screenseeker.transforms import Transform


def test_identity_is_noop():
    t = Transform.identity()
    assert t.point_to_original(10, 20) == (10, 20)


def test_crop_offset_maps_back():
    # crop taken at (100,50), no resize (200x200 -> 200x200)
    crop = Box(100, 50, 300, 250)
    t = Transform.identity().compose_crop(crop, resized_w=200, resized_h=200)
    # a point at local (0,0) is the crop origin in original px
    assert t.point_to_original(0, 0) == (100, 50)
    # local (200,200) is the far corner
    assert t.point_to_original(200, 200) == (300, 250)


def test_crop_with_resize_scales_back():
    # crop is 400x400 in original px, resized to 100x100 for the model
    crop = Box(0, 0, 400, 400)
    t = Transform.identity().compose_crop(crop, resized_w=100, resized_h=100)
    # local (50,50) -> original (200,200)
    x, y = t.point_to_original(50, 50)
    assert math.isclose(x, 200) and math.isclose(y, 200)


def test_box_round_trip():
    crop = Box(100, 100, 500, 500)         # 400x400 original
    t = Transform.identity().compose_crop(crop, resized_w=200, resized_h=200)
    local = Box(10, 10, 20, 20)
    orig = t.box_to_original(local)
    # scale is 400/200 = 2, offset 100
    assert orig.x1 == 120 and orig.x2 == 140
