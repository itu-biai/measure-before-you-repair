import numpy as np
import pytest

from bamr.data.slices import CANVAS, axial_canvas, canvas_origin, zstar


def _box(shape, lo, hi):
    m = np.zeros(shape, bool)
    m[tuple(slice(a, b) for a, b in zip(lo, hi))] = True
    return m


def test_zstar():
    m = _box((60, 70, 80), (20, 20, 30), (40, 50, 45))
    m[10:50, 10:60, 37] = True  # the largest axial cross-section
    assert zstar(m) == 37


@pytest.mark.parametrize("shape", [(175, 239, 240), (179, 240, 239), (250, 249, 200)])
def test_canvas_shape_and_position(shape):
    m = _box(shape, (40, 50, 60), (shape[0] - 40, shape[1] - 50, 150))
    o_lr, o_ap = canvas_origin(m)
    vol = np.zeros(shape, np.float32)
    i, j, k = shape[0] // 2 + 7, shape[1] // 2 - 11, 100
    vol[i, j, k] = 5.0
    c = axial_canvas(vol, k, (o_lr, o_ap))
    assert c.shape == (CANVAS[1], CANVAS[0]) and c.dtype == np.float32
    assert c[j - o_ap, i - o_lr] == 5.0 and c.sum() == 5.0


def test_window_contains_brain():
    shape = (250, 250, 200)
    m = _box(shape, (30, 60, 40), (170, 235, 160))  # brain box 140 (LR) x 175 (AP), off-centre
    o_lr, o_ap = canvas_origin(m)
    k = 100
    c = axial_canvas(m.astype(np.uint8), k, (o_lr, o_ap))
    assert c.sum() == m[:, :, k].sum()
