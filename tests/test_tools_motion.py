"""Task 2 Stage 3: the motion tool (D103 = A, D105) on synthetic phantoms (no dataset file is read).

A smooth phantom leaks less per unit of damage than a head (0.07–0.15 against ~0.23 on the dev volumes: fewer fine
edges), so a known move here is 3–4 voxels; the scale factor is checked on the dev run.
"""
import functools
import json
import time

import numpy as np
import pytest

import bamr.tools.motion as M
from bamr.tools.motion import motion_tool
from phantoms import dim_every_nth, head_phantom, magnitude, moved_once, rayleigh_air, smooth_bias

X0 = head_phantom((192, 240, 240))


@functools.lru_cache(maxsize=None)
def moved(shift=(0, 3, 0), sigma=0.02, seed=3):
    return magnitude(moved_once(X0, shift, 0.5), sigma, seed)


@pytest.mark.parametrize("shift", [(0, 3, 0), (0, 0, 4)])
def test_motion_known(shift):
    """Whichever way the head moves, the smear runs along the slow phase-encode axis (A–P) only."""
    f = motion_tool(moved(shift))
    ev = f.evidence
    assert f.present is True and ev["leak_ap"] >= M.LINE
    assert ev["leak_ap"] > ev["leak_lr"] and ev["leak_ap"] > ev["leak_si"]
    assert f.value == round(ev["leak_ap"] / M.SCALE, 6) and "leaks into the air along A–P" in f.why


def test_motion_ghost_cannot_tell():
    """D105: ghosting leaks along A–P like motion; with its rhythm heard, motion is never a plain yes or no."""
    f = motion_tool(magnitude(dim_every_nth(X0, 3), 0.02, 1))
    assert f.present is None and f.value is None and f.confidence == "low"
    assert "ghosting" in f.why and f.evidence["rhythm_peak"] >= M.RHYTHM_LINE


@pytest.mark.parametrize("kind", ["clean", "bias"])
def test_motion_none_on(kind):
    c = {"clean": X0, "bias": smooth_bias(X0)}[kind]
    f = motion_tool(magnitude(c, 0.02, 2))
    assert f.present is False and f.confidence == "high" and f.evidence["noisy_air"] is False


def test_motion_noisy_low_confidence():
    """D105: noisy air hides a weak leak, so a no there carries low confidence and says why."""
    f = motion_tool(magnitude(X0, 0.045, 2))
    assert f.present is False and f.confidence == "low" and f.evidence["noisy_air"] is True
    assert "noisy" in f.why


def test_motion_noise_unmeasured_low_confidence():
    """Review I2: when the noise tool cannot measure (here a volume not scaled to WM = 1), motion must not claim quiet
    air: noisy_air stays None and the answer carries low confidence."""
    f = motion_tool(3 * magnitude(X0, 0.045, 2))
    assert f.present is False and f.confidence == "low" and f.evidence["noisy_air"] is None
    assert "noise could not be measured" in f.why


def ixi_like_zeros(x, seed=0):
    """Exact zeros as in the dev inputs that have them (s0006): more of the air zero the further from the head, alike
    on and beside the head's A–P lines (≈ 0 % at 10 voxels, 90 % from 64 on)."""
    from scipy import ndimage as ndi
    dist = ndi.distance_transform_edt(X0 <= 0)
    y = x.copy()
    y[(dist > 8) & (np.random.default_rng(seed).random(x.shape) < np.clip((dist - 10) / 60, 0, 0.9))] = 0
    return y


@pytest.mark.parametrize("kind", ["moved", "clean"])
def test_motion_zero_air_invariant(kind):
    """D69, at the dark air floor of the dev inputs with zeros (σ 0.004): with the zeros as they are or turned into faint
    Rayleigh noise the decision stays, and the leak stays within 0.002 of the same volume before it had zeros. Still,
    the Rayleigh copy also reads within 0.002 (D104; every dev input with zeros is still). Moved, it reads lower (0.011
    vs 0.015): the noise stands in for smear the zeros blanked, which D107 = B no longer reads. (A non-zero floor that
    fades with distance still reads as leak: a blind spot measured on phantoms and reported in the after-doc.)"""
    c = moved_once(X0, (0, 3, 0), 0.5) if kind == "moved" else X0
    x0 = magnitude(c, 0.004, 3)
    x = ixi_like_zeros(x0)
    a, b, t = motion_tool(x), motion_tool(rayleigh_air(x)), motion_tool(x0)
    assert a.present is (kind == "moved") and a.present == b.present
    assert abs(a.evidence["leak_ap"] - t.evidence["leak_ap"]) <= 0.002
    if kind == "clean":
        assert abs(a.evidence["leak_ap"] - b.evidence["leak_ap"]) <= 0.002


def test_motion_leaves_exact_zero_air_out():
    """D107 = B: exact zeros are not air the leak reads (D69). Zeros everywhere but on the head's own A–P lines (the
    floor probe's worst case) made a still head read 0.022, a false alarm; without them the air left is one floor."""
    from scipy import ndimage as ndi
    col = np.broadcast_to((X0 > 0).any(axis=1, keepdims=True), X0.shape)
    x = magnitude(X0, 0.02, 3)
    x[~col & (ndi.distance_transform_edt(X0 <= 0) > 3)] = 0
    f = motion_tool(x)
    assert f.present is False and abs(f.evidence["leak_ap"]) < M.LINE / 2


def test_motion_all_air_exact_zero_cannot_tell():
    """D107 = B: when every air voxel is an exact zero there is no air left to read: cannot tell, not a quiet no."""
    from scipy import ndimage as ndi
    x = magnitude(X0, 0.02, 3)
    x[ndi.distance_transform_edt(X0 <= 0) > 2] = 0
    f = motion_tool(x)
    assert f.present is None and f.value is None and f.confidence == "low" and "exact zeros" in f.why


@pytest.mark.parametrize("bad", ["zeros", "constant", "nan", "2d", "small"])
def test_motion_graceful(bad):
    nan = head_phantom().astype(np.float32)
    nan[48, 60, 60] = np.nan
    x = {"zeros": np.zeros((96, 120, 120), np.float32), "constant": np.full((96, 120, 120), 0.7, np.float32),
         "nan": nan, "2d": nan[:, :, 60], "small": np.ones((16, 16, 16), np.float32)}[bad]
    f = motion_tool(x)
    assert f.present is None and f.value is None and f.confidence == "low" and f.why


def test_motion_no_air():
    x = np.ones((96, 120, 120), np.float32)
    x[0, 0, 0] = 1.1  # not constant, but a "head" filling the box
    f = motion_tool(x)
    assert f.present is None and f.why


def test_motion_deterministic():
    x = moved()
    assert motion_tool(x).to_json() == motion_tool(x.copy()).to_json()


def test_motion_field_types():
    f = motion_tool(moved())
    d = json.loads(f.to_json())
    assert d["tool"] == "motion" and d["version"] == 2 and d["artifact"] == "motion" and isinstance(f.present, bool)
    assert isinstance(f.value, float) and f.confidence in ("high", "low") and isinstance(f.why, str)
    assert set(d["evidence"]) == {"leak_ap", "leak_lr", "leak_si", "raw", "line", "scale", "rhythm_peak",
                                  "rhythm_line", "noisy_air", "noise_sigma"}


def test_motion_value_never_negative(monkeypatch):
    """R3: a damage is never below 0; the leak itself (here −0.001) stays in the evidence and decides."""
    monkeypatch.setattr(M, "leak", lambda gs: -0.001)
    f = motion_tool(magnitude(X0, 0.02, 2))
    assert f.present is False and f.value == 0.0 and f.evidence["leak_ap"] == -0.001


def test_motion_why_never_contradicts_the_line(monkeypatch):
    monkeypatch.setattr(M, "leak", lambda gs: 0.0037994)  # 0.003799 as reported (6 decimals, D99 #9)
    f = motion_tool(magnitude(X0, 0.02, 2))
    assert f.present is False and "(0.003799, line 0.0038)" in f.why


def test_motion_picture_written(tmp_path):
    f = motion_tool(moved(), picture=tmp_path / "m.png")
    assert (tmp_path / "m.png").stat().st_size > 1000 and f.picture == "m.png"


def test_motion_negative_values_ok():
    x = moved().copy()
    x[:5] = -0.05
    f = motion_tool(x)
    assert f.present is True


def test_motion_speed():
    x = moved()
    t = time.perf_counter()
    motion_tool(x)
    assert time.perf_counter() - t < 10
