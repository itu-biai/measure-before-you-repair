"""Task 2 Stage 2: the ghosting tool, option C (D97) on synthetic phantoms (no dataset file is read)."""
import functools
import json
import time

import numpy as np
import pytest

import bamr.tools.ghosting as G
from bamr.tools.ghosting import ghosting_tool
from phantoms import (add_spikes, air_zeros, dim_every_nth, head_phantom, magnitude, periodic_modulation,
                      rayleigh_air, slow_drift, smooth_bias, smooth_phase)

# Full size: the period view's disk (64 samples) and its line 150 were set on 192 × 240 × 240 volumes, where the disk
# is the centre of k-space. On a half-size box the disk covers the whole plane, out where a phantom holds only noise.
X0 = head_phantom((192, 240, 240))
FOV = 240


@functools.lru_cache(maxsize=None)
def ghosted(n=3, seed=1):
    return magnitude(dim_every_nth(X0, n), 0.02, seed)


@pytest.mark.parametrize("n", [2, 3, 4])
def test_ghost_every_nth(n):
    """Copies every 240 / n mm; n = 4 must not read as its overtone n = 2 (the 30 % rule, D97)."""
    f = ghosting_tool(ghosted(n))
    assert f.present is True and "period" in f.evidence["views"]
    assert abs(f.evidence["copy_mm"] - FOV / n) <= 2 and abs(f.evidence["period_lines"] - n) < 0.1
    assert f.value > 0


def test_ghost_phase_period_2_6():
    """Scanner-like: image phase, a complex modulation, period 2.6 lines (not a divisor of the field of view). Depth
    0.25: a smooth phantom shows less than a head does; how faint a ghost is still found is measured on real anatomy
    (dev run, scanner-like check), not here."""
    c = periodic_modulation(X0 * np.exp(1j * smooth_phase(X0.shape)), 2.6, depth=0.25, theta=0.0)
    f = ghosting_tool(magnitude(c, 0.02, 3))
    assert f.present is True and abs(f.evidence["copy_mm"] - FOV / 2.6) <= 2


def test_air_view_finds_copy_shift():
    """The air view's measurement: a faint copy of the head ±60 voxels along A–P is found at that shift. Its line
    (0.0056) needs a real head's fine detail to be reached and is tested through `_decide`."""
    from bamr.tools.regions import head_mask
    c = X0 + 0.02 * (np.roll(X0, 60, axis=1) + np.roll(X0, -60, axis=1))
    x = magnitude(c, 0.02, 3)
    a = G.air_view(x, head_mask(x), 1)
    assert abs(a["shift"] - 60) <= 1 and a["share"] > 0.005 and a["prominence"] > 0


@pytest.mark.parametrize("kind", ["noise", "bias", "drift", "spike"])
def test_ghost_none_on(kind):
    c = {"noise": X0, "bias": smooth_bias(X0), "drift": slow_drift(X0),
         "spike": add_spikes(X0, [(5, -15, 12)], [0.05])}[kind]
    f = ghosting_tool(magnitude(c, 0.02, 2))
    assert f.present is False and f.evidence["views"] == [] and f.evidence["copy_mm"] is None


def test_ghost_decide_rule():
    assert G._decide(200, 0.001) == (True, ["period"], "high")
    assert G._decide(50, 0.008) == (True, ["air"], "low")  # only the air view: low confidence
    assert G._decide(200, 0.008) == (True, ["period", "air"], "high")
    assert G._decide(140, 0.001) == (False, [], "low")  # within 10 % of the 150 line
    assert G._decide(20, 0.001) == (False, [], "high")


def test_ghost_zero_air_invariant():
    x = air_zeros(ghosted(3).copy(), X0)
    a, b = ghosting_tool(x), ghosting_tool(rayleigh_air(x))
    assert a.present is True and a.present == b.present
    assert abs(b.value / a.value - 1) < 0.01


def test_ghost_deterministic():
    x = ghosted(3)
    assert ghosting_tool(x).to_json() == ghosting_tool(x.copy()).to_json()


def test_ghost_field_types():
    f = ghosting_tool(ghosted(3))
    d = json.loads(f.to_json())
    assert d["tool"] == "ghosting" and d["version"] == 1 and isinstance(f.present, bool)
    assert isinstance(f.value, float) and f.confidence in ("high", "low") and f.why.startswith("A–P line energy repeats")
    assert set(d["evidence"]) == {"period_lines", "copy_mm", "peak", "peak_line", "air_copy", "air_line", "views",
                                  "spec_amp", "head_rms", "raw", "scale"}
    assert set(d["evidence"]["air_copy"]) == {"shift_mm", "share", "prominence"}


def test_ghost_picture_written(tmp_path):
    f = ghosting_tool(ghosted(3), picture=tmp_path / "g.png")
    assert (tmp_path / "g.png").stat().st_size > 1000 and f.picture == "g.png"


def test_ghost_negative_values_ok():
    x = ghosted(3).copy()
    x[:5] = -0.05
    f = ghosting_tool(x)
    assert f.present is True and f.value > 0


@pytest.mark.parametrize("bad", ["zeros", "constant", "nan", "2d", "small"])
def test_ghost_graceful(bad):
    nan = head_phantom().astype(np.float32)
    nan[48, 60, 60] = np.nan
    x = {"zeros": np.zeros((96, 120, 120), np.float32), "constant": np.full((96, 120, 120), 0.7, np.float32),
         "nan": nan, "2d": nan[:, :, 60], "small": np.ones((16, 16, 16), np.float32)}[bad]
    f = ghosting_tool(x)
    assert f.present is None and f.value is None and f.confidence == "low" and f.why


def test_ghost_speed():
    x = ghosted(3)
    t = time.perf_counter()
    ghosting_tool(x)
    assert time.perf_counter() - t < 10


# ---------------------------------------------------------------- final-review fixes (I1, M1)
def test_air_view_symmetric_uses_measured_side():
    """I1: a copy sits at +Δ and −Δ; where one side cannot be measured (the shifted head never reaches the air),
    the other side alone is used, so a shift keeps all its neighbours (as the pilot's numbers do)."""
    g = np.array([0.0, 0.1, 0.2, 0.3, np.nan, np.nan])  # N = 6: g(4) = g(−2) and g(5) = g(−1) unmeasured
    assert np.allclose(G._symmetric(g), [0.0, 0.1, 0.2, 0.3, 0.2, 0.1])
    assert np.isnan(G._symmetric(np.array([0.0, np.nan, 0.5, np.nan])))[[1, 3]].all()


def test_air_view_roundoff_denominator_unmeasured():
    """I1: a shift whose overlap with the air is round-off (|den| ≤ 1e-6 × the largest) is unmeasured, not a number."""
    den = np.array([0.0, 1e-12, 5e-3, -2e-9, 0.07])
    assert G._measured(den).tolist() == [False, False, True, False, True]


def test_ghost_why_never_contradicts_the_line(monkeypatch):
    """M1: next to a line, the reason prints enough digits to stay on the side the decision is on."""
    f0, rel = np.fft.rfftfreq(240), np.ones(121)
    monkeypatch.setattr(G, "period_view", lambda K, axis: {"peak": 149.6, "f_star": 0.33, "period_lines": 3.0,
                                                           "spec_amp": 0.01, "rz": np.zeros(240), "f": f0, "rel": rel})
    monkeypatch.setattr(G, "air_view", lambda x, head, axis: {"prominence": 0.00559, "shift": 40, "share": 0.01})
    f = ghosting_tool(ghosted(3))
    assert f.present is False
    assert "peak 149.6, line 150" in f.why and "(0.00559, line 0.0056)" in f.why


# ---------------------------------------------------------------- Stage 3: the curve the motion tool reads (R2)
def test_air_curve_matches_air_view():
    """air_curve is the symmetric g(Δ) air_view picks its copy from: same number at the same shift."""
    from bamr.tools.regions import head_mask
    c = X0 + 0.02 * (np.roll(X0, 60, axis=1) + np.roll(X0, -60, axis=1))
    x = magnitude(c, 0.02, 3)
    head = head_mask(x)
    gs, a = G.air_curve(x, head, 1), G.air_view(x, head, 1)
    assert gs.shape == (x.shape[1],) and np.isclose(gs[a["shift"]], a["share"])


def test_air_curve_no_air_is_nan():
    x = np.ones((40, 48, 56), np.float32)
    assert np.isnan(G.air_curve(x, np.ones(x.shape, bool), 1)).all()
