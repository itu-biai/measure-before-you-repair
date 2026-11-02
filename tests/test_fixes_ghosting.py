"""Task 3 Stage 3: the ghosting fix (GF, D122 A) on synthetic phantoms (no dataset file is read).

Full size (192 × 240 × 240), as the ghosting tool's tests: its period line (150) was set there. The ghost is our
simulator's (phantoms.dim_every_nth): every n-th A–P plane of k-space dimmed by (1 − I), the centre plane kept. The
target is the phantom without the ghost, with the same noise draw where there is noise (before-doc §6).
"""
import functools
import json

import numpy as np
import pytest
from scipy import ndimage as ndi

import bamr.fixes.ghosting as FG
from bamr.eval.guard import blind_guard
from bamr.fixes.ghosting import fix_ghosting
from bamr.fixes.report import head_change
from bamr.tools.finding import PROTOCOL_V2, Finding, Protocol
from bamr.tools.ghosting import ghosting_tool
from bamr.tools.regions import head_mask
from phantoms import air_zeros, dim_every_nth, head_phantom, magnitude, periodic_modulation, smooth_phase

X0 = head_phantom((192, 240, 240))
OBJ = X0 > 0
SIGMA = 0.02


def case(n=3, I=0.3, sigma=0.0, seed=1):
    """(input, target): |ghosted| and the phantom, or both with the same complex noise draw (one cache key per case,
    however it is called: review M7)."""
    return _case(n, I, sigma, seed)


@functools.lru_cache(maxsize=None)
def _case(n, I, sigma, seed):
    c = dim_every_nth(X0, n, I)
    if not sigma:
        return np.abs(c).astype(np.float32), X0
    return magnitude(c, sigma, seed), magnitude(X0.astype(np.complex128), sigma, seed)


def fixed(n=3, I=0.3, sigma=0.0):
    return _fixed(n, I, sigma)


@functools.lru_cache(maxsize=None)
def _fixed(n, I, sigma):
    x, t = case(n, I, sigma)
    y, rep = fix_ghosting(x)
    return x, t, y, rep


def removed(y, x, t, m=OBJ):
    e = lambda a: np.sqrt(np.mean((a.astype(np.float64) - t)[m] ** 2))  # noqa: E731
    return 1 - e(y) / e(x)


@pytest.mark.parametrize("n", [2, 3, 4])
def test_known_n_and_I(n):
    x, t, y, rep = fixed(n)
    ev = rep.evidence
    assert rep.changed is True
    assert ev["period_lines"] == pytest.approx(n, abs=0.1) and ev["n"] == n
    assert ev["I"] == pytest.approx(0.3, rel=0.05)
    assert rep.what.startswith(f"every {n}{FG.ordinal(n)} A–P plane")


def test_black_air_error_falls():
    x, t, y, rep = fixed()
    assert removed(y, x, t) >= 0.9


def test_noisy_air_strength_and_tool_after():
    """The noise came after the ghost: I is still read within 10 %, and the noise is not lifted on every n-th plane,
    so the ghosting tool hears no rhythm afterwards."""
    x, t, y, rep = fixed(sigma=SIGMA)
    ev = rep.evidence
    assert ev["I"] == pytest.approx(0.3, rel=0.10)
    assert ev["ghost_after"]["present"] is False
    assert ghosting_tool(y).present is False
    assert removed(y, x, t) >= 0.5


def test_no_ghost_unchanged():
    x = magnitude(X0, SIGMA, 2)
    f = ghosting_tool(x)
    y, rep = fix_ghosting(x)
    assert f.present is False
    assert y is x and rep.changed is False and rep.change == 0.0 and rep.why == f.why


def test_period_not_whole_unchanged():
    """Periodic motion with a period of 2.6 lines (the ghosting tool's own test case): found, but not every n-th plane
    dimmed, so the model this fix inverts does not hold (before-doc §7)."""
    c = periodic_modulation(X0 * np.exp(1j * smooth_phase(X0.shape)), 2.6, depth=0.25, theta=0.0)
    x = magnitude(c, SIGMA, 3)
    f = ghosting_tool(x)
    y, rep = fix_ghosting(x)
    assert f.present is True and abs(f.evidence["period_lines"] - 2.6) < 0.1
    assert y is x and rep.changed is False
    assert rep.why.startswith(f.why) and "not a whole number" in rep.why
    assert rep.evidence["period_lines"] == pytest.approx(f.evidence["period_lines"], abs=1e-6)


def test_air_copy_only_unchanged(monkeypatch):
    """The tool found a copy in the air but no rhythm: no period, so no planes to undo."""
    monkeypatch.setattr(FG, "ghosting_tool", lambda a, p=None: Finding(
        "ghosting", 1, "ghosting", True, 0.01, "u", "low", "a copy of the head in the air", {"period_lines": None}))
    x = case()[0]
    y, rep = fix_ghosting(x)
    assert y is x and rep.changed is False and "no period" in rep.why


def test_zeros_stay_zero():
    """Exact zeros in the far air (D69) stay 0; the air nearer the head is still read."""
    x = case(sigma=SIGMA)[0].copy()
    x[ndi.distance_transform_edt(~head_mask(x)) > 30] = 0
    y, rep = fix_ghosting(x)
    assert rep.changed is True and (y[x == 0] == 0).all()
    assert rep.evidence["I"] == pytest.approx(0.3, rel=0.10)
    near = ndi.distance_transform_edt(~head_mask(x)) > 4  # review M3: the zeros are left out of the air I is read in
    assert rep.evidence["air_voxels"] == int((near & (x != 0)).sum()) < int(near.sum())


def test_air_all_zero_unchanged():
    """No air left but exact zeros: the strength cannot be read where the copies are, and zeros are not information
    (D69), so the fix declines with the reason."""
    x = air_zeros(case(sigma=SIGMA)[0], X0)
    f = ghosting_tool(x)
    y, rep = fix_ghosting(x)
    assert f.present is True
    assert y is x and rep.changed is False and "exact zeros" in rep.why


@pytest.mark.parametrize("bad", ["nan", "2d", "small"])
def test_cannot_measure_unchanged(bad):
    x = {"nan": np.full((96, 120, 120), np.nan, np.float32), "2d": np.ones((120, 120), np.float32),
         "small": np.ones((20, 20, 20), np.float32)}[bad]
    y, rep = fix_ghosting(x)
    assert y is x and rep.changed is False and rep.why and not rep.why.startswith("fix error")


def test_error_becomes_reason(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("fft")
    monkeypatch.setattr(FG, "strength", boom)
    x = case()[0]
    y, rep = fix_ghosting(x)
    assert y is x and rep.changed is False and rep.why.startswith("fix error: RuntimeError")


def test_protocol_axis():
    """A–P read from the protocol: the same volume with its first two axes swapped gives the same answer swapped."""
    x, _, y, rep = fixed()
    p = Protocol(readout="SI", phase_slow="AP", phase_fast="LR", array_axes=("AP", "LR", "SI"), voxel_mm=1.0, wm=1.0,
                 orientation="ARS")
    y2, rep2 = fix_ghosting(np.ascontiguousarray(x.swapaxes(0, 1)), p)
    assert rep2.changed is True and rep2.evidence["n"] == rep.evidence["n"]
    assert rep2.evidence["I"] == pytest.approx(rep.evidence["I"], abs=1e-4)
    assert np.abs(y2.swapaxes(0, 1) - y).max() < 1e-3


def test_runs_under_guard(tmp_path):
    x = case()[0]
    with blind_guard(tmp_path / "refs", tmp_path / "labels"):
        y, rep = fix_ghosting(x)
    assert rep.changed is True


def test_deterministic():
    x, _, y, rep = fixed()
    y2, rep2 = fix_ghosting(x.copy())
    assert y.tobytes() == y2.tobytes() and rep.to_json() == rep2.to_json()


def test_output_shape_dtype():
    x, _, y, _ = fixed()
    assert y.shape == x.shape and y.dtype == np.float32 and np.isfinite(y).all() and (y >= 0).all()
    assert y.flags.c_contiguous


def test_input_not_changed():
    x = case(2)[0]
    keep = x.copy()
    fix_ghosting(x)
    assert x.tobytes() == keep.tobytes()


def test_report_words():
    x, _, y, rep = fixed(sigma=SIGMA)
    s = rep.to_json()
    d = json.loads(s)
    assert d["why"] == ghosting_tool(x).why
    assert "the missing signal put back" in d["what"] and "ghosting tool after: no" in d["what"]
    assert abs(rep.change - head_change(y, x, head_mask(x))) < 1e-9 and rep.change > 0
    ev = d["evidence"]
    assert {"period_lines", "n", "planes", "I", "sigma_air", "ghost_after", "motion_after", "method"} <= set(ev)
    assert set(ev["ghost_after"]) >= {"present", "value"} and set(ev["motion_after"]) >= {"present", "value"}
    assert not any(w in s.lower() for w in ("fixed", "good", "clean", "corrected"))


def test_default_protocol():
    assert FG.fix_ghosting.__defaults__ == (PROTOCOL_V2,)


def test_ordinal():
    assert [FG.ordinal(n) for n in (2, 3, 4, 5, 10)] == ["nd", "rd", "th", "th", "th"]
