"""Task 3 Stage 1: the bias fix (N4 once, D114) on synthetic phantoms (no dataset file is read).

As in the bias tool's tests, HD-BET on a phantom means nothing, so the brain is a stand-in: the head outline moved 2
voxels in. The dataset's order (before-doc §3): the smooth field multiplies the reference's magnitude image, so the
target is that reference.
"""
import functools
import json

import numpy as np
import pytest
from scipy import ndimage as ndi

import bamr.fixes.bias as FB
import bamr.tools.bias as BI
from bamr.eval.guard import blind_guard
from bamr.fixes.bias import fix_bias
from bamr.tools.regions import head_mask
from phantoms import add_spikes, air_zeros, head_phantom, magnitude

X0 = head_phantom((192, 240, 240))
OBJ = X0 > 0
G = np.ogrid[:192, :240, :240]
U = [(g - s / 2) / (s / 2) for g, s in zip(G, X0.shape)]  # −1 … 1 along LR, AP, SI; index grows toward R, A, S
FIELD_AP = -0.35 * U[1] + 0.1 * U[2] ** 2  # log field: brighter at the back (low AP index), darker at the front


def stand_in_brain(x, protocol=None):
    return ndi.binary_erosion(head_mask(x), iterations=2)


@pytest.fixture(autouse=True)
def _brain(monkeypatch):
    monkeypatch.setattr(FB, "brain_mask", stand_in_brain)
    monkeypatch.setattr(BI, "brain_mask", stand_in_brain)


@functools.lru_cache(maxsize=None)
def biased():
    ref = magnitude(X0 + 0j, 0.02, 3)
    return (ref * np.exp(np.broadcast_to(FIELD_AP, X0.shape))).astype(np.float32), ref


@functools.lru_cache(maxsize=None)
def fixed():
    x, t = biased()
    y, rep = fix_bias(x)
    return x, t, y, rep


def removed(y, x, t, m=OBJ):
    e = lambda a: np.sqrt(np.mean((a.astype(np.float64) - t)[m] ** 2))  # noqa: E731
    return 1 - e(y) / e(x)


def test_known_field():
    x, t, y, rep = fixed()
    assert rep.changed is True
    assert removed(y, x, t) >= 0.65
    assert rep.evidence["reading_after"] < BI.LINE <= rep.evidence["reading_before"]
    assert (rep.evidence["brightest"], rep.evidence["darkest"]) == ("back", "front")


def test_reading_equals_bias_tool():
    x, _, _, rep = fixed()
    assert rep.evidence["reading_before"] == BI.bias_tool(x).evidence["raw"]


@pytest.mark.parametrize("kind", ["clean", "noise", "spike"])
def test_none_unchanged(kind):
    c = {"clean": X0, "noise": X0, "spike": add_spikes(X0, [(5, -15, 12)], [0.05])}[kind]
    x = magnitude(c, 0.03 if kind == "noise" else 0.02, 2)
    y, rep = fix_bias(x)
    assert y is x and rep.changed is False and rep.change == 0.0
    assert "N4 reads" in rep.why and "line 0.083" in rep.why


def test_brain_mean_kept():
    x, _, y, _ = fixed()
    b = stand_in_brain(x)
    assert abs(y[b].mean() / x[b].mean() - 1) < 0.05


def test_zero_air_stays_zero():
    x = air_zeros(biased()[0], X0)
    y, rep = fix_bias(x)
    assert rep.changed is True and (y[x == 0] == 0).all()


def test_no_brain_unchanged(monkeypatch):
    monkeypatch.setattr(FB, "brain_mask", lambda x, protocol=None: np.zeros(x.shape, bool))
    x = biased()[0]
    y, rep = fix_bias(x)
    assert y is x and rep.changed is False and "no brain" in rep.why


def test_error_becomes_reason(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("n4")
    monkeypatch.setattr(FB, "n4_logfield", boom)
    x = biased()[0]
    y, rep = fix_bias(x)
    assert y is x and rep.changed is False and rep.why.startswith("fix error: RuntimeError")


@pytest.mark.parametrize("bad", ["nan", "2d", "small"])
def test_cannot_measure_unchanged(bad):
    x = {"nan": np.full((96, 120, 120), np.nan, np.float32), "2d": np.ones((120, 120), np.float32),
         "small": np.ones((20, 20, 20), np.float32)}[bad]
    y, rep = fix_bias(x)
    assert y is x and rep.changed is False and rep.why and not rep.why.startswith("fix error")


def test_runs_under_guard(tmp_path):
    x = biased()[0]
    with blind_guard(tmp_path / "refs", tmp_path / "labels"):
        y, rep = fix_bias(x)
    assert rep.changed is True


def test_deterministic():
    x, _, y, rep = fixed()
    y2, rep2 = fix_bias(x.copy())
    assert y.tobytes() == y2.tobytes() and rep.to_json() == rep2.to_json()


def test_output_shape_dtype():
    x, _, y, _ = fixed()
    assert y.shape == x.shape and y.dtype == np.float32 and np.isfinite(y).all()


def test_report_words():
    _, _, _, rep = fixed()
    s = rep.to_json()
    json.loads(s)
    assert "shading divided out" in rep.what and rep.change > 0
    assert not any(w in s.lower() for w in ("fixed", "good", "clean"))


def test_nan_reading_is_not_a_bias(monkeypatch):
    """Review M2: the gate is the bias tool's own decision (present = reading ≥ line); a NaN reading is a no."""
    monkeypatch.setattr(FB, "n4_logfield", lambda x, m, v: np.full(x.shape, np.nan, np.float32))
    x = biased()[0]
    y, rep = fix_bias(x)
    assert y is x and rep.changed is False
