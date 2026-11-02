"""Task 2 Stage 3: the bias tool (D102) on synthetic phantoms (no dataset file is read).

HD-BET on a phantom means nothing, so the brain is a stand-in: the head outline moved 2 voxels in. On a phantom N4
recovers 0.87–0.96 of a smooth field (even the dataset's own order-3 fields); on real brains 0.6, which the one scale
factor absorbs. So these tests check N4's reading against the true field; the value is checked on the dev run.
"""
import functools
import json
import time

import numpy as np
import pytest
from scipy import ndimage as ndi

import bamr.tools.bias as BI
from bamr.tools.bias import bias_tool
from bamr.tools.regions import head_mask
from phantoms import add_spikes, air_zeros, head_phantom, magnitude, rayleigh_air

X0 = head_phantom((192, 240, 240))
G = np.ogrid[:192, :240, :240]
U = [(g - s / 2) / (s / 2) for g, s in zip(G, X0.shape)]  # −1 … 1 along LR, AP, SI; index grows toward R, A, S
FIELDS = {"ap": lambda: -0.35 * U[1] + 0.1 * U[2] ** 2,  # brighter at the back (low AP index), darker at the front
          "lr": lambda: 0.3 * U[0] + 0 * U[1]}  # brighter at the right


def stand_in_brain(x, protocol=None):
    return ndi.binary_erosion(head_mask(x), iterations=2)


@pytest.fixture(autouse=True)
def _brain(monkeypatch):
    monkeypatch.setattr(BI, "brain_mask", stand_in_brain)


@functools.lru_cache(maxsize=None)
def biased(kind="ap", sigma=0.02, seed=3):
    f = np.exp(np.broadcast_to(FIELDS[kind](), X0.shape))
    obj = X0 > 0
    return magnitude(X0 * f, sigma, seed), float(f[obj].max() / f[obj].min() - 1)


@pytest.mark.parametrize("kind,bright,dark", [("ap", "P", "A"), ("lr", "R", "L")])
def test_bias_known_field(kind, bright, dark):
    x, truth = biased(kind)
    f = bias_tool(x)
    assert f.present is True
    assert abs(f.evidence["raw"] / truth - 1) < 0.15
    assert f.value == round(f.evidence["raw"] / BI.SCALE, 6)
    assert (f.evidence["bright_side"], f.evidence["dark_side"]) == (bright, dark)


def test_bias_why_names_sides():
    f = bias_tool(biased("ap")[0])
    assert "brightest at the back" in f.why and "darkest at the front" in f.why and "line 0.083" in f.why


@pytest.mark.parametrize("kind", ["clean", "noise", "spike"])
def test_bias_none_on(kind):
    c = {"clean": X0, "noise": X0, "spike": add_spikes(X0, [(5, -15, 12)], [0.05])}[kind]
    f = bias_tool(magnitude(c, 0.03 if kind == "noise" else 0.02, 2))
    assert f.present is False and f.evidence["raw"] < BI.LINE


def test_bias_zero_air_invariant():
    x = air_zeros(biased("ap")[0].copy(), X0)
    a, b = bias_tool(x), bias_tool(rayleigh_air(x))
    assert a.present is True and a.present == b.present
    assert abs(b.value / a.value - 1) < 0.05


def test_bias_no_brain(monkeypatch):
    x = biased("ap")[0]
    monkeypatch.setattr(BI, "brain_mask", lambda x, protocol=None: np.zeros(x.shape, bool))
    f = bias_tool(x)
    assert f.present is None and f.value is None and "no brain" in f.why
    small = np.zeros(x.shape, bool)
    small[90:100, 110:130, 110:130] = True  # 4 ml
    monkeypatch.setattr(BI, "brain_mask", lambda x, protocol=None: small)
    f = bias_tool(x)
    assert f.present is None and "4 ml" in f.why


@pytest.mark.parametrize("bad", ["zeros", "constant", "nan", "2d", "small"])
def test_bias_graceful(bad):
    nan = head_phantom().astype(np.float32)
    nan[48, 60, 60] = np.nan
    x = {"zeros": np.zeros((96, 120, 120), np.float32), "constant": np.full((96, 120, 120), 0.7, np.float32),
         "nan": nan, "2d": nan[:, :, 60], "small": np.ones((16, 16, 16), np.float32)}[bad]
    f = bias_tool(x)
    assert f.present is None and f.value is None and f.confidence == "low" and f.why


def test_bias_deterministic():
    x = biased("ap")[0]
    assert bias_tool(x).to_json() == bias_tool(x.copy()).to_json()


def test_bias_field_types():
    f = bias_tool(biased("ap")[0])
    d = json.loads(f.to_json())
    assert d["tool"] == "bias" and d["version"] == 1 and d["artifact"] == "bias" and isinstance(f.present, bool)
    assert isinstance(f.value, float) and f.confidence in ("high", "low") and f.why.startswith("smooth shading")
    assert set(d["evidence"]) == {"raw", "line", "scale", "brain_ml", "bright_side", "dark_side", "bright_offset_mm",
                                  "dark_offset_mm"}


def test_bias_picture_written(tmp_path):
    f = bias_tool(biased("ap")[0], picture=tmp_path / "b.png")
    assert (tmp_path / "b.png").stat().st_size > 1000 and f.picture == "b.png"


def test_bias_negative_values_ok():
    x = biased("ap")[0].copy()
    x[:5] = -0.05
    f = bias_tool(x)
    assert f.present is True and f.value > 0


def test_bias_confidence_rule():
    assert BI._confidence(0.085) == "low" and BI._confidence(0.078) == "low"  # within ±10 % of 0.083
    assert BI._confidence(0.20) == "high" and BI._confidence(0.05) == "high"


def test_bias_why_never_contradicts_the_line(monkeypatch):
    """Next to the line, the reason prints enough digits to stay on the side the decision is on (D99 #9)."""
    real = BI.n4_logfield
    monkeypatch.setattr(BI, "n4_logfield", lambda x, m, v: real(x, m, v) * 0 + np.log1p(0.08299) * (U[1] > 0))
    f = bias_tool(biased("ap")[0])
    assert f.present is False and "0.08299 (line 0.083)" in f.why


def test_bias_n4_speed():
    x = biased("ap")[0]
    t = time.perf_counter()
    bias_tool(x)
    assert time.perf_counter() - t < 8  # the brain (HD-BET, ≈ 11 s on the GPU) comes on top, measured on the dev run
