"""Task 2 Stage 2: the spike tool (D96) on synthetic phantoms (no dataset file is read)."""
import json
import time

import numpy as np
import pytest

import bamr.tools.spike as S
from bamr.tools.spike import spike_tool
from phantoms import (add_spikes, air_zeros, dim_every_nth, head_phantom, magnitude, rayleigh_air, slow_drift,
                      smooth_bias, true_stripe_rms)

X0 = head_phantom()


def near(k, k0, tol=1.0):
    k, k0 = np.asarray(k, float), np.asarray(k0, float)
    return np.abs(k - k0).max() <= tol or np.abs(k + k0).max() <= tol


def spiked(ks, amps, x0=X0, seed=1):
    return magnitude(add_spikes(x0, ks, amps), 0.02, seed)


@pytest.mark.parametrize("k0", [(5, -15, 12), (5.5, -15.5, 12.5), (20.3, 9.7, -17.4)])
def test_spike_found_and_sized(k0):
    """On a grid point and between grid points: found within 1 sample, value within 15 % (before-doc §12)."""
    f = spike_tool(spiked([k0], [0.035]))
    sp = f.evidence["spikes"]
    assert f.present is True and len(sp) == 1 and near(sp[0]["k"], k0)
    assert abs(f.value / true_stripe_rms(X0, [k0], [0.035]) - 1) <= 0.15


def test_spike_two_spikes_beat_is_echo():
    k1, k2 = (12, -20, 15), (-6, 4, 25)
    f = spike_tool(spiked([k1, k2], [0.04, 0.04]))
    sp, ec = f.evidence["spikes"], f.evidence["echoes"]
    beat = np.subtract(k1, k2)
    assert len(sp) == 2 and any(near(s["k"], k1) for s in sp) and any(near(s["k"], k2) for s in sp)
    assert not any(near(s["k"], beat) for s in sp)
    assert any(near(e["k"], beat) and e["reason"] == "stripe in the air" for e in ec)


def test_spike_strong_products_are_echoes():
    """|1 + a·e^{iθ}| holds a weak wave at 2k: listed as an echo, never as a second spike (plan Review Focus 3)."""
    k = (8, -12, 10)
    f = spike_tool(spiked([k], [0.15]))
    assert len(f.evidence["spikes"]) == 1 and near(f.evidence["spikes"][0]["k"], k)
    assert not any(near(s["k"], np.multiply(2, k)) for s in f.evidence["spikes"])


@pytest.mark.parametrize("kind", ["noise", "bias", "drift", "ghosting"])
def test_spike_none_on(kind):
    c = {"noise": X0, "bias": smooth_bias(X0), "drift": slow_drift(X0), "ghosting": dim_every_nth(X0, 3)}[kind]
    f = spike_tool(magnitude(c, 0.02, 2))
    assert f.present is False and f.evidence["spikes"] == [] and f.value == 0.0


def test_spike_zero_air_invariant():
    x = air_zeros(spiked([(5, -15, 12)], [0.035]), X0)
    a, b = spike_tool(x), spike_tool(rayleigh_air(x))
    assert a.present is True and a.present == b.present
    assert abs(b.value / a.value - 1) < 0.01


def test_spike_deterministic():
    x = spiked([(5, -15, 12)], [0.035])
    assert spike_tool(x).to_json() == spike_tool(x.copy()).to_json()


def test_spike_field_types():
    f = spike_tool(spiked([(5, -15, 12)], [0.035]))
    d = json.loads(f.to_json())
    assert d["tool"] == "spike" and d["version"] == 1 and isinstance(f.present, bool) and isinstance(f.value, float)
    assert f.confidence in ("high", "low") and f.why.startswith("1 bright k-space point")
    assert set(d["evidence"]) == {"spikes", "echoes", "line", "best_z", "raw", "scale", "head_voxels"}
    s = d["evidence"]["spikes"][0]
    assert set(s) == {"k", "z", "stripe_mm", "direction", "amplitude", "air_share"}
    assert all(isinstance(v, int) for v in s["k"]) and len(s["direction"]) == 3


def test_spike_picture_written(tmp_path):
    f = spike_tool(spiked([(5, -15, 12)], [0.035]), picture=tmp_path / "s.png")
    assert (tmp_path / "s.png").stat().st_size > 1000 and f.picture == "s.png"


def test_spike_negative_values_ok():
    x = spiked([(5, -15, 12)], [0.035])
    x[:5] = -0.05
    f = spike_tool(x)
    assert f.present is True and f.value > 0


@pytest.mark.parametrize("bad", ["zeros", "constant", "nan", "2d", "small"])
def test_spike_graceful(bad):
    nan = X0.astype(np.float32)
    nan[48, 60, 60] = np.nan
    x = {"zeros": np.zeros((96, 120, 120), np.float32), "constant": np.full((96, 120, 120), 0.7, np.float32),
         "nan": nan, "2d": X0[:, :, 60], "small": np.ones((16, 16, 16), np.float32)}[bad]
    f = spike_tool(x)
    assert f.present is None and f.value is None and f.confidence == "low" and f.why


def test_spike_confidence_rule():
    assert S._confidence(7.3, 1, 0) == "low" and S._confidence(9.0, 1, 0) == "high"
    assert S._confidence(9.0, 0, 1) == "low" and S._confidence(4.0, 0, 0) == "high"


def test_spike_speed():
    x = spiked([(20, -30, 25)], [0.035], x0=head_phantom((192, 240, 240)))
    t = time.perf_counter()
    spike_tool(x)
    assert time.perf_counter() - t < 10


# ---------------------------------------------------------------- final-review fixes (M1, M6)
def test_spike_why_never_contradicts_the_line(monkeypatch):
    """M1: best z 6.96 is below the 7.0 line, so the reason must not print 7.0."""
    monkeypatch.setattr(S, "find_peaks", lambda K: (6.96, []))
    f = spike_tool(spiked([(5, -15, 12)], [0.035]))
    assert f.present is False and f.why == "no bright k-space point: best z 6.96 (line 7.0)"


def test_spike_strong_product_listed_as_echo():
    """M6 (Review Focus 3): the 2k product of a strong spike is found and listed as an echo, for the right reason."""
    k = (8, -12, 10)
    f = spike_tool(spiked([k], [0.15]))
    assert any(near(e["k"], np.multiply(2, k)) and e["reason"] == "product of found spikes"
               for e in f.evidence["echoes"])
