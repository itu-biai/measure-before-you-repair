"""Task 3 Stage 1: the spike fix (SE, D113) on synthetic phantoms (no dataset file is read).

The target is the phantom's own magnitude without the spike, with the same noise draw: what the dataset scores
against (the label replayed without the spike). Removed = 1 − error after / error before, over the object.
"""
import functools
import json

import numpy as np
import pytest

import bamr.fixes.spike as FS
from bamr.eval.guard import blind_guard
from bamr.fixes.spike import fix_spike, se_repair
from bamr.tools.finding import Finding
from bamr.tools.regions import head_mask
from phantoms import add_spikes, air_zeros, head_phantom, magnitude

X0 = head_phantom()
OBJ = X0 > 0
K1, K2 = (5, -15, 12), (20, 9, -17)


def removed(y, x, t, m=OBJ):
    e = lambda a: np.sqrt(np.mean((a.astype(np.float64) - t)[m] ** 2))  # noqa: E731
    return 1 - e(y) / e(x)


def spike_on(ref, ks):
    """The dataset's order (before-doc §3): the spike on the reference's magnitude image, |ref + Z|."""
    return np.abs(add_spikes(ref, ks, [0.035] * len(ks))).astype(np.float32)


@functools.lru_cache(maxsize=None)
def case(ks=(K1,), noise_after=None):
    """Input and target. The reference has its own faint Rician air (σ 0.002); with `noise_after`, noise is added
    after the spike, as in a mixed volume, and the target is the reference with the same noise draw."""
    ref = magnitude(X0 + 0j, 0.002, 1)
    x = spike_on(ref, ks)
    if noise_after:
        return magnitude(x + 0j, noise_after, 2), magnitude(ref + 0j, noise_after, 2)
    return x, ref


@functools.lru_cache(maxsize=None)
def fixed(ks=(K1,), noise_after=None):
    x, t = case(ks, noise_after)
    y, rep = fix_spike(x)
    return x, t, y, rep


def near(k, k0, tol=1):
    k, k0 = np.asarray(k), np.asarray(k0)
    return np.abs(k - k0).max() <= tol or np.abs(k + k0).max() <= tol


def test_one_stripe_quiet_air():
    x, t, y, rep = fixed()
    assert rep.changed is True
    assert removed(y, x, t) >= 0.95
    assert rep.evidence["rounds"][0]["air"] == "quiet"
    assert rep.evidence["spike_tool_after"]["present"] is False


def test_two_stripes():
    x, t, y, rep = fixed((K1, K2))
    assert removed(y, x, t) >= 0.90
    done = [k for r in rep.evidence["rounds"] for k in r["k"]]
    assert all(any(near(k, k0) for k in done) for k0 in (K1, K2))


def test_noisy_air():
    x, t, y, rep = fixed(noise_after=0.02)
    assert rep.evidence["rounds"][0]["air"] == "noisy"
    assert removed(y, x, t) >= 0.80
    from scipy import ndimage as ndi
    far = ndi.distance_transform_edt(~head_mask(x)) > 8
    assert removed(y, x, t, far) >= 0


def test_no_spike_unchanged():
    x = magnitude(X0 + 0j, 0.02, 1)
    y, rep = fix_spike(x)
    assert y is x and rep.changed is False and rep.change == 0.0
    assert "no bright k-space point" in rep.why


def test_echo_only_unchanged(monkeypatch):
    f = Finding("spike", 1, "spike", True, 0.01, "u", "low", "bright k-space point z 9.0 (line 7.0), but its "
                "stripe sits in the air", {"spikes": [], "echoes": [{"k": [1, 2, 3]}], "best_z": 9.0})
    monkeypatch.setattr(FS, "spike_tool", lambda x, protocol=None: f)
    x = case()[0]
    y, rep = fix_spike(x)
    assert y is x and rep.changed is False and "only echoes" in rep.why


@pytest.mark.parametrize("bad", ["nan", "2d", "small", "zeros"])
def test_cannot_measure_unchanged(bad):
    x = {"nan": np.full((96, 120, 120), np.nan, np.float32), "2d": np.ones((120, 120), np.float32),
         "small": np.ones((20, 20, 20), np.float32), "zeros": np.zeros((96, 120, 120), np.float32)}[bad]
    y, rep = fix_spike(x)
    assert y is x and rep.changed is False and rep.why


def test_error_becomes_reason(monkeypatch):
    def boom(*a, **k):
        raise MemoryError("x")
    monkeypatch.setattr(FS, "se_repair", boom)
    x = case()[0]
    y, rep = fix_spike(x)
    assert y is x and rep.changed is False and rep.why.startswith("fix error: MemoryError")


def test_zero_air_stays_zero():
    x0, t = case()
    x = air_zeros(x0, X0)
    y, rep = fix_spike(x)
    assert rep.changed is True
    assert (y[x == 0] == 0).all()
    assert removed(y, x, t) >= 0.90


def test_no_far_air():
    """Every voxel within 8 of the head: the head fit alone, still repaired."""
    x0 = np.zeros((96, 120, 120))
    x0[4:-4, 4:-4, 4:-4] = 1.0
    t = magnitude(x0 + 0j, 0.002, 1)
    x = spike_on(t, [K1])
    y, info = se_repair(x, [list(K1)], head_mask(x), x == 0)
    assert info["mode"] == "no far air"
    assert removed(y, x, t, x0 > 0) >= 0.90


def test_runs_under_guard(tmp_path):
    x = case()[0]
    with blind_guard(tmp_path / "refs", tmp_path / "labels"):
        y, rep = fix_spike(x)
    assert rep.changed is True


def test_deterministic():
    x, _, y, rep = fixed()
    y2, rep2 = fix_spike(x.copy())
    assert y.tobytes() == y2.tobytes() and rep.to_json() == rep2.to_json()


def test_output_shape_dtype():
    x, _, y, _ = fixed()
    assert y.shape == x.shape and y.dtype == np.float32 and np.isfinite(y).all() and (y >= 0).all()


def test_report_words():
    _, _, _, rep = fixed()
    s = rep.to_json()
    json.loads(s)
    assert "stripe" in rep.what and rep.change > 0
    assert not any(w in s.lower() for w in ("fixed", "good", "clean"))
