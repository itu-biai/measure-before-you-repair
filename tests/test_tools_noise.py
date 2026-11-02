"""Task 2 Stage 1: the noise tool, option A (D71), on synthetic phantoms (no dataset file is read)."""
import json
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

import bamr.tools.noise as N
from bamr.tools.noise import noise_tool

REPO = Path(__file__).resolve().parents[1]


def ellipsoid(shape=(96, 120, 120), frac=0.42):
    g = np.ogrid[tuple(slice(0, d) for d in shape)]
    r = np.sqrt(sum(((gi - d / 2) / (d * frac)) ** 2 for gi, d in zip(g, shape)))
    return (r < 1).astype(np.float32)


def noisy(base, sigma, seed=0, rician=False):
    rng = np.random.default_rng(seed)
    x = base + rng.normal(0, sigma, base.shape)
    if rician:
        x = np.sqrt(x ** 2 + rng.normal(0, sigma, base.shape) ** 2)
    return x.astype(np.float32)


@pytest.fixture
def unit_scale(monkeypatch):
    monkeypatch.setattr(N, "SCALE", 1.0)


@pytest.mark.parametrize("rician", [False, True])
@pytest.mark.parametrize("sigma,tol", [(0.02, 0.05), (0.03, 0.05), (0.045, 0.05), (0.06, 0.08)])
def test_noise_known_sigma(unit_scale, sigma, tol, rician):
    f = noise_tool(noisy(ellipsoid(), sigma, rician=rician))
    assert abs(f.value / sigma - 1) <= tol  # 6 %: the ±15 % flat-tissue rule clips the tails (plan Review Focus 3)


def test_noise_ramp_edges(unit_scale):
    base = ellipsoid()
    ramp = np.linspace(0.92, 1.08, base.shape[0], dtype=np.float32)[:, None, None]
    base = base * ramp
    base[:, 50, :] += 0.08 * (base[:, 50, :] > 0)
    base[:, 70, :] += 0.08 * (base[:, 70, :] > 0)
    f = noise_tool(noisy(base, 0.03))
    assert abs(f.value / 0.03 - 1) <= 0.05


def test_noise_zero_air_invariant(unit_scale):
    obj = ellipsoid()
    x = noisy(obj, 0.04) * (obj > 0)  # exact-zero air, like IXI
    rng = np.random.default_rng(1)
    z = x == 0
    y = x.copy()
    y[z] = 0.005 * np.hypot(rng.standard_normal(z.sum()), rng.standard_normal(z.sum()))
    a, b = noise_tool(x), noise_tool(y)
    assert a.present == b.present
    assert abs(b.value / a.value - 1) < 0.01


def test_noise_deterministic():
    x = noisy(ellipsoid(), 0.04)
    assert noise_tool(x).to_json() == noise_tool(x.copy()).to_json()


def test_noise_empty_graceful():
    f = noise_tool(np.zeros((60, 70, 80), np.float32))
    assert f.present is None and f.value is None
    assert f.confidence == "low" and "tissue" in f.why


def test_noise_tiny_head_graceful():
    g = np.ogrid[:60, :70, :80]
    ball = ((g[0] - 30) ** 2 + (g[1] - 35) ** 2 + (g[2] - 40) ** 2 < 36).astype(np.float32)
    f = noise_tool(noisy(ball, 0.03))
    assert f.present is None and f.confidence == "low"


@pytest.mark.parametrize("bad", ["nan", "2d"])
def test_noise_nonfinite_graceful(bad):
    x = noisy(ellipsoid(), 0.03)
    if bad == "nan":
        x[48, 60, 60] = np.nan
    else:
        x = x[:, :, 60]
    f = noise_tool(x)
    assert f.present is None and f.confidence == "low" and "finite 3-D" in f.why


def test_noise_negative_values_ok():
    x = noisy(ellipsoid(), 0.04)
    x[:5] = -0.05
    f = noise_tool(x)
    assert f.present is True and f.value > 0


def test_noise_near_line_low_confidence(unit_scale):
    f = noise_tool(noisy(ellipsoid(), 0.0305 / 0.984))  # the phantom reads ~1.6 % low at this σ
    assert abs(f.value - 0.030) < N.NEAR and f.confidence == "low"


def test_noise_line_matches_config():
    cfg = yaml.safe_load((REPO / "configs" / "dataset_v2.yaml").read_text())
    assert N.LINE == cfg["stage2"]["noise"]["sigma_total"][0][0] == cfg["stage3"]["max_native_sigma"]


def test_noise_finding_value_not_level():
    """D95: σ and yes/no for the agent, no level name; the reason states σ."""
    d = json.loads(noise_tool(noisy(ellipsoid(), 0.04)).to_json())
    assert "level" not in d and d["present"] is True and d["version"] == 2
    assert d["why"].startswith("σ ") and "% of WM" in d["why"]


def test_noise_speed():
    x = noisy(ellipsoid((192, 240, 240)), 0.04)
    t = time.perf_counter()
    noise_tool(x)
    assert time.perf_counter() - t < 10


def test_noise_picture_written(tmp_path):
    f = noise_tool(noisy(ellipsoid(), 0.04), picture=tmp_path / "p.png")
    assert (tmp_path / "p.png").stat().st_size > 1000 and f.picture == "p.png"


def test_tools_blind_static():
    src = REPO / "src" / "bamr" / "tools"
    for p in src.glob("*.py"):
        text = p.read_text()
        for word in ("refs", "labels", "blindio", "nibabel", "manifest"):
            assert word not in text, f"{p.name} mentions {word}"


# ---------------------------------------------------------------- final-review fixes (Important #1, #2, #3, #8)
from bamr.tools.finding import Protocol  # noqa: E402


def test_noise_strong_noise_says_so():
    f = noise_tool(noisy(ellipsoid(), 0.15))
    assert f.present is None and "very strong" in f.why
    assert f.evidence["neighbour_sigma"] > 0.1


def test_noise_wrong_scale_says_so():
    f = noise_tool(noisy(ellipsoid(), 0.03) * 1000)
    assert f.present is None and "not at 1" in f.why


@pytest.mark.parametrize("bad", ["thin", "empty", "object", "complex", "axes"])
def test_noise_odd_inputs_graceful(bad):
    x, p = noisy(ellipsoid(), 0.04), N.PROTOCOL_V2
    if bad == "thin":
        x = x[46:50]
    elif bad == "empty":
        x = np.zeros((0, 240, 240), np.float32)
    elif bad == "object":
        x = x.astype(object)
    elif bad == "complex":
        x = x.astype(np.complex64)
    else:
        p = Protocol("SI", "AP", "LR", ("RL", "PA", "IS"), 1.0, 1.0)
    f = noise_tool(x, protocol=p)
    assert f.present is None and f.confidence == "low" and f.why


def test_noise_internal_error_graceful(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(N, "tile_sigma", boom)
    f = noise_tool(noisy(ellipsoid(), 0.04))
    assert f.present is None and "RuntimeError" in f.why


def test_noise_zero_raw_no_crash(monkeypatch):
    monkeypatch.setattr(N, "tile_sigma", lambda *a, **k: (0.0, 5000))
    f = noise_tool(noisy(ellipsoid(), 0.04))
    assert f.present is False and f.evidence["fineness"] is None


def test_noise_picture_new_folder(tmp_path):
    p = tmp_path / "a" / "b" / "p.png"
    noise_tool(noisy(ellipsoid(), 0.04), picture=p)
    assert p.stat().st_size > 1000


# ---------------------------------------------------------------- Stage 1 leftovers (D99)
def test_noise_present_on_reported_value(monkeypatch):
    """#9: the yes/no follows the value the agent reads: 0.0299996 prints as 0.03, so present."""
    monkeypatch.setattr(N, "tile_sigma", lambda *a, **k: (0.0299996 * N.SCALE, 5000))
    d = json.loads(noise_tool(noisy(ellipsoid(), 0.03)).to_json())
    assert d["value"] == 0.03 and d["present"] is True


def test_noise_constant_volume_graceful():
    """#10: a constant volume has no head; a reason, not a crash."""
    f = noise_tool(np.full((96, 120, 120), 0.7, np.float32))
    assert f.present is None and f.value is None and "no head" in f.why


def test_noise_field_types():
    """#10: field types on a real finding, and the evidence survives strict JSON."""
    f = noise_tool(noisy(ellipsoid(), 0.04))
    assert isinstance(f.present, bool) and isinstance(f.value, float) and f.confidence in ("high", "low")
    assert isinstance(f.why, str) and f.why and isinstance(f.evidence, dict)
    assert json.loads(f.to_json())["evidence"].keys() == f.evidence.keys()


def test_tools_do_not_import_pyplot():
    """#10: plotting is set up where pictures are drawn; importing a tool leaves matplotlib's global state alone."""
    import subprocess
    import sys
    mods = ["bamr.tools.noise", "bamr.tools.picture", "bamr.tools.spike", "bamr.tools.ghosting", "bamr.tools.brain",
            "bamr.tools.bias", "bamr.tools.motion"]
    code = f"import sys; import {', '.join(mods)}; assert 'matplotlib.pyplot' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)
