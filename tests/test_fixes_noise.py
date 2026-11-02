"""Task 3 Stage 2: the noise fix (NF3, D118 A) and its pretrained weights (D120) on synthetic phantoms (no dataset file
is read).

DRUNet runs for real, with the KAIR v1.0 weights in ~/.cache/bamr (scripts/get_weights.py fetches them once). The
phantom's sides are multiples of 8, as the dataset's (192, 240, 240) are; the target is the phantom before the noise,
scored over the object.
"""
import functools
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

import bamr.fixes.drunet as DR
import bamr.fixes.noise as FN
from bamr.eval.guard import blind_guard
from bamr.fixes.noise import fix_noise
from bamr.fixes.report import head_change
from bamr.tools.noise import LINE, noise_tool
from bamr.tools.regions import head_mask
from phantoms import air_zeros, head_phantom, magnitude

X0 = head_phantom((96, 120, 120))
OBJ = X0 > 0


@functools.lru_cache(maxsize=None)
def noisy():
    return magnitude(X0, 0.045, 3)


@functools.lru_cache(maxsize=None)
def fixed():
    x = noisy()
    y, rep = fix_noise(x)
    return x, y, rep


def removed(y, x, t=X0, m=OBJ):
    e = lambda a: np.sqrt(np.mean((a.astype(np.float64) - t)[m] ** 2))  # noqa: E731
    return 1 - e(y) / e(x)


def test_known_noise():
    x, y, rep = fixed()
    assert rep.changed is True
    assert removed(y, x) >= 0.5
    ev = rep.evidence
    assert ev["sigma_before"] == noise_tool(x).value >= LINE
    assert ev["sigma_after"] < LINE and ev["present_after"] is False


def test_denoised_with_the_tools_sigma(monkeypatch):
    seen = {}

    def spy(a, s, **k):
        seen["s"] = s
        return a
    monkeypatch.setattr(FN, "denoise3", spy)
    x = noisy()
    fix_noise(x)
    assert seen["s"] == noise_tool(x).value


def test_under_line_unchanged():
    x = magnitude(X0, 0.02, 2)
    y, rep = fix_noise(x)
    assert y is x and rep.changed is False and rep.change == 0.0
    assert rep.why == noise_tool(x).why and "below the 3.0 % line" in rep.why
    assert rep.evidence["present_before"] is False


def test_zero_air_stays_zero():
    x = air_zeros(fixed()[0], X0)
    y, rep = fix_noise(x)
    assert rep.changed is True and (y[x == 0] == 0).all()


@pytest.mark.parametrize("bad", ["nan", "2d", "small"])
def test_cannot_measure_unchanged(bad):
    x = {"nan": np.full((96, 120, 120), np.nan, np.float32), "2d": np.ones((120, 120), np.float32),
         "small": np.ones((20, 20, 20), np.float32)}[bad]
    y, rep = fix_noise(x)
    assert y is x and rep.changed is False and rep.why and not rep.why.startswith("fix error")


def test_error_becomes_reason(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("mps")
    monkeypatch.setattr(FN, "denoise3", boom)
    x = noisy()
    y, rep = fix_noise(x)
    assert y is x and rep.changed is False and rep.why.startswith("fix error: RuntimeError")


def test_missing_weights_clear_error(tmp_path, monkeypatch):
    with pytest.raises(FileNotFoundError, match="get_weights"):
        DR.load(tmp_path / "none.pth")
    monkeypatch.setattr(DR, "WEIGHTS", tmp_path / "none.pth")
    x = noisy()
    y, rep = fix_noise(x)
    assert y is x and rep.changed is False
    assert rep.why.startswith("fix error: FileNotFoundError") and "get_weights" in rep.why


def test_wrong_weights_never_unpickled(tmp_path, monkeypatch):
    p = tmp_path / "drunet_gray.pth"
    p.write_bytes(b"not the weights")
    monkeypatch.setattr(torch, "load", lambda *a, **k: pytest.fail("torch read a file that failed its check"))
    with pytest.raises(ValueError, match="sha256"):
        DR.load(p)


def test_weights_pinned():
    assert DR.SHA256 == "e27fb29456732c604c8ee3ac3e92ecadd2a7a4e36a8be675e92ebbf93a240de4"
    assert DR.URL == "https://github.com/cszn/KAIR/releases/download/v1.0/drunet_gray.pth"
    assert DR.sha256(DR.WEIGHTS) == DR.SHA256


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="no Apple GPU")
def test_cpu_and_mps_agree():
    x = magnitude(head_phantom((32, 48, 48)), 0.045, 4)
    a = DR.denoise3(x, 0.045, device="cpu")
    b = DR.denoise3(x, 0.045, device="mps")
    assert np.abs(a - b).max() < 1e-3


def test_sides_not_multiple_of_8():
    c = head_phantom((90, 118, 121))
    x = magnitude(c, 0.045, 5)
    y = DR.denoise3(x, 0.045)
    assert y.shape == x.shape and y.dtype == np.float32 and np.isfinite(y).all()
    assert removed(y, x, c, c > 0) >= 0.5


def test_runs_under_guard(tmp_path):
    x = fixed()[0]
    with blind_guard(tmp_path / "refs", tmp_path / "labels"):
        y, rep = fix_noise(x)
    assert rep.changed is True


def test_deterministic():
    x, y, rep = fixed()
    y2, rep2 = fix_noise(x.copy())
    assert y.tobytes() == y2.tobytes() and rep.to_json() == rep2.to_json()


def test_output_shape_dtype():
    x, y, _ = fixed()
    assert y.shape == x.shape and y.dtype == np.float32 and np.isfinite(y).all()


def test_report_words():
    x, y, rep = fixed()
    s = rep.to_json()
    d = json.loads(s)
    assert d["what"].startswith("denoised with σ") and "noise tool after" in d["what"]
    assert d["why"] == noise_tool(x).why
    assert abs(rep.change - head_change(y, x, head_mask(x))) < 1e-9 and rep.change > 0
    assert d["evidence"]["method"] == "drunet_gray x3 (KAIR v1.0)" and d["evidence"]["weights_sha256"] == DR.SHA256
    assert not any(w in s.lower() for w in ("fixed", "good", "clean"))


def test_get_weights_checks_the_hash(tmp_path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import get_weights
    dest = tmp_path / "w.pth"
    with pytest.raises(ValueError, match="sha256"):
        get_weights.get(dest, fetch=lambda url, p: Path(p).write_bytes(b"wrong"))
    assert not dest.exists() and not list(tmp_path.iterdir())
    calls = []
    assert get_weights.get(DR.WEIGHTS, fetch=lambda *a: calls.append(a)) == DR.WEIGHTS and calls == []


def test_after_reading_never_on_the_wrong_side(monkeypatch):
    """Review M2 (D99 #9): a reading just under the line is never printed as "3.00 %"."""
    from bamr.tools.finding import Finding
    reads = iter([0.041, 0.029996])
    monkeypatch.setattr(FN, "noise_tool", lambda a, p=None: (lambda v: Finding(
        "noise", 2, "noise", v >= LINE, v, "sigma", "high", f"σ {v}", {}))(next(reads)))
    monkeypatch.setattr(FN, "denoise3", lambda a, s, **k: a)
    y, rep = fix_noise(noisy())
    assert rep.changed is True and "noise tool after: 2.9996 %" in rep.what


def test_cached_phantom_is_not_a_patched_run():
    """Review M1: tests that patch the fix use the noisy phantom, never the cached fix result."""
    x, y, rep = fixed()
    assert rep.changed is True and rep.what.startswith("denoised")
