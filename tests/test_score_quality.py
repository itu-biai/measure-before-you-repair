"""Task 4 Stage 2 (D158): image quality against the clean reference — PSNR and SSIM inside the brain (D42's
definitions) and LPIPS on axial slices, averaged over the brain. Synthetic arrays; no dataset file is read."""
import numpy as np
import pytest

from bamr.eval.quality import Lpips, quality


@pytest.fixture(scope="module")
def lp():
    return Lpips()


def _head(seed=0):
    rng = np.random.default_rng(seed)
    ref = np.zeros((64, 72, 12), np.float32)
    ref[8:56, 10:62, 2:10] = 0.6 + 0.4 * rng.random((48, 52, 8), np.float32)
    brain = np.zeros(ref.shape, bool)
    brain[12:52, 14:58, 3:9] = True
    return ref, brain


def test_identical_is_perfect(lp):
    ref, brain = _head()
    q = quality(ref, ref.copy(), brain, lp)
    assert q["psnr"] == float("inf") and q["ssim"] == pytest.approx(1.0) and q["lpips"] == pytest.approx(0, abs=1e-6)


def test_more_damage_scores_worse_on_all_three(lp):
    ref, brain = _head()
    rng = np.random.default_rng(1)
    n = rng.standard_normal(ref.shape).astype(np.float32)
    a, b = quality(ref, ref + 0.03 * n, brain, lp), quality(ref, ref + 0.15 * n, brain, lp)
    assert a["psnr"] > b["psnr"] and a["ssim"] > b["ssim"] and a["lpips"] < b["lpips"]


def test_lpips_ignores_slices_without_brain(lp):
    ref, brain = _head()
    out = ref + 0.05 * np.random.default_rng(2).standard_normal(ref.shape).astype(np.float32)
    junk = out.copy()
    junk[:, :, 0] = 5.0  # axial slice 0 holds no brain
    assert lp(ref, junk, brain) == pytest.approx(lp(ref, out, brain))
