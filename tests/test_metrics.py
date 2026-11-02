import numpy as np
import pytest

from bamr.data.metrics import masked_psnr, masked_ssim


def _pair():
    rng = np.random.default_rng(0)
    ref = rng.uniform(0.2, 1.2, (240, 192)).astype(np.float32)
    brain = np.zeros(ref.shape, bool)
    brain[40:200, 30:160] = True
    return ref, brain


def test_psnr_ssim():
    ref, brain = _pair()
    assert masked_psnr(ref, ref, brain) == np.inf
    assert masked_ssim(ref, ref, brain) == pytest.approx(1.0)
    d = 0.05
    rng_ = float(ref[brain].max())
    assert masked_psnr(ref, ref + d, brain) == pytest.approx(20 * np.log10(rng_ / d), rel=1e-5)
    assert masked_ssim(ref, ref + 0.3 * np.random.default_rng(1).normal(size=ref.shape), brain) < 0.9
