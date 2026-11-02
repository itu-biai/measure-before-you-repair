import numpy as np
import pytest
from scipy import ndimage as ndi

from bamr.data.noise_sigma import local_std, mad_sigma, noise_rolloff_db, resample_white_gain


def _phantom(shape=(96, 96, 24)):
    """Piecewise-constant 3-level volume (0.3 / 0.6 / 1.0) and the eroded 1.0 region."""
    v = np.full(shape, 0.3)
    v[16:80, 16:80] = 0.6
    v[28:68, 28:68] = 1.0
    return v, ndi.binary_erosion(v == 1.0, iterations=2)


@pytest.mark.parametrize("sigma", [0.01, 0.03, 0.08])
def test_mad_recovers_sigma(sigma):
    v, m = _phantom()
    noisy = v + np.random.default_rng(0).normal(0, sigma, v.shape)
    assert mad_sigma(noisy, m) == pytest.approx(sigma, rel=0.10)


@pytest.mark.parametrize("sigma", [0.01, 0.03, 0.08])
def test_local_std_recovers_sigma(sigma):
    v, m = _phantom()
    noisy = v + np.random.default_rng(0).normal(0, sigma, v.shape)
    assert local_std(noisy, m) == pytest.approx(sigma, rel=0.10)


def test_rolloff_white_is_flat():
    v, m = _phantom((96, 96, 40))
    noisy = v + np.random.default_rng(0).normal(0, 0.03, v.shape)
    assert abs(noise_rolloff_db(noisy, m)) < 1.5


def test_rolloff_lowpass_positive():
    v, m = _phantom((96, 96, 40))
    n = ndi.gaussian_filter(np.random.default_rng(0).normal(0, 0.03, v.shape), sigma=(1, 1, 0))
    assert noise_rolloff_db(v + n, m) > 10


def test_gain_identity_grid():
    assert resample_white_gain(np.eye(4), (48, 48, 48)) == pytest.approx(1.0, abs=0.02)


def test_gain_ixi_grid():
    # IXI layout: axes (P, S, R) at 0.9375 / 0.9375 / 1.2 mm
    aff = np.array([[0, 0, 1.2, 0], [-0.9375, 0, 0, 0], [0, 0.9375, 0, 0], [0, 0, 0, 1.0]])
    assert 0.6 < resample_white_gain(aff, (64, 64, 40)) < 0.95


@pytest.mark.parametrize("fn", [mad_sigma, local_std, noise_rolloff_db])
def test_too_few_raises(fn):
    v, _ = _phantom()
    m = np.zeros(v.shape, bool)
    m[40:42, 40:42, 5] = True
    with pytest.raises(ValueError):
        fn(v, m)
