"""Task 2 Stage 2: the shared k-space helpers (synthetic arrays only)."""
import numpy as np

from bamr.tools.kspace import excess, line_energy, low_freq_power, periodogram, shell_z


def test_shell_z_matches_loop():
    rng = np.random.default_rng(0)
    d = rng.normal(size=(20, 20, 20)).astype(np.float32)
    shell = rng.integers(0, 6, d.shape)
    keep = rng.random(d.shape) < 0.7
    z = shell_z(d, shell, keep)
    want = np.full(d.shape, -np.inf)
    for s in range(6):
        m = keep & (shell == s)
        med = np.median(d[m])
        want[m] = (d[m] - med) / (1.4826 * np.median(np.abs(d[m] - med)))
    assert np.allclose(z, want, atol=1e-5) and z.dtype == np.float32


def test_shell_z_flat_shell_is_minus_inf():
    d = np.zeros((6, 6, 6), np.float32)
    assert np.isneginf(shell_z(d, np.zeros(d.shape, int), np.ones(d.shape, bool))).all()


def test_low_freq_power_matches_fft():
    mask = np.random.default_rng(1).random((16, 18, 20)) < 0.4
    F = np.fft.fftn(mask.astype(float))
    want = sum(abs(F[i, j, k]) ** 2 for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1))
    assert np.isclose(low_freq_power(mask), want)


def test_excess_single_point():
    K = np.zeros((20, 20, 20), complex)
    K[10, 9, 11] = 2.0
    assert np.isclose(excess(K, (10, 9, 11)), 4.0)


def test_line_energy_median_per_plane():
    K = np.ones((10, 12, 14), complex) * (np.arange(12) + 1)[None, :, None]
    K[5, 3, 7] = 1e6  # one bright sample does not move the median
    assert np.allclose(line_energy(K, axis=1, disk=64), (np.arange(12) + 1) ** 2)


def test_periodogram_finds_period():
    ky = np.arange(120) - 60
    f, pw, rz = periodogram(np.exp(0.1 * np.cos(2 * np.pi * ky / 3)))
    band = f >= 0.1
    assert abs(f[band][np.argmax(pw[band])] - 1 / 3) < 1e-9 and rz.shape == (120,)
