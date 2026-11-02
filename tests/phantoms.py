"""Synthetic head phantoms shared by the tool tests (no dataset file is read)."""
import numpy as np


def head_phantom(shape=(96, 120, 120), seed=0) -> np.ndarray:
    """An ellipsoid of 1.0 (semi-axes 0.38 × shape) holding six blobs at 0.7 or 1.25: some structure, like a head."""
    rng = np.random.default_rng(seed)
    g = np.ogrid[tuple(slice(0, s) for s in shape)]
    c = [s / 2 for s in shape]
    x = np.where(sum(((gi - ci) / (0.38 * s)) ** 2 for gi, ci, s in zip(g, c, shape)) < 1, 1.0, 0.0)
    for _ in range(6):
        cc = [ci + rng.uniform(-0.2, 0.2) * s for ci, s in zip(c, shape)]
        r = sum(((gi - ci) / (rng.uniform(0.05, 0.12) * s)) ** 2 for gi, ci, s in zip(g, cc, shape))
        x = np.where(r < 1, rng.choice([0.7, 1.25]), x)
    return x


def magnitude(c, sigma=0.02, seed=1) -> np.ndarray:
    """|c + complex Gaussian noise|, as a scanner delivers a magnitude image (Rician)."""
    rng = np.random.default_rng(seed)
    return np.abs(c + sigma * (rng.standard_normal(c.shape) + 1j * rng.standard_normal(c.shape))).astype(np.float32)


def _grid(shape):
    return np.ogrid[tuple(slice(0, s) for s in shape)]


def add_spikes(x0, ks, amps, phase=0.7) -> np.ndarray:
    """x0 + Σ a·exp(i(2π k·r/N + phase)): each spike is one k-space point, a complex stripe wave over the whole box,
    before the magnitude is taken (as TorchIO and a scanner make it)."""
    g = _grid(x0.shape)
    c = x0.astype(np.complex128)
    for k, a in zip(ks, amps):
        c = c + a * np.exp(1j * (2 * np.pi * sum(kk * gi / s for kk, gi, s in zip(k, g, x0.shape)) + phase))
    return c


def true_stripe_rms(x0, ks, amps) -> float:
    """The label's definition: RMS over the head of |spiked| − clean, without noise."""
    return float(np.sqrt(np.mean((np.abs(add_spikes(x0, ks, amps)) - x0)[x0 > 0] ** 2)))


def kspace_lines(c, m, axis=1) -> np.ndarray:
    """Multiply the k-space planes along `axis` (A–P) by m(ky); complex image back."""
    ky = np.fft.fftfreq(c.shape[axis]) * c.shape[axis]
    shape = [1, 1, 1]
    shape[axis] = -1
    return np.fft.ifftn(np.fft.fftn(c) * np.asarray(m(ky)).reshape(shape))


def dim_every_nth(x0, n, depth=0.2, axis=1) -> np.ndarray:
    """Every n-th A–P line weaker, the centre line kept (our simulator's ghosting, TorchIO with restore=None):
    copies at FOV / n. Keeping the centre sets the copies on a positive base in the air, as in the dataset."""
    return kspace_lines(x0, lambda ky: np.where((np.round(ky) % n == 0) & (np.round(ky) != 0), 1 - depth, 1.0), axis)


def slow_drift(x0, axis=1) -> np.ndarray:
    """A slow change over the A–P lines (period 60 lines): no ghost, a slight blur."""
    return kspace_lines(x0, lambda ky: 1 + 0.2 * np.cos(2 * np.pi * ky / 60), axis)


def smooth_bias(x0) -> np.ndarray:
    return x0 * (1 + 0.2 * np.linspace(-1, 1, x0.shape[0])[:, None, None])


def air_zeros(x, x0, gap=3) -> np.ndarray:
    """Exact zeros in the air more than `gap` voxels from the head, as in IXI (D69)."""
    from scipy import ndimage as ndi
    y = x.copy()
    y[ndi.distance_transform_edt(x0 <= 0) > gap] = 0
    return y


def rayleigh_air(x, seed=5) -> np.ndarray:
    """The exact zeros replaced by faint Rayleigh noise (σ 0.005), the zero-air test (D69)."""
    rng = np.random.default_rng(seed)
    y = x.copy()
    z = x == 0
    y[z] = 0.005 * np.hypot(rng.standard_normal(z.sum()), rng.standard_normal(z.sum()))
    return y


def smooth_phase(shape, seed=2) -> np.ndarray:
    """A smooth image phase (up to ~±2 rad), as coil combination and B0 leave in a real image (pilot `_phase`)."""
    rng = np.random.default_rng(seed)
    u = [np.linspace(-1, 1, s)[tuple(slice(None) if a == i else None for a in range(3))] for i, s in enumerate(shape)]
    c = rng.uniform(-1, 1, 6)
    return c[0] * u[0] + c[1] * u[1] + c[2] * u[2] + c[3] * u[0] * u[1] + c[4] * u[2] ** 2 + c[5] * u[1] * u[2]


def periodic_modulation(c, period, depth=0.1, axis=1, psi=1.0, theta=0.5) -> np.ndarray:
    """Periodic motion: the A–P lines × (1 + depth·cos(2π ky / period + ψ)·e^{iθ}), a period that need not divide
    the field of view (pilot realism check)."""
    return kspace_lines(c, lambda ky: 1 + depth * np.cos(2 * np.pi * ky / period + psi) * np.exp(1j * theta), axis)


def moved_once(x0, shift, frac=0.5, axis=1) -> np.ndarray:
    """One rigid move during the scan: the A–P k-space lines acquired after `frac` of the scan (lines in order from
    −N/2) come from the head shifted by `shift` voxels (LR, AP, SI); complex image back (D76, zaitsev2015)."""
    k0, k1 = np.fft.fftn(x0), np.fft.fftn(np.roll(x0, shift, axis=(0, 1, 2)))
    n = x0.shape[axis]
    late = (np.fft.fftfreq(n) * n + n // 2) >= frac * n
    shape = [1, 1, 1]
    shape[axis] = -1
    return np.fft.ifftn(np.where(late.reshape(shape), k1, k0))
