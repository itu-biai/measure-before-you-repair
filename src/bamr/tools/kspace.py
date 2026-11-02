"""K-space helpers shared by the spike and ghosting tools (D96, D97).

The spectrum is the volume's full 3-D FFT, centred (fftshift): K[k + N//2] for k = −N//2 … N//2 − 1 on each axis.
The full one, not a real FFT's half: a peak between samples near the half's edge would be cut in two.
"""
import numpy as np
import scipy.fft as sfft
from scipy import ndimage as ndi


def spectrum(x: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(sfft.fftn(np.asarray(x, np.float32), workers=1))


def kgrid(shape) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Open integer grids of k along each axis, matching `spectrum`."""
    g = np.ogrid[tuple(slice(0, s) for s in shape)]
    return tuple(gi - s // 2 for gi, s in zip(g, shape))


def shell_z(d: np.ndarray, shell: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """(d − shell median) / (1.4826 · shell MAD) over the kept samples, one shell per label; −∞ elsewhere and in a
    shell with no spread. A robust z within each distance from the centre, where the spectrum's own strength is alike."""
    idx = np.flatnonzero(keep)
    sv = shell.reshape(-1)[idx]
    o = np.argsort(sv, kind="stable")
    sv, dv = sv[o], d.reshape(-1)[idx][o].astype(np.float64)
    z = np.full(dv.shape, -np.inf)
    starts = np.flatnonzero(np.r_[True, sv[1:] != sv[:-1]]) if sv.size else np.array([], int)
    for a, b in zip(starts, np.r_[starts[1:], sv.size]):
        seg = dv[a:b]
        med = np.median(seg)
        mad = 1.4826 * np.median(np.abs(seg - med))
        if mad > 0:
            z[a:b] = (seg - med) / mad
    out = np.full(d.shape, -np.inf, np.float32)
    out.reshape(-1)[idx[o]] = z
    return out


def peak_contrast(P: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """d = log(mean power in the 3×3×3 around each sample) − log(mean power in the 7³ ring around that), with the
    3³ sums s3 and ring means: a compact peak stands out of its ring; smooth anatomy does not (zhang2001)."""
    s3 = ndi.uniform_filter(P, 3, mode="nearest") * 27
    ring = (ndi.uniform_filter(P, 7, mode="nearest") * 343 - s3) / 316
    return np.log(s3 / 27 + 1e-30) - np.log(np.maximum(ring, 0) + 1e-30), s3, ring


def excess(K: np.ndarray, idx) -> float:
    """Power in the 3³ around idx above what its 7³ ring predicts there (≥ 0): the peak's own energy."""
    b7 = np.abs(K[tuple(slice(j - 3, j + 4) for j in idx)]).astype(np.float64) ** 2
    s3 = b7[2:5, 2:5, 2:5].sum()
    return max(float(s3 - 27 * (b7.sum() - s3) / 316), 0.0)


def low_freq_power(mask: np.ndarray) -> float:
    """Σ |FFT(mask)|² over its 27 lowest frequencies (k ∈ {−1, 0, 1} on each axis), by a separable direct DFT.
    A stripe a·cos(k·r) over the head puts (a/2)²× this into the 3³ around ±k: so a = 2·√(excess / this)."""
    out = np.asarray(mask, np.float64)
    for n in mask.shape:
        out = np.tensordot(out, np.exp(-2j * np.pi * np.outer(np.arange(n), [-1, 0, 1]) / n), axes=([0], [0]))
    return float((np.abs(out) ** 2).sum())


def line_energy(K: np.ndarray, axis: int, disk: int = 64) -> np.ndarray:
    """Typical energy of each k-space plane along `axis`: median |K|² within `disk` samples of the centre of the
    plane, where the head's signal is above the noise floor. The median ignores a spike's few samples."""
    Kt = np.moveaxis(K, axis, 0)
    a, b = Kt.shape[1:]
    u = (np.arange(a) - a // 2)[:, None] ** 2 + (np.arange(b) - b // 2)[None, :] ** 2
    return np.median(np.abs(Kt[:, u <= disk ** 2]).astype(np.float64) ** 2, axis=1)


def periodogram(E: np.ndarray, window: int = 9, gap: int = 2) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Periodogram of the line energies' fast wiggle: log E minus its running median over `window` lines, the
    `gap` lines next to the centre set aside (they carry the image's contrast). Returns f, power, residual."""
    L = np.log(np.maximum(E, 1e-30))
    res = L - ndi.median_filter(L, window, mode="nearest")
    ky = np.arange(L.size) - L.size // 2
    use = np.abs(ky) > gap
    rz = np.where(use, res - res[use].mean(), 0.0)
    return np.fft.rfftfreq(rz.size), np.abs(np.fft.rfft(rz)) ** 2, rz
