"""Native-noise estimators (spec Stage 1 step 5; D12, D16; lesson L8).

Convention: `v` is 3-D, the in-plane axes are (0, 1) and slices run along axis 2 (raw IXI: the
sagittal acquisition plane; 1 mm RAS: the axial plane). `m` is a bool region of the same shape,
normally eroded white matter. All functions raise ValueError when too few samples qualify.
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

MIN_SAMPLES = 200


def _need(n: int, what: str) -> None:
    if n < MIN_SAMPLES:
        raise ValueError(f"only {n} {what} inside the region (need {MIN_SAMPLES})")


def _detrend_matrix(p: int) -> np.ndarray:
    """R = I − A(AᵀA)⁻¹Aᵀ removes the best plane a + b·row + c·col from a flattened p×p patch."""
    yy, xx = np.mgrid[0:p, 0:p]
    a = np.c_[np.ones(p * p), yy.ravel(), xx.ravel()]
    return np.eye(p * p) - a @ np.linalg.pinv(a)


def mad_sigma(v: np.ndarray, m: np.ndarray) -> float:
    """Donoho: σ = median|HH| / 0.6745, one orthonormal Haar level per slice, 2×2 support inside m."""
    h, w = v.shape[0] // 2 * 2, v.shape[1] // 2 * 2
    x, mm = v[:h, :w].astype(np.float64), m[:h, :w]
    hh = (x[0::2, 0::2] - x[0::2, 1::2] - x[1::2, 0::2] + x[1::2, 1::2]) / 2
    sel = mm[0::2, 0::2] & mm[0::2, 1::2] & mm[1::2, 0::2] & mm[1::2, 1::2]
    _need(int(sel.sum()), "Haar coefficients")
    return float(np.median(np.abs(hh[sel])) / 0.6745)


def _tiles(a: np.ndarray, p: int) -> np.ndarray:
    """Non-overlapping p×p in-plane tiles → array (n_tiles, p*p)."""
    h, w = a.shape[0] // p * p, a.shape[1] // p * p
    t = a[:h, :w].reshape(h // p, p, w // p, p, a.shape[2]).transpose(0, 2, 4, 1, 3)
    return t.reshape(-1, p * p)


def local_std(v: np.ndarray, m: np.ndarray, patch: int = 8) -> float:
    """Upper bound on per-voxel σ: sqrt(median variance) of plane-detrended patch×patch tiles inside m."""
    keep = _tiles(m, patch).all(axis=1)
    _need(int(keep.sum()), f"{patch}x{patch} patches")
    res = _tiles(v.astype(np.float64), patch)[keep] @ _detrend_matrix(patch)
    var = (res ** 2).sum(axis=1) / (patch * patch - 3)
    return float(np.sqrt(np.median(var)))


def noise_spectrum(v: np.ndarray, m: np.ndarray, patch: int = 16, step: int = 4) -> np.ndarray:
    """Mean power spectrum (fftshifted, patch×patch) of detrended, Hann-windowed patches inside m."""
    keep = sliding_window_view(m, (patch, patch), axis=(0, 1))[::step, ::step].all(axis=(-2, -1))
    _need(int(keep.sum()), f"{patch}x{patch} patches")
    win = sliding_window_view(v, (patch, patch), axis=(0, 1))[::step, ::step]
    p = win[keep].reshape(-1, patch * patch).astype(np.float64) @ _detrend_matrix(patch)
    p = p.reshape(-1, patch, patch) * np.outer(np.hanning(patch), np.hanning(patch))
    return np.fft.fftshift((np.abs(np.fft.fft2(p)) ** 2).mean(axis=0))


def noise_rolloff_db(v: np.ndarray, m: np.ndarray, patch: int = 16, step: int = 4) -> float:
    """In-plane noise power at 0.25 Nyquist over 0.875 Nyquist, in dB (white noise ≈ 0)."""
    s = noise_spectrum(v, m, patch, step)
    c = patch // 2
    prof = (s.mean(axis=1)[c:] + s.mean(axis=0)[c:]) / 2  # bin i = i / (patch/2) Nyquist
    return float(10 * np.log10(prof[patch // 8] / prof[7 * patch // 16]))


def resample_white_gain(affine: np.ndarray, shape: tuple, seed: int = 0, trim: int = 8) -> float:
    """Std of unit white noise after the pipeline's ToCanonical + Resample(1 mm, B-spline)."""
    import torch
    import torchio as tio

    noise = np.random.default_rng(seed).standard_normal(shape).astype(np.float32)
    img = tio.ScalarImage(tensor=torch.from_numpy(noise)[None], affine=affine)
    out = tio.Compose([tio.ToCanonical(), tio.Resample(1, image_interpolation="bspline")])(img)
    a = out.data[0].numpy()
    return float(a[trim:-trim, trim:-trim, trim:-trim].std())
