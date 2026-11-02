"""Reference-free image quality measures for the v2 clinical calibration (D84, D85).

Both need no clean image and no registration, so they compare real graded scans and our simulated ones on equal terms:
- entropy focus criterion (EFC): Atkinson et al. 1997, normalised by its maximum as in MRIQC (Esteban et al. 2017);
  ghosting and blurring spread the energy over more voxels and raise it;
- air signal: the mean outside the head grown by about 6 mm (MRIQC's background measures look at the same region);
  ghosts and stripes leak signal there.
"""
import numpy as np
from scipy import ndimage as ndi


def air_mask(v: np.ndarray) -> np.ndarray:
    """Air = outside the head grown by about 6 mm; built at half resolution for speed, the same rule for every scan."""
    h = ndi.binary_fill_holes(ndi.gaussian_filter(v[::2, ::2, ::2], 1) > 0.1)
    h = ndi.binary_dilation(h, iterations=3)
    full = np.repeat(np.repeat(np.repeat(h, 2, 0), 2, 1), 2, 2)[:v.shape[0], :v.shape[1], :v.shape[2]]
    return ~full


def air_signal(v: np.ndarray, air: np.ndarray) -> float:
    return float(v[air].mean())


def efc(v: np.ndarray) -> float:
    """Atkinson's entropy focus criterion over the positive voxels, normalised by its maximum (MRIQC's definition)."""
    x = np.abs(v[v > 0]).astype(np.float64)
    b = x / np.sqrt((x ** 2).sum())
    n = v.size
    emax = np.sqrt(n) * np.log(np.sqrt(n))
    return float(-(b * np.log(b)).sum() / emax)
