"""Shared regions for all tools (D70): the head, and the flat tissue (white matter) to measure in.

Works on the whole 3-D volume. Inputs are WM-normalized (D56), so no tissue brightness is estimated.
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu

SMOOTH = 2.0  # voxels
OTSU_FRACTION = 0.5
INSET = 8  # voxels from the head's edge
TOL = 0.15  # flat tissue: within ±15 % of WM = 1, decided voxel by voxel


def head_mask(x: np.ndarray) -> np.ndarray:
    """Smooth by 2 voxels, threshold at half of Otsu's threshold, keep the biggest 3-D piece, fill holes."""
    s = ndi.gaussian_filter(np.asarray(x, np.float32), SMOOTH)
    m = s > OTSU_FRACTION * threshold_otsu(s) if np.ptp(s) > 0 else np.zeros(s.shape, bool)
    lab, n = ndi.label(m)
    if n == 0:
        return np.zeros(s.shape, bool)
    m = lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1)
    return ndi.binary_fill_holes(m)


def flat_tissue(x: np.ndarray, head: np.ndarray) -> np.ndarray:
    """8 voxels inside the head and within ±15 % of 1 (a smoothed copy was tried and dropped: bias false alarms)."""
    inner = inset(head) >= INSET
    return inner & (np.abs(np.asarray(x) - 1.0) < TOL)


def inset(head: np.ndarray) -> np.ndarray:
    """Distance to outside the head, counting the array's faces as outside (D99 #7: a head cut by the box)."""
    return ndi.distance_transform_edt(np.pad(head, 1))[1:-1, 1:-1, 1:-1]
