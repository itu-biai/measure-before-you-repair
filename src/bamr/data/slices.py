"""Axial slices on the fixed Stage 2/3 canvas (D13, D30, D39).

Volumes are RAS at 1 mm (Stage 1 output). A canvas is an (AP, LR) array = vol[:, :, k].T cut to
CANVAS = (LR, AP) = (192, 240): rows run towards anterior, columns towards the patient's right, so the
last axis is L–R = the phase-encoding axis of Stage 2.
"""
import numpy as np

CANVAS = (192, 240)  # (LR, AP); N_LR // 2 = 96 is a multiple of 2, 3 and 4 (D30)


def zstar(mask: np.ndarray) -> int:
    """Axial index (RAS axis 2) with the largest brain area (D13)."""
    return int(np.argmax(mask.reshape(-1, mask.shape[2]).sum(axis=0)))


def canvas_origin(mask: np.ndarray, canvas: tuple[int, int] = CANVAS) -> tuple[int, int]:
    """(LR, AP) start index of a canvas window centred on the 3-D brain bounding box (D39).

    May be negative or run past the volume; axial_canvas zero-pads there."""
    out = []
    for ax, n in zip((0, 1), canvas):
        idx = np.flatnonzero(mask.any(axis=tuple(a for a in range(3) if a != ax)))
        centre = (idx[0] + idx[-1] + 1) / 2
        out.append(int(round(centre - n / 2)))
    return out[0], out[1]


def axial_canvas(vol: np.ndarray, k: int, origin: tuple[int, int], canvas: tuple[int, int] = CANVAS) -> np.ndarray:
    """Axial slice k cut to the canvas window at `origin`; (AP, LR) array, zeros outside the volume."""
    sl = vol[:, :, k]
    out = np.zeros((canvas[1], canvas[0]), vol.dtype)
    src, dst = [], []
    for o, n, size in zip(origin, canvas, sl.shape):
        lo, hi = max(o, 0), min(o + n, size)
        src.append(slice(lo, hi))
        dst.append(slice(lo - o, hi - o))
    out[dst[1], dst[0]] = sl[src[0], src[1]].T
    return out
