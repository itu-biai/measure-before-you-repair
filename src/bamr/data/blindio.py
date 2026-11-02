"""Blind case files (D11, D56): only the pixels may carry information.

Every NIfTI file has the same header (shape 192 × 240 × 1, one centred 1 mm RAS affine, empty description); the gzip
stream has no time stamp or file name, and every written file gets the same modification time.
"""
import gzip
import os
from pathlib import Path

import nibabel as nib
import numpy as np

from bamr.data.slices import CANVAS

AFFINE = np.eye(4)
AFFINE[:3, 3] = -(np.array((*CANVAS, 1)) - 1) / 2  # canvas centre at the origin: (−95.5, −119.5, 0)
FILE_TIME = 1767225600  # 2026-01-01 00:00 UTC, the same for every file


def nifti_bytes(canvas: np.ndarray, dtype) -> bytes:
    """(AP, LR) canvas → gzipped NIfTI-1 of shape (LR, AP, 1) with the shared header."""
    img = nib.Nifti1Image(np.ascontiguousarray(np.asarray(canvas).T[:, :, None].astype(dtype)), AFFINE)
    img.set_sform(AFFINE, code=1)
    img.set_qform(AFFINE, code=1)
    img.header["descrip"] = b""
    img.header.set_xyzt_units("mm")
    return gzip.compress(img.to_bytes(), compresslevel=6, mtime=0)


def write_file(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    os.utime(path, (FILE_TIME, FILE_TIME))


def read_canvas(path: Path) -> np.ndarray:
    """Inverse of nifti_bytes: the (AP, LR) canvas in its stored dtype."""
    img = nib.load(path)
    return np.asarray(img.dataobj)[:, :, 0].T.copy()


# ---------------------------------------------------------------- v2: whole volumes (D81)
BOX3 = (192, 240, 240)  # (LR, AP, SI), as artifacts3d.BOX
AFFINE3 = np.eye(4)
AFFINE3[:3, 3] = -(np.array(BOX3) - 1) / 2  # box centre at the origin: (−95.5, −119.5, −119.5)


def nifti_bytes3(vol: np.ndarray, dtype) -> bytes:
    """(LR, AP, SI) volume → gzipped NIfTI-1 with the one shared v2 header (D56's rules in 3-D, D81)."""
    vol = np.asarray(vol)
    if vol.shape != BOX3:
        raise ValueError(f"v2 volumes are {BOX3}, got {vol.shape}")
    img = nib.Nifti1Image(np.ascontiguousarray(vol.astype(dtype)), AFFINE3)
    img.set_sform(AFFINE3, code=1)
    img.set_qform(AFFINE3, code=1)
    img.header["descrip"] = b""
    img.header.set_xyzt_units("mm")
    return gzip.compress(img.to_bytes(), compresslevel=6, mtime=0)


def read_volume(path: Path) -> np.ndarray:
    """Inverse of nifti_bytes3: the (LR, AP, SI) volume in its stored dtype."""
    return np.asarray(nib.load(path).dataobj).copy()
