import gzip
import os

import nibabel as nib
import numpy as np

from bamr.data.blindio import AFFINE, FILE_TIME, nifti_bytes, read_canvas, write_file


def canvas(seed):
    return np.random.default_rng(seed).uniform(0, 1.5, (240, 192)).astype(np.float32)


def test_roundtrip_exact(tmp_path):
    x = canvas(0)
    p = tmp_path / "a.nii.gz"
    write_file(p, nifti_bytes(x, np.float32))
    assert nib.load(p).shape == (192, 240, 1)
    np.testing.assert_allclose(nib.load(p).affine, AFFINE)
    y = read_canvas(p)
    assert y.dtype == np.float32 and np.array_equal(y, x)
    m = canvas(1) > 0.7
    write_file(tmp_path / "m.nii.gz", nifti_bytes(m, np.uint8))
    assert read_canvas(tmp_path / "m.nii.gz").dtype == np.uint8
    assert np.array_equal(read_canvas(tmp_path / "m.nii.gz") > 0, m)


def test_headers_identical():
    a, b = (gzip.decompress(nifti_bytes(canvas(s), np.float32)) for s in (0, 1))
    assert a[:352] == b[:352] and a[352:] != b[352:]
    h = nib.Nifti1Image.from_bytes(a).header
    assert h["descrip"].item() == b"" and int(h["sform_code"]) == 1 and int(h["qform_code"]) == 1
    assert h.get_xyzt_units()[0] == "mm"


def test_gzip_has_no_time_or_name():
    payload = nifti_bytes(canvas(0), np.float32)
    assert payload[:2] == b"\x1f\x8b"
    assert payload[4:8] == b"\0\0\0\0"
    assert payload[3] & 0x08 == 0


def test_same_input_same_bytes():
    assert nifti_bytes(canvas(3), np.float32) == nifti_bytes(canvas(3), np.float32)


def test_mtime_fixed(tmp_path):
    p = tmp_path / "sub" / "x.json"
    write_file(p, b"{}")
    assert os.stat(p).st_mtime == FILE_TIME
