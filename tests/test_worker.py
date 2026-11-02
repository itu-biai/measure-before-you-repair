import json

import nibabel as nib
import numpy as np
import pytest

from bamr.config import load_config
from bamr.data.subjects import Subject
from bamr.data.worker import OUTPUTS, build_manifest, process_subject

PSR = np.array([[0, 0, 1.2, -90], [-0.9375, 0, 0, 120], [0, 0.9375, 0, -100], [0, 0, 0, 1.0]])


def _write_subject(tmp_path, empty_mask=False, n=(96, 96, 72), island=False):
    i, j, k = np.meshgrid(*[np.arange(s) - (s - 1) / 2 for s in n], indexing="ij")
    r = np.sqrt((i * 0.9375) ** 2 + (j * 0.9375) ** 2 + (k * 1.2) ** 2)
    v = np.where(r <= 40, 0.6, 0.05)
    v[r <= 32] = 1.0
    v[r <= 6] = 0.3
    field = 1 + 0.2 * (i / n[0] + (j / n[1]) ** 2)
    v = np.abs(v * field + np.random.default_rng(0).normal(0, 0.02, n)) * 400
    raw = tmp_path / "raw" / "IXI999-HH-0000-T1.nii.gz"
    raw.parent.mkdir()
    nib.save(nib.Nifti1Image(v.astype(np.float32), PSR), raw)
    mdir = tmp_path / "prep" / "IXI999"
    mdir.mkdir(parents=True)
    mask = np.zeros(n, np.uint8) if empty_mask else (r <= 40).astype(np.uint8)
    if island:  # a false-positive blob outside the brain, like HD-BET's chin/sinus islands
        mask[2:6, 2:6, 2:6] = 1
    nib.save(nib.Nifti1Image(mask, PSR), mdir / "brain_mask_native.nii.gz")
    return Subject("IXI999", "HH", raw), mdir / "brain_mask_native.nii.gz", mdir


def test_worker_synthetic_end_to_end(tmp_path):
    subj, mpath, out = _write_subject(tmp_path)
    rec = process_subject(subj, mpath, out, load_config())
    assert rec["error"] is None, rec["error"]
    for name in OUTPUTS:
        assert (out / name).exists(), name
    t1 = nib.load(out / "t1_1mm.nii.gz")
    m1 = np.asarray(nib.load(out / "brain_mask_1mm.nii.gz").dataobj) > 0
    assert nib.aff2axcodes(t1.affine) == ("R", "A", "S")
    assert t1.get_data_dtype() == np.float32
    a = t1.get_fdata()
    assert np.median(a[m1 & (a > 0.85)]) == pytest.approx(1.0, abs=0.03)
    assert json.loads((out / "qc.json").read_text())["id"] == "IXI999"
    assert rec["cov_after"] < rec["cov_before"]
    assert 0.6 < rec["resample_gain"] < 0.95
    assert rec["sigma_mad_native"] == pytest.approx(0.02, rel=0.25)


def test_worker_empty_mask_is_error_row(tmp_path):
    subj, mpath, out = _write_subject(tmp_path, empty_mask=True, n=(32, 32, 24))
    rec = process_subject(subj, mpath, out, load_config())
    assert rec["error"]
    assert rec["all_ok"] is False
    assert (out / "qc.json").exists()


def test_worker_skips_done(tmp_path):
    subj, mpath, out = _write_subject(tmp_path, empty_mask=True, n=(32, 32, 24))
    process_subject(subj, mpath, out, load_config())
    t = (out / "qc.json").stat().st_mtime_ns
    rec = process_subject(subj, mpath, out, load_config())
    assert (out / "qc.json").stat().st_mtime_ns == t
    assert rec["id"] == "IXI999"


def test_build_manifest(tmp_path):
    subj, mpath, out = _write_subject(tmp_path, empty_mask=True, n=(32, 32, 24))
    process_subject(subj, mpath, out, load_config())
    df = build_manifest(tmp_path / "prep", order=["IXI999"], n_priority=60)
    assert list(df["id"]) == ["IXI999"] and df["rank"].iloc[0] == 0 and bool(df["priority"].iloc[0])
    assert (tmp_path / "prep" / "manifest.csv").exists()


def test_worker_keeps_largest_component(tmp_path):
    subj, mpath, out = _write_subject(tmp_path, island=True)
    rec = process_subject(subj, mpath, out, load_config())
    assert rec["error"] is None, rec["error"]
    assert rec["n_cc"] == 2 and rec["removed_ml"] > 0 and rec["mask_main_ok"]
    from scipy import ndimage as ndi
    m1 = np.asarray(nib.load(out / "brain_mask_1mm.nii.gz").dataobj) > 0
    assert ndi.label(m1, structure=np.ones((3, 3, 3)))[1] == 1
