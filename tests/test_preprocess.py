import nibabel as nib
import numpy as np
import pytest
from scipy import ndimage as ndi

from bamr.data.preprocess import (local_contrast_ratio, n4_correct, qc_flags, tissue_classes, to_ras_1mm, wm_cov,
                                  wm_gm_contrast, wm_normalize)

QC = {"brain_l": [0.9, 2.0], "main_cc_min": 0.95, "contrast_ratio_min": 0.95, "sigma": [0.005, 0.10]}


def _brain(n=64, scale=300.0, noise=0.01, seed=0):
    """Concentric 'brain': CSF ventricle (0.3) in WM (1.0) in a GM shell (0.6), zero outside."""
    z, y, x = np.mgrid[:n, :n, :n] - (n - 1) / 2
    r = np.sqrt(x ** 2 + (1.15 * y) ** 2 + z ** 2)
    v = np.zeros((n, n, n))
    v[r <= 28] = 0.6
    v[r <= 22] = 1.0
    v[r <= 6] = 0.3
    brain = r <= 28
    v = v + np.random.default_rng(seed).normal(0, noise, v.shape) * brain
    return np.clip(v, 0, None) * scale, brain, ndi.binary_erosion(v * brain > 0.9, iterations=2)


def _field(n=64, amp=0.25):
    z, y, x = (np.mgrid[:n, :n, :n] - (n - 1) / 2) / (n / 2)
    g = x + 0.6 * y ** 2 - 0.4 * z + 0.3 * x * z
    return 1 + amp * g / np.abs(g).max()


def test_n4_removes_known_field():
    v, brain, wm = _brain()
    f = _field()
    corr, field = n4_correct(v * f, brain, spacing=(1.0, 1.0, 1.0))
    assert wm_cov(corr, wm) < 0.5 * wm_cov(v * f, wm)
    rel = np.abs(field / np.median(field[brain]) - f / np.median(f[brain])) / (f / np.median(f[brain]))
    assert np.median(rel[brain]) < 0.05


def test_n4_ignores_zeros():
    v, brain, _ = _brain()
    v[30:34, 30:34, 30:34] = 0
    corr, field = n4_correct(v * _field(), brain, spacing=(1.0, 1.0, 1.2))
    assert np.isfinite(corr).all() and np.isfinite(field).all()


def _oblique_psr(deg=10.0):
    th = np.deg2rad(deg)
    rot = np.array([[1, 0, 0], [0, np.cos(th), -np.sin(th)], [0, np.sin(th), np.cos(th)]])
    psr = np.array([[0, 0, 1.2], [-0.9375, 0, 0], [0, 0.9375, 0]])  # columns: voxel axes → mm, (P, S, R)
    aff = np.eye(4)
    aff[:3, :3] = rot @ psr
    aff[:3, 3] = [-20, 25, -30]
    return aff


def test_to_ras_1mm_geometry():
    aff = _oblique_psr()
    rng = np.random.default_rng(0)
    img = ndi.gaussian_filter(rng.uniform(0, 1000, (60, 60, 40)), 2) + 50
    mask = np.zeros((60, 60, 40), np.uint8)
    mask[15:45, 12:50, 8:32] = 1
    im1, m1, aff1 = to_ras_1mm(img, mask, aff)
    assert nib.aff2axcodes(aff1) == ("R", "A", "S")
    assert np.allclose(np.linalg.norm(aff1[:3, :3], axis=0), 1.0, atol=1e-6)
    assert set(np.unique(m1)) <= {0, 1}
    assert im1.min() >= 0 and im1.dtype == np.float32 and m1.dtype == np.uint8
    vol0 = mask.sum() * abs(np.linalg.det(aff[:3, :3]))
    assert m1.sum() == pytest.approx(vol0, rel=0.03)
    assert im1.shape == m1.shape


def test_tissue_classes_and_normalize():
    v, brain, wm_true = _brain(scale=500.0)
    wm, gm, csf, th = tissue_classes(v, brain)
    assert (wm & wm_true).sum() / wm_true.sum() > 0.99
    assert not (wm & gm).any() and not (gm & csf).any()
    out, scale = wm_normalize(v, brain)
    assert out.dtype == np.float32
    assert scale == pytest.approx(500.0, rel=0.02)
    assert np.median(out[wm]) == pytest.approx(1.0, abs=1e-6)


def test_contrast_and_cov():
    v, brain, _ = _brain(noise=0.03)
    wm, gm, _, _ = tissue_classes(v, brain)
    assert wm_gm_contrast(v, wm, gm) == pytest.approx(1.0 / 0.6, rel=1e-2)
    assert wm_cov(v, wm) == pytest.approx(0.03, rel=0.1)


def test_local_contrast_smooth_field_kept():
    """A smooth field (true bias) changes global WM/GM contrast but not the local one."""
    v, brain, _ = _brain(noise=0.02)
    wm, gm, _, _ = tissue_classes(v, brain)
    z, y, x = (np.mgrid[:64, :64, :64] - 31.5) / 32
    f = 1 + 0.3 * (1 - (x ** 2 + y ** 2 + z ** 2))  # centre-bright, like 3T
    assert wm_gm_contrast(v * f, wm, gm) / wm_gm_contrast(v, wm, gm) > 1.05  # global inflated ~8 % (IXI127: 9 %)
    # the phantom is ~2.5x smaller than a brain, so the 16 mm block scales to 6 mm (real IXI127 gave 0.978)
    assert local_contrast_ratio(v * f, v, wm, gm, (1.0, 1.0, 1.0), block_mm=6) > 0.96


def test_local_contrast_anatomy_eating_field_flagged():
    """A 'field' that follows tissue (removes WM/GM contrast) lowers the local ratio."""
    v, brain, _ = _brain(noise=0.02)
    wm, gm, _, _ = tissue_classes(v, brain)
    f = 1 + 0.2 * wm
    assert local_contrast_ratio(v, v / f, wm, gm, (1.0, 1.0, 1.0), block_mm=6) < 0.9


def _ok():
    return dict(brain_l=1.4, largest_cc_frac=0.999, cov_before=0.10, cov_after=0.06, contrast_local_ratio=0.99,
                sigma_mad_native=0.012, sigma_lstd_native=0.025)


@pytest.mark.parametrize("key,val,flag,expect", [
    ("brain_l", 0.85, "brain_volume_ok", False), ("brain_l", 1.4, "brain_volume_ok", True),
    ("brain_l", 2.1, "brain_volume_ok", False), ("largest_cc_frac", 0.90, "mask_main_ok", False),
    ("largest_cc_frac", 0.986, "mask_main_ok", True),
    ("cov_after", 0.11, "cov_drops", False), ("contrast_local_ratio", 0.94, "contrast_kept", False),
    ("contrast_local_ratio", 0.96, "contrast_kept", True), ("sigma_mad_native", 0.004, "sigma_ok", False),
    ("sigma_lstd_native", 0.11, "sigma_ok", False),
])
def test_qc_flags_thresholds(key, val, flag, expect):
    m = _ok()
    m[key] = val
    f = qc_flags(m, QC)
    assert f[flag] is expect
    assert f["all_ok"] is all(v for k, v in f.items() if k != "all_ok")


def test_qc_flags_missing_values_fail():
    f = qc_flags({"brain_l": float("nan")}, QC)
    assert not any(f.values())
