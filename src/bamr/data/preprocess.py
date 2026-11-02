"""Stage 1 steps as pure functions on numpy arrays (spec Stage 1 steps 2-4 and 6; D14, D15, D17, D20).

Arrays keep nibabel's voxel order (i, j, k); `affine`/`spacing` describe that order.
"""
import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi
from skimage.filters import threshold_multiotsu


def n4_correct(img: np.ndarray, mask: np.ndarray, spacing: tuple, shrink: int = 4, levels: int = 4,
               iters: int = 50, conv: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """N4 (tustison2010) fitted on a `shrink`× subsampled grid inside `mask & img > 0`.

    Returns (img / field, field), both float32 on the input grid; field = exp(N4 log-field).
    """
    x = sitk.GetImageFromArray(img.astype(np.float32))  # sitk reverses axis order → reverse spacing
    x.SetSpacing(tuple(float(s) for s in spacing[::-1]))
    m = sitk.GetImageFromArray(((mask > 0) & (img > 0)).astype(np.uint8))
    m.CopyInformation(x)
    f = sitk.N4BiasFieldCorrectionImageFilter()
    f.SetMaximumNumberOfIterations([iters] * levels)
    f.SetConvergenceThreshold(conv)
    f.Execute(sitk.Shrink(x, [shrink] * 3), sitk.Shrink(m, [shrink] * 3))
    field = np.exp(sitk.GetArrayFromImage(f.GetLogBiasFieldAsImage(x))).astype(np.float32)
    return (img / field).astype(np.float32), field


def to_ras_1mm(img: np.ndarray, mask: np.ndarray, affine: np.ndarray,
               image_interp: str = "bspline", mask_interp: str = "nearest",
               spacing_mm: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """TorchIO ToCanonical + Resample(1 mm). Returns (image ≥ 0 float32, mask uint8, new affine).

    The output shape is whatever TorchIO computes from the input extent; never assume it.
    """
    import torch
    import torchio as tio

    subj = tio.Subject(img=tio.ScalarImage(tensor=torch.from_numpy(img.astype(np.float32))[None], affine=affine),
                       mask=tio.LabelMap(tensor=torch.from_numpy(mask.astype(np.uint8))[None], affine=affine))
    out = tio.Compose([tio.ToCanonical(), tio.Resample(spacing_mm, image_interpolation=image_interp,
                                                       label_interpolation=mask_interp)])(subj)
    im = np.clip(out["img"].data[0].numpy(), 0, None).astype(np.float32)
    return im, out["mask"].data[0].numpy().astype(np.uint8), np.asarray(out["img"].affine)


def tissue_classes(img: np.ndarray, brain: np.ndarray, smooth: float = 0.0):
    """3-class multi-Otsu (otsu1979) inside `brain` → (wm, gm, csf, thresholds); brightest class = WM on T1."""
    brain = brain.astype(bool)
    if brain.sum() < 1000:
        raise ValueError(f"brain mask has only {int(brain.sum())} voxels")
    src = ndi.gaussian_filter(img.astype(np.float32), smooth) if smooth else img
    t = threshold_multiotsu(src[brain], classes=3)
    return brain & (src > t[1]), brain & (src > t[0]) & (src <= t[1]), brain & (src <= t[0]), t


def wm_normalize(img: np.ndarray, brain: np.ndarray) -> tuple[np.ndarray, float]:
    """Divide by the WM median (D15). Returns (float32 image with WM = 1, WM median)."""
    wm = tissue_classes(img, brain)[0]
    med = float(np.median(img[wm]))
    if not med > 0:
        raise ValueError(f"WM median is {med}")
    return (img / med).astype(np.float32), med


def wm_cov(img: np.ndarray, wm: np.ndarray) -> float:
    v = img[wm].astype(np.float64)
    return float(v.std() / v.mean())


def wm_gm_contrast(img: np.ndarray, wm: np.ndarray, gm: np.ndarray) -> float:
    return float(img[wm].astype(np.float64).mean() / img[gm].astype(np.float64).mean())


def local_contrast_ratio(before: np.ndarray, after: np.ndarray, wm: np.ndarray, gm: np.ndarray, spacing: tuple,
                         block_mm: float = 16.0, min_vox: int = 50) -> float:
    """Median over block_mm cubes of (WM/GM after) / (WM/GM before), using block-wise medians (D20, revised).

    A smooth bias field is ~constant inside a block, so removing it leaves this ratio ≈ 1; a field that
    follows the anatomy lowers it. (The global WM/GM ratio is unsuitable: a centre-bright field inflates it.)
    """
    b = [max(1, int(round(block_mm / s))) for s in spacing]
    r = []
    for i in range(0, wm.shape[0], b[0]):
        for j in range(0, wm.shape[1], b[1]):
            for k in range(0, wm.shape[2], b[2]):
                sl = np.s_[i:i + b[0], j:j + b[1], k:k + b[2]]
                w, g = wm[sl], gm[sl]
                if w.sum() >= min_vox and g.sum() >= min_vox:
                    cb = np.median(before[sl][w]) / np.median(before[sl][g])
                    ca = np.median(after[sl][w]) / np.median(after[sl][g])
                    r.append(ca / cb)
    if len(r) < 10:
        raise ValueError(f"only {len(r)} blocks contain both WM and GM")
    return float(np.median(r))


def qc_flags(m: dict, cfg_qc: dict) -> dict[str, bool]:
    """QC flags (D17, D20r, D37). Missing or NaN values make the flag False."""
    def g(k):
        v = m.get(k)
        return float("nan") if v is None else float(v)

    lo_b, hi_b = cfg_qc["brain_l"]
    lo_s, hi_s = cfg_qc["sigma"]
    f = {
        "brain_volume_ok": bool(lo_b <= g("brain_l") <= hi_b),
        "mask_main_ok": bool(g("largest_cc_frac") >= cfg_qc["main_cc_min"]),
        "cov_drops": bool(g("cov_after") < g("cov_before")),
        "contrast_kept": bool(g("contrast_local_ratio") >= cfg_qc["contrast_ratio_min"]),
        "sigma_ok": bool(lo_s <= g("sigma_mad_native") and g("sigma_lstd_native") <= hi_s),
    }
    f["all_ok"] = all(f.values())
    return f
