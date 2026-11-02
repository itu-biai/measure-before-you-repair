"""One subject through Stage 1 steps 2-7 (N4 → σ → 1 mm RAS → WM = 1 → QC), plus the manifest.

`qc.json` is written last and marks the subject as done; every file goes tmp → rename.
"""
import json
import math
import os
import time
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

from bamr.data.noise_sigma import local_std, mad_sigma, noise_rolloff_db, resample_white_gain
from bamr.data.preprocess import (local_contrast_ratio, n4_correct, qc_flags, tissue_classes, to_ras_1mm, wm_cov,
                                  wm_gm_contrast, wm_normalize)
from bamr.data.subjects import Subject

FIELD_STEP = 4  # the N4 field is smooth: store every 4th voxel (affine adjusted) to save ~18 MB/subject
OUTPUTS = ("t1_1mm.nii.gz", "brain_mask_1mm.nii.gz", "n4_field_native_x4.nii.gz", "qc.png", "qc.json")


def _save_nii(arr: np.ndarray, affine: np.ndarray, path: Path, dtype=None) -> None:
    img = nib.Nifti1Image(arr, affine)
    if dtype is not None:
        img.set_data_dtype(dtype)  # int16 → nibabel picks scl_slope/inter
    tmp = path.with_name(".tmp_" + path.name)
    nib.save(img, tmp)
    os.replace(tmp, path)


def _write_json(rec: dict, path: Path) -> None:
    tmp = path.with_name(".tmp_" + path.name)
    tmp.write_text(json.dumps(rec, indent=1, default=float))
    os.replace(tmp, path)


def _view(sl: np.ndarray) -> np.ndarray:
    """2-D slice of an RAS array → radiological display (patient right on the left, A/S up)."""
    return sl.T[::-1, ::-1]


def _qc_png(t1: np.ndarray, m1: np.ndarray, field: np.ndarray, rec: dict, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = int(np.argmax(m1.sum(axis=(0, 1))))
    ci = [int(round(c)) for c in ndi.center_of_mass(m1)]
    views = [(_view(t1[:, :, k]), _view(m1[:, :, k]), f"axial z*={k}"),
             (_view(t1[:, ci[1], :]), _view(m1[:, ci[1], :]), "coronal"),
             (_view(t1[ci[0], :, :]), _view(m1[ci[0], :, :]), "sagittal")]
    fig, axs = plt.subplots(1, 4, figsize=(13, 3.4))
    for ax, (im, mm, title) in zip(axs, views):
        ax.imshow(im, cmap="gray", vmin=0, vmax=1.6)
        ax.contour(mm, [0.5], colors="#ff3b30", linewidths=0.7)
        ax.set_title(title, fontsize=9)
    fsl = field[:, :, field.shape[2] // 2].T[::-1, :]  # raw (P, S, R): sagittal, anterior on the left
    h = axs[3].imshow(fsl, cmap="magma")
    fig.colorbar(h, ax=axs[3], fraction=0.046)
    axs[3].set_title("N4 field (native, sagittal)", fontsize=9)
    for ax in axs:
        ax.axis("off")
    flags = " ".join(f"{k}={'✓' if rec.get(k) else '✗'}" for k in
                     ("brain_volume_ok", "mask_main_ok", "cov_drops", "contrast_kept", "sigma_ok"))
    fig.suptitle(f"{rec['id']} {rec['site']} · brain {rec.get('brain_l', float('nan')):.2f} L · "
                 f"σ_MAD {100 * rec.get('sigma_mad_native', float('nan')):.2f}% · {flags}", fontsize=9, y=0.99)
    fig.subplots_adjust(top=0.82)
    tmp = path.with_name(".tmp_" + path.name)
    fig.savefig(tmp, dpi=70, format="png")
    plt.close(fig)
    os.replace(tmp, path)


def process_subject(subj: Subject, mask_path: Path, out_dir: Path, cfg: dict) -> dict:
    out_dir = Path(out_dir)
    qc_path = out_dir / "qc.json"
    if qc_path.exists():
        return json.loads(qc_path.read_text())
    s1, t_start = cfg["stage1"], time.time()
    rec: dict = {"id": subj.id, "site": subj.site, "error": None}
    try:
        nii = nib.load(subj.path)
        raw = np.asarray(nii.dataobj, dtype=np.float32)
        aff, zooms = nii.affine, tuple(float(z) for z in nii.header.get_zooms()[:3])
        mask = np.asarray(nib.load(mask_path).dataobj) > 0
        rec.update(axcodes_raw="".join(nib.aff2axcodes(aff)), shape_raw=list(raw.shape),
                   spacing_raw=[round(z, 5) for z in zooms])
        lab, n_cc = ndi.label(mask, structure=np.ones((3, 3, 3)))
        sizes = np.bincount(lab.ravel())[1:]
        if n_cc > 1:  # D37: HD-BET's extra pieces are false positives (chin, sinus); keep the main one
            mask = lab == (int(np.argmax(sizes)) + 1)
        vox_ml = math.prod(zooms) / 1e3
        rec.update(brain_l=float(mask.sum() * vox_ml / 1e3), n_cc=int(n_cc),
                   largest_cc_frac=float(sizes.max() / sizes.sum()) if n_cc else 0.0,
                   removed_ml=float((sizes.sum() - sizes.max()) * vox_ml) if n_cc else 0.0)
        if mask.sum() < 1000:
            raise ValueError(f"brain mask empty or tiny ({int(mask.sum())} voxels)")

        t = time.time()  # step 2: N4 on the raw grid
        n4 = s1["n4"]
        corr, field = n4_correct(raw, mask, zooms, n4["shrink"], n4["levels"], n4["iterations"], n4["convergence"])
        rec["t_n4"] = time.time() - t

        t = time.time()  # step 5 (native σ) + step 6 inputs: regions from the N4-corrected image
        wm, gm, _, _ = tissue_classes(corr, mask, smooth=s1["wm"]["sigma_smooth_vox"])
        wm_e = ndi.binary_erosion(wm, iterations=s1["wm"]["erode_iter"])
        gm_e = ndi.binary_erosion(gm, iterations=1)
        w_nat = wm_normalize(corr, mask)[1]
        rec.update(wm_median_native=w_nat,
                   cov_before=wm_cov(raw, wm_e), cov_after=wm_cov(corr, wm_e),
                   contrast_before=wm_gm_contrast(raw, wm_e, gm_e), contrast_after=wm_gm_contrast(corr, wm_e, gm_e),
                   contrast_local_ratio=local_contrast_ratio(raw, corr, wm_e, gm_e, zooms),
                   field_wm_gm=float(field[wm_e].mean() / field[gm_e].mean()),
                   sigma_mad_native=mad_sigma(corr, wm_e) / w_nat, sigma_lstd_native=local_std(corr, wm_e) / w_nat)
        rec["rolloff_db"] = float("nan")
        for step in (4, 2):  # thin WM can leave too few 16x16 patches at stride 4
            try:
                rec["rolloff_db"] = noise_rolloff_db(corr, wm_e, step=step)
                break
            except ValueError:
                pass
        rec["resample_gain"] = resample_white_gain(aff, raw.shape)
        rec["sigma_mad_1mm"] = rec["sigma_mad_native"] * rec["resample_gain"]
        rec["sigma_lstd_1mm"] = rec["sigma_lstd_native"] * rec["resample_gain"]
        rec["t_sigma"] = time.time() - t

        t = time.time()  # steps 3-4: RAS 1 mm, then WM = 1
        rs = s1["resample"]
        im1, m1, aff1 = to_ras_1mm(corr, mask, aff, rs["image_interp"], rs["mask_interp"], rs["spacing_mm"])
        im1, scale = wm_normalize(im1, m1 > 0)
        wm1 = ndi.binary_erosion(tissue_classes(im1, m1 > 0, smooth=s1["wm"]["sigma_smooth_vox"])[0],
                                 iterations=s1["wm"]["erode_iter"])
        rec.update(shape_1mm=list(im1.shape), wm_median_1mm_pre_norm=scale,
                   sigma_lstd_1mm_measured=local_std(im1, wm1))
        rec["t_resample"] = time.time() - t

        _save_nii(im1, aff1, out_dir / "t1_1mm.nii.gz")
        _save_nii(m1, aff1, out_dir / "brain_mask_1mm.nii.gz")
        aff_x4 = aff @ np.diag([FIELD_STEP] * 3 + [1])
        _save_nii(field[::FIELD_STEP, ::FIELD_STEP, ::FIELD_STEP], aff_x4, out_dir / "n4_field_native_x4.nii.gz")
        rec.update(qc_flags(rec, s1["qc"]))
        _qc_png(im1, m1, field, rec, out_dir / "qc.png")
    except Exception as e:  # one bad subject must not stop the run
        rec["error"] = f"{type(e).__name__}: {e}"
    rec.update(qc_flags(rec, s1["qc"]))
    rec["all_ok"] = rec["all_ok"] and rec["error"] is None  # partial flags stay for diagnosis
    rec["t_total"] = time.time() - t_start
    _write_json(rec, qc_path)
    return rec


def build_manifest(prep_root: Path, order: list[str] | None = None, n_priority: int = 60):
    """Collect every <prep>/<id>/qc.json into <prep>/manifest.csv (rank = position in the processing order)."""
    import pandas as pd

    rows = [json.loads(p.read_text()) for p in sorted(Path(prep_root).glob("*/qc.json"))]
    df = pd.DataFrame(rows)
    for c in ("shape_raw", "spacing_raw", "shape_1mm"):
        if c in df:
            df[c] = df[c].map(lambda v: "x".join(str(x) for x in v) if isinstance(v, list) else v)
    if order is not None and len(df):
        rank = {s: i for i, s in enumerate(order)}
        df.insert(1, "rank", df["id"].map(rank))
        df.insert(2, "priority", df["rank"] < n_priority)
        df = df.sort_values("rank")
    tmp = Path(prep_root) / ".tmp_manifest.csv"
    df.to_csv(tmp, index=False)
    os.replace(tmp, Path(prep_root) / "manifest.csv")
    return df
