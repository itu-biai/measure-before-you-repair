"""D46: anchor the motion bands on real motion. MR-ART (Nárai et al. 2022, OpenNeuro ds004173, CC0) has, per person,
one still T1w and two scans with deliberate head motion (low, high), each scored good / medium / bad (1 / 2 / 3) by
neuroradiologists. We measure the real image error in our unit (brain RMS, WM = 1, on our axial canvas slices) and
our simulated motion on the same still slices.

  uv run python scripts/mrart_check.py select      # 30 subjects (all 9 with a 'good' motion scan + 21 at random)
  uv run python scripts/mrart_check.py download    # 90 T1w scans (≈ 1 GB) → ../dataset/MR-ART (then read-only)
  uv run python scripts/mrart_check.py prep        # Stage 1 unchanged: HD-BET, N4, 1 mm RAS, WM = 1 → MR-ART-prep
  uv run python scripts/mrart_check.py measure     # register motion → still, real errors + our simulation → csv
Every step is restartable and skips finished work.
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import copy  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import urllib.request  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import SimpleITK as sitk  # noqa: E402

from bamr.config import REPO, load_config  # noqa: E402
from bamr.data import artifacts as A  # noqa: E402
from bamr.data.hdbet_batch import MASK_NAME, run_hdbet  # noqa: E402
from bamr.data.slices import axial_canvas, canvas_origin, zstar  # noqa: E402
from bamr.data.subjects import Subject  # noqa: E402
from bamr.data.worker import process_subject  # noqa: E402

S3 = "https://s3.amazonaws.com/openneuro.org/ds004173"
RAW = (REPO / "../dataset/MR-ART").resolve()
PREP = (REPO / "../dataset/MR-ART-prep").resolve()
OUT = (REPO / "../dataset/stage2-mrart").resolve()
ACQS = ("standard", "headmotion1", "headmotion2")
N_SUBJECTS = 30
SIM_DRAWS = 10  # motion prior draws per slice and level, as in the IXI calibration (D32)


def scan_id(sub: str, acq: str) -> str:
    return f"{sub}_acq-{acq}"


def _get(url: str, dst: Path, tries: int = 4) -> None:
    for t in range(tries):
        try:
            tmp = dst.with_name(".tmp_" + dst.name)
            with urllib.request.urlopen(url, timeout=120) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            os.replace(tmp, dst)
            return
        except OSError:
            if t == tries - 1:
                raise
            time.sleep(5 * (t + 1))


def scores() -> pd.DataFrame:
    f = RAW / "derivatives" / "scores.tsv"
    if not f.exists():
        f.parent.mkdir(parents=True, exist_ok=True)
        _get(f"{S3}/derivatives/scores.tsv", f)
    s = pd.read_csv(f, sep="\t")
    s["sub"] = s.bids_name.str.split("_").str[0]
    s["acq"] = s.bids_name.str.extract(r"acq-([a-z0-9]+)_")[0]
    return s.pivot(index="sub", columns="acq", values="score")


def select(cfg: dict) -> pd.DataFrame:
    """Reference must be scored good. 'Good' motion scans are rare (10 of 295), so all subjects with one are taken;
    the rest are drawn at random with the project seed."""
    sc = scores().dropna()
    ok = sc[sc.standard == 1]
    rare = ok[(ok.headmotion1 == 1) | (ok.headmotion2 == 1)]
    rest = ok.drop(rare.index)
    rng = np.random.default_rng(cfg["seed"])
    pick = sorted(rare.index) + sorted(rng.choice(sorted(rest.index), N_SUBJECTS - len(rare), replace=False))
    sel = ok.loc[pick].astype(int).reset_index()
    sel["why"] = ["has a good motion scan"] * len(rare) + ["random"] * (N_SUBJECTS - len(rare))
    OUT.mkdir(parents=True, exist_ok=True)
    sel.to_csv(OUT / "selection.csv", index=False)
    return sel


def _download_scan(sub: str, acq: str) -> str:
    name = f"{scan_id(sub, acq)}_T1w.nii.gz"
    dst = RAW / sub / "anat" / name
    if not dst.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        _get(f"{S3}/{sub}/anat/{name}", dst)
        nib.load(dst).get_fdata()  # fails on a truncated file
    return name


def download(sel: pd.DataFrame, threads: int = 6) -> None:
    """One S3 connection gives ≈ 0.7 MB/s, so fetch several scans at once."""
    from concurrent.futures import ThreadPoolExecutor

    jobs = [(s, a) for s in sel["sub"] for a in ACQS]
    with ThreadPoolExecutor(threads) as ex:
        for k, name in enumerate(ex.map(lambda j: _download_scan(*j), jobs), 1):
            print(f"[download] {k}/{len(jobs)} {name}", flush=True)


def _subjects(sel: pd.DataFrame) -> list[Subject]:
    return [Subject(scan_id(s, a), "MR-ART", RAW / s / "anat" / f"{scan_id(s, a)}_T1w.nii.gz")
            for s in sel["sub"] for a in ACQS]


def _init_worker() -> None:
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)


def prep(cfg: dict, sel: pd.DataFrame, workers: int) -> None:
    subs = _subjects(sel)
    run_hdbet([(s.path, PREP / s.id / MASK_NAME) for s in subs], device=cfg["stage1"]["hdbet"]["device"],
              tta=cfg["stage1"]["hdbet"]["tta"])
    todo = [s for s in subs if not (PREP / s.id / "qc.json").exists()]
    with ProcessPoolExecutor(workers, initializer=_init_worker) as ex:
        futs = {ex.submit(process_subject, s, PREP / s.id / MASK_NAME, PREP / s.id, cfg): s for s in todo}
        for k, f in enumerate(as_completed(futs), 1):
            r = f.result()
            print(f"[prep] {k}/{len(todo)} {futs[f].id} {r.get('t_total', 0):.0f}s err={r.get('error')}", flush=True)


# ---------------------------------------------------------------- registration and errors
def _sitk(vol: np.ndarray, affine: np.ndarray) -> sitk.Image:
    """RAS 1 mm array (i, j, k) → SimpleITK image; both scans use the same convention, so only consistency matters."""
    im = sitk.GetImageFromArray(np.ascontiguousarray(vol.transpose(2, 1, 0)).astype(np.float32))
    im.SetOrigin(tuple(float(v) for v in affine[:3, 3]))
    return im


def register(still: np.ndarray, still_aff, brain: np.ndarray, moving: np.ndarray,
             moving_aff) -> tuple[np.ndarray, list, float]:
    """Rigid (6 dof) moving → still, mean squares inside the dilated still brain, then B-spline resampling onto the
    still grid (Stage 1 interpolation). Mean squares minimises our own error, so real errors are a lower bound.
    Also returns k: the share of white-noise variance that survives the same resampling (B-spline smooths noise)."""
    f, m = _sitk(still, still_aff), _sitk(moving, moving_aff)
    mask = sitk.BinaryDilate(_sitk(brain.astype(np.float32), still_aff) > 0.5, [3, 3, 3])
    init = sitk.CenteredTransformInitializer(f, m, sitk.Euler3DTransform(),
                                             sitk.CenteredTransformInitializerFilter.GEOMETRY)
    r = sitk.ImageRegistrationMethod()
    r.SetMetricAsMeanSquares()
    r.SetMetricFixedMask(mask)
    r.SetMetricSamplingStrategy(r.NONE)
    r.SetInterpolator(sitk.sitkLinear)
    r.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0, minStep=1e-4, numberOfIterations=300,
                                              relaxationFactor=0.5)
    r.SetOptimizerScalesFromPhysicalShift()
    r.SetShrinkFactorsPerLevel([4, 2, 1])
    r.SetSmoothingSigmasPerLevel([2, 1, 0])
    r.SetInitialTransform(init, inPlace=False)
    t = r.Execute(f, m)
    out = sitk.Resample(m, f, t, sitk.sitkBSpline, 0.0)
    noise = _sitk(np.random.default_rng(0).standard_normal(moving.shape).astype(np.float32), moving_aff)
    k = float(sitk.GetArrayFromImage(sitk.Resample(noise, f, t, sitk.sitkBSpline, 0.0)).transpose(2, 1, 0)[brain].var())
    return np.clip(sitk.GetArrayFromImage(out).transpose(2, 1, 0), 0, None), list(t.GetParameters()), k


def register_2d(ref: np.ndarray, brain: np.ndarray, out: np.ndarray) -> np.ndarray:
    """The same alignment for a simulated slice: rigid 2-D, mean squares in the dilated brain, B-spline."""
    f, m = sitk.GetImageFromArray(ref.astype(np.float32)), sitk.GetImageFromArray(out.astype(np.float32))
    mask = sitk.BinaryDilate(sitk.GetImageFromArray(brain.astype(np.uint8)), [3, 3])
    r = sitk.ImageRegistrationMethod()
    r.SetMetricAsMeanSquares()
    r.SetMetricFixedMask(mask)
    r.SetInterpolator(sitk.sitkLinear)
    r.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0, minStep=1e-4, numberOfIterations=200,
                                              relaxationFactor=0.5)
    r.SetOptimizerScalesFromPhysicalShift()
    r.SetShrinkFactorsPerLevel([2, 1])
    r.SetSmoothingSigmasPerLevel([1, 0])
    r.SetInitialTransform(sitk.CenteredTransformInitializer(f, m, sitk.Euler2DTransform()), inPlace=False)
    t = r.Execute(f, m)
    return np.clip(sitk.GetArrayFromImage(sitk.Resample(m, f, t, sitk.sitkBSpline, 0.0)), 0, None)


def _rms(a, b, m) -> float:
    return float(np.sqrt(np.mean((a - b)[m] ** 2)))


def air_mask(ref: np.ndarray, margin: int = 5) -> np.ndarray:
    """Air at least `margin` px away from the head (ref < 0.05 · WM), where ghosts show without anatomy."""
    from scipy import ndimage as ndi

    return ~ndi.binary_dilation(ref > 0.05, iterations=margin)


def air_excess(out: np.ndarray, ref: np.ndarray, air: np.ndarray) -> float:
    """Mean extra signal in the air (WM = 1): ghost energy, rough (D46). Pixels outside either field of view skipped."""
    m = air & (out > 0) & (ref > 0)
    return float(np.mean(out[m] - ref[m]))


def _load(sid: str):
    d = PREP / sid
    t1 = nib.load(d / "t1_1mm.nii.gz")
    return (np.asarray(t1.dataobj, dtype=np.float32), t1.affine,
            np.asarray(nib.load(d / "brain_mask_1mm.nii.gz").dataobj) > 0, json.loads((d / "qc.json").read_text()))


def _seed(base: int, *keys: int) -> int:
    return int(np.random.SeedSequence([base, 46, *keys]).generate_state(1)[0])


def measure_subject(si: int, sub: str, cfg: dict) -> tuple[list[dict], list[dict]]:
    """Real: each motion scan registered to the still, brain RMS per canvas slice. Sim: our motion prior on the same
    still slices, error before (= our label) and after the same alignment."""
    still, s_aff, brain, s_qc = _load(scan_id(sub, "standard"))
    k0, origin = zstar(brain), canvas_origin(brain)
    offsets = [0] + list(cfg["stage2"]["calibration"]["offsets_mm"])
    sig = float(s_qc["sigma_lstd_1mm_measured"])
    cut = lambda v: np.stack([axial_canvas(v, k0 + o, origin) for o in offsets])  # noqa: E731
    keep = {"still": cut(still), "brain": cut(brain), "offsets_mm": np.array(offsets)}
    real, sim = [], []
    for acq in ACQS[1:]:
        mov, m_aff, _, m_qc = _load(scan_id(sub, acq))
        reg, params, k = register(still, s_aff, brain, mov, m_aff)
        keep[acq] = cut(reg)
        for o in offsets:
            ref, br, r = (axial_canvas(v, k0 + o, origin) for v in (still, brain, reg))
            e = _rms(r, ref, br)
            real.append({"sub": sub, "acq": acq, "offset_mm": o, "rms": e, "native_sigma": sig,
                         "air_excess": air_excess(r, ref, air_mask(ref)),
                         # thermal noise: the still σ for both scans (same session and protocol; the motion scan's
                         # own σ estimate is inflated by the artifact), the moved scan's share shrunk by k
                         "noise_k": k, "rms_corr": float(np.sqrt(max(e**2 - (1 + k) * sig**2, 0.0))),
                         "rms_3d": _rms(reg, still, brain) if o == 0 else np.nan, "rigid": params,
                         "moving_qc_ok": bool(m_qc.get("all_ok")), "moving_error": m_qc.get("error")})
    np.savez_compressed(OUT / "slices" / f"{sub}.npz", **keep)
    c2 = copy.deepcopy(cfg["stage2"])
    c2["motion"]["rms_band"] = None  # the prior, as in calibrate_stage2.motion_prior
    for oi, o in enumerate(offsets):
        ref, br = axial_canvas(still, k0 + o, origin), axial_canvas(brain, k0 + o, origin)
        for li, lev in enumerate(A.LEVELS):
            for d in range(SIM_DRAWS):
                p = A.sample_motion(np.random.default_rng(_seed(cfg["seed"], si, oi, li, d)), lev, c2["motion"], ref, br)
                out = np.abs(A._apply(ref, "motion", p))
                sim.append({"sub": sub, "offset_mm": o, "artifact": "motion", "level": lev, "draw": d,
                            "knob": p["motion_rms"], "rms_label": p["motion_rms"],
                            "rms_aligned": _rms(register_2d(ref, br, out), ref, br),
                            "air_excess": air_excess(out, ref, air_mask(ref))})
                rng = np.random.default_rng(_seed(cfg["seed"], si, oi, li, d, 1))
                g = A.sample_ghosting(rng, c2["ghosting"]["rms_band"][li], c2["ghosting"], ref, br)
                out = np.abs(A._apply(ref, "ghosting", g))
                sim.append({"sub": sub, "offset_mm": o, "artifact": "ghosting", "level": lev, "draw": d,
                            "knob": g["ghost_ratio"], "rms_label": _rms(out, ref, br), "rms_aligned": np.nan,
                            "air_excess": air_excess(out, ref, air_mask(ref))})
    return real, sim


def measure(cfg: dict, sel: pd.DataFrame, workers: int) -> None:
    (OUT / "slices").mkdir(parents=True, exist_ok=True)
    done = OUT / "per_subject"
    done.mkdir(exist_ok=True)
    todo = [(i, s) for i, s in enumerate(sel["sub"]) if not (done / f"{s}.json").exists()]
    with ProcessPoolExecutor(workers, initializer=_init_worker) as ex:
        futs = {ex.submit(measure_subject, i, s, cfg): s for i, s in todo}
        for k, f in enumerate(as_completed(futs), 1):
            real, sim = f.result()
            (done / f"{futs[f]}.json").write_text(json.dumps({"real": real, "sim": sim}))
            print(f"[measure] {k}/{len(todo)} {futs[f]}", flush=True)
    real, sim = [], []
    for s in sel["sub"]:
        r = json.loads((done / f"{s}.json").read_text())
        real += r["real"]
        sim += r["sim"]
    sc = sel.set_index("sub")
    real = pd.DataFrame(real)
    real["score"] = [int(sc.loc[s, a]) for s, a in zip(real["sub"], real["acq"])]
    real.to_csv(OUT / "real.csv", index=False)
    pd.DataFrame(sim).to_csv(OUT / "sim.csv", index=False)
    print(real.groupby("score")[["rms", "rms_corr"]].describe().round(3).to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["select", "download", "prep", "measure", "all"])
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    cfg = load_config()
    sel = select(cfg) if args.step in ("select", "all") or not (OUT / "selection.csv").exists() \
        else pd.read_csv(OUT / "selection.csv")
    if args.step == "select":
        print(sel.to_string())
        print(pd.Series(sel[["headmotion1", "headmotion2"]].to_numpy().ravel()).value_counts().sort_index())
    if args.step in ("download", "all"):
        download(sel)
    if args.step in ("prep", "all"):
        prep(cfg, sel, args.workers)
    if args.step in ("measure", "all"):
        measure(cfg, sel, args.workers)


if __name__ == "__main__":
    sys.exit(main())
