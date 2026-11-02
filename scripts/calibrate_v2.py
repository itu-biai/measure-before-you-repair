"""blind-mri-v2 clinical calibration (D82–D86): where do our 3-D motion, ghosting and spike look like real graded motion?

  uv run python scripts/calibrate_v2.py mrart    # real + simulated reference-free measures, 30 MR-ART subjects
  uv run python scripts/calibrate_v2.py fit      # → bands.json: edges per artifact with bootstrap 95 % intervals
  uv run python scripts/calibrate_v2.py head     # can entropy compare artifact types? whole box vs inside the head (D90)
  uv run python scripts/calibrate_v2.py grid     # 3-D calibration grid on the 10 IXI calibration subjects (D28, D29)

MR-ART (Narai et al. 2022): each subject has a still scan and two motion scans, each scored good / medium / bad by
neuroradiologists; all Stage 1-prepared (../dataset/MR-ART-prep, read-only). Every volume is cut to the v2 box (D75),
centred on its own brain. The same code measures entropy (EFC) and air signal on the real scans and on our artifacts
applied to the same subjects' still scans, each as a change vs the still scan (D84). Output: ../dataset/v2-calibration/.
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "2")

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bamr.config import REPO, load_config  # noqa: E402
from bamr.data import artifacts3d as B  # noqa: E402
from bamr.data.artifacts import LEVELS, ghosting_params, spike_params  # noqa: E402
from bamr.data.metrics import masked_psnr, masked_ssim  # noqa: E402
from bamr.data.refree import air_mask, air_signal, efc  # noqa: E402

DATA = REPO.parent / "dataset"
MRART_PREP, SCORES = DATA / "MR-ART-prep", DATA / "MR-ART" / "derivatives" / "scores.tsv"
OUT = DATA / "v2-calibration"
SEED = 20261022
DRAWS = {"motion": 10, "ghosting": 6, "spike": 6}
SCALE_RANGE = (0.12, 1.0)      # motion: amplitude scale of a prior draw, so the sweep covers mild to beyond bad
TARGET_RANGE = (0.004, 0.06)   # ghosting, spike: whole-brain damage targets of the sweep
QUANTS = {"medium p25": (2, 0.25), "medium median": (2, 0.5), "medium p75": (2, 0.75), "bad median": (3, 0.5)}
N_BOOT = 2000


def load_boxed(prep: Path, sid: str):
    d = prep / sid
    v = nib.load(d / "t1_1mm.nii.gz").get_fdata(dtype=np.float32)
    m = np.asarray(nib.load(d / "brain_mask_1mm.nii.gz").dataobj) > 0
    o = B.box_origin(m)
    return B.cut_box(v, o), B.cut_box(m, o)


def measures(v, base, air=None):
    air = air_mask(v) if air is None else air
    return {"d_air": air_signal(v, air) - base["air"], "d_efc": efc(v) - base["efc"]}


def _threads():
    import torch
    torch.set_num_threads(2)


def subject_job(args):
    """Real measures of the subject's three scans + our sweeps on its still scan."""
    sub, scores, mcfg, gcfg, scfg, si = args
    _threads()
    still, brain = load_boxed(MRART_PREP, f"{sub}_acq-standard")
    air = air_mask(still)
    base = {"air": air_signal(still, air), "efc": efc(still)}
    real = []
    for acq in ("standard", "headmotion1", "headmotion2"):
        sid = f"{sub}_acq-{acq}"
        v = still if acq == "standard" else load_boxed(MRART_PREP, sid)[0]
        real.append({"sub": sub, "acq": acq, "score": int(scores[sid])} | measures(v, base))
    rng = np.random.default_rng([SEED, si])
    sim = []
    for _ in range(DRAWS["motion"]):
        s = float(rng.uniform(*SCALE_RANGE))
        p = B.scale_motion(B.draw_motion3(rng, mcfg), s)
        out = np.abs(B.apply3(still, "motion", p))
        e = float(np.sqrt(np.mean((out - still)[brain] ** 2)))
        sim.append({"sub": sub, "artifact": "motion", "damage": e, "knob": s} | measures(out, base, air))
    for _ in range(DRAWS["ghosting"]):
        n = int(rng.integers(gcfg["num_ghosts"][0], gcfg["num_ghosts"][1] + 1))
        t = float(rng.uniform(*TARGET_RANGE))
        p, e = B._hit3(still, brain, "ghosting", lambda g: ghosting_params(n, g, gcfg["restore"]), B.GHOST_PROBE, t)
        out = np.abs(B.apply3(still, "ghosting", p))
        sim.append({"sub": sub, "artifact": "ghosting", "damage": e, "knob": p["ghost_ratio"]} | measures(out, base, air))
    for _ in range(DRAWS["spike"]):
        pos = B.spike_positions3(rng, scfg, B.to_t(still).shape)
        t = float(rng.uniform(*TARGET_RANGE))
        p, e = B._hit3(still, brain, "spike", lambda a: spike_params(pos, a, still), B.SPIKE_PROBE, t)
        out = np.abs(B.apply3(still, "spike", p))
        sim.append({"sub": sub, "artifact": "spike", "damage": e, "knob": p["stripe_amplitude"]} | measures(out, base, air))
    return real, sim


def run_mrart(cfg, workers):
    c2 = cfg["stage2"]
    sc = pd.read_csv(SCORES, sep="\t")
    scores = dict(zip(sc.bids_name.str.replace("_T1w", "", regex=False), sc.score))
    subs = sorted({p.name.split("_acq-")[0] for p in MRART_PREP.iterdir() if p.is_dir()})
    jobs = [(s, {k: v for k, v in scores.items() if k.startswith(s)}, c2["motion"], c2["ghosting"], c2["spike"], i)
            for i, s in enumerate(subs)]
    real, sim = [], []
    t0 = time.time()
    with get_context("spawn").Pool(workers) as pool:
        for k, (r, s) in enumerate(pool.imap_unordered(subject_job, jobs)):
            real += r
            sim += s
            print(f"{k + 1}/{len(jobs)} {r[0]['sub']} {time.time() - t0:.0f} s", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(real).sort_values(["sub", "acq"]).to_csv(OUT / "real.csv", index=False)
    pd.DataFrame(sim).sort_values(["artifact", "sub", "damage"]).to_csv(OUT / "sim.csv", index=False)


# ---------------------------------------------------------------- fit
def targets(real: pd.DataFrame, col: str) -> dict:
    mv = real[real.acq != "standard"]
    return {k: float(mv[mv.score == g][col].quantile(q)) for k, (g, q) in QUANTS.items()} | \
        {"good median": float(mv[mv.score == 1][col].median())}


def slope(sim: pd.DataFrame, col: str) -> float:
    """Least squares line through the origin: measure = slope × damage (zero damage = the still scan itself)."""
    return float((sim.damage * sim[col]).sum() / (sim.damage ** 2).sum())


def edges_for(real, sim, col):
    t = targets(real, col)
    out = {}
    for a, s in sim.groupby("artifact"):
        k = slope(s, col)
        out[a] = {"slope": k, "r": float(np.corrcoef(s.damage, s[col])[0, 1]),
                  "edges": [t[q] / k for q in QUANTS], "floor": t["good median"] / k}
    return t, out


def run_fit():
    real, sim = pd.read_csv(OUT / "real.csv"), pd.read_csv(OUT / "sim.csv")
    subs = np.array(sorted(real["sub"].unique()))
    res = {}
    rng = np.random.default_rng(SEED)
    for col in ("d_efc", "d_air"):
        t, est = edges_for(real, sim, col)
        boot = {a: [] for a in est}
        for _ in range(N_BOOT):  # resample subjects: real and simulated scans of a subject stay together
            pick = rng.choice(subs, len(subs))
            rb = pd.concat([real[real["sub"] == s] for s in pick])
            sb = pd.concat([sim[sim["sub"] == s] for s in pick])
            _, e = edges_for(rb, sb, col)
            for a in est:
                boot[a].append(e[a]["edges"] + [e[a]["floor"]])
        for a in est:
            b = np.array(boot[a])
            est[a]["ci95"] = [[float(np.quantile(b[:, j], 0.025)), float(np.quantile(b[:, j], 0.975))]
                              for j in range(b.shape[1])]
        res[col] = {"targets": t, "artifacts": est}
    mv = real[real.acq != "standard"]
    res["n_real"] = {int(g): int((mv.score == g).sum()) for g in (1, 2, 3)}
    res["n_sim"] = sim.artifact.value_counts().to_dict()
    res["subjects"] = len(subs)
    # Motion's bands (D84): its entropy edges, the primary measure, cross-checked by air. Entropy and air are dominated
    # by signal in the air, which a spike fills everywhere, so they cannot compare artifact types fairly ("head" step:
    # inside the head the three types change entropy alike per unit of brain damage). Ghosting and spike therefore use
    # motion's clinically calibrated whole-brain damage scale (D90); their entropy-matched edges are kept as a
    # sensitivity bound. The D65 floor: the damage at which our motion differs from the still scan as much as two good
    # scans do (good-scan median entropy).
    e = res["d_efc"]["artifacts"]
    pairs = lambda ed: [[round(x, 4) for x in pair] for pair in zip(ed[:-1], ed[1:])]  # noqa: E731
    res["bands_entropy_matched"] = {a: pairs(e[a]["edges"]) for a in e}
    res["bands"] = {a: pairs(e["motion"]["edges"]) for a in e}
    res["equiv"] = {a: 1.0 for a in e}
    res["equiv_entropy_matched"] = {a: round(e[a]["slope"] / e["motion"]["slope"], 4) for a in e}
    res["min_share"] = round(e["motion"]["floor"], 4)
    (OUT / "bands.json").write_text(json.dumps(res, indent=1) + "\n")
    for col in ("d_efc", "d_air"):
        print(col, {k: round(v, 5) for k, v in res[col]["targets"].items()})
        for a, x in res[col]["artifacts"].items():
            print(f"  {a:9s} r={x['r']:.2f} edges", [round(v, 4) for v in x["edges"]], "floor", round(x["floor"], 4),
                  "ci", [[round(v, 4) for v in c] for c in x["ci95"]])
    print("bands", res["bands"], "equiv", res["equiv"], "min_share", res["min_share"])


# ---------------------------------------------------------------- 3-D calibration grid on IXI (D28, D29)
def grid_job(args):
    sid, native, c2, si = args
    _threads()
    ref, brain = load_boxed(DATA / "IXI-T1-prep", sid)
    base = {"efc": efc(ref), "air": 0.0}
    rows = []
    for ai, a in enumerate(("bias", "motion", "ghosting", "spike", "noise")):
        for li, lv in enumerate(LEVELS):
            out, lab = B.corrupt3(ref, brain, {a: lv}, int(np.random.SeedSequence([SEED, 7, si, ai, li]).generate_state(1)[0]),
                                  c2, native)
            d = float(np.sqrt(np.mean((out.astype(np.float64) - ref)[brain] ** 2)))
            rows.append({"subject": sid, "artifact": a, "level": lv, "damage": d, "psnr": masked_psnr(ref, out, brain),
                         "ssim": masked_ssim(ref, out, brain), "d_efc": efc(out) - base["efc"]})
    return rows


def run_grid(cfg, workers):
    subs = json.loads((cfg["paths"]["calib"] / "subjects.json").read_text())
    jobs = [(s["id"], s["native_sigma"], cfg["stage2"], i) for i, s in enumerate(subs)]
    rows, t0 = [], time.time()
    with get_context("spawn").Pool(workers) as pool:
        for k, r in enumerate(pool.imap_unordered(grid_job, jobs)):
            rows += r
            print(f"{k + 1}/{len(jobs)} {r[0]['subject']} {time.time() - t0:.0f} s", flush=True)
    df = pd.DataFrame(rows).sort_values(["artifact", "level", "subject"])
    df.to_csv(OUT / "grid.csv", index=False)
    print(df.groupby(["artifact", "level"], sort=False)[["damage", "psnr", "ssim", "d_efc"]].median().round(4))


# ---------------------------------------------------------------- can entropy compare artifact types? (D90)
def efc_in(v, m):
    """EFC over the voxels of mask m only (normalised by m's own maximum)."""
    x = np.abs(v[m]).astype(np.float64)
    x = x[x > 0]
    b = x / np.sqrt((x ** 2).sum())
    n = m.sum()
    return float(-(b * np.log(b)).sum() / (np.sqrt(n) * np.log(np.sqrt(n))))


def head_job(args):
    """Entropy over the whole box and inside the head only, for our three artifacts at matched brain damage."""
    sub, cfg2, si = args
    _threads()
    still, brain = load_boxed(MRART_PREP, f"{sub}_acq-standard")
    head = ~air_mask(still)
    b0 = {"efc": efc(still), "efc_head": efc_in(still, head)}
    rng = np.random.default_rng([SEED, 9, si])
    rows = []
    for t in (0.015, 0.03, 0.05):
        p = B.sample_motion3(rng, [t, t], cfg2["motion"], still, brain)
        n = int(rng.integers(2, 5))
        pg, _ = B._hit3(still, brain, "ghosting", lambda g: ghosting_params(n, g, None), B.GHOST_PROBE, t)
        pos = B.spike_positions3(rng, cfg2["spike"], B.to_t(still).shape)
        ps, _ = B._hit3(still, brain, "spike", lambda a: spike_params(pos, a, still), B.SPIKE_PROBE, t)
        for a, q in (("motion", p), ("ghosting", pg), ("spike", ps)):
            out = np.abs(B.apply3(still, a, q))
            rows.append({"sub": sub, "artifact": a, "damage": float(np.sqrt(np.mean((out - still)[brain] ** 2))),
                         "d_efc": efc(out) - b0["efc"], "d_efc_head": efc_in(out, head) - b0["efc_head"]})
    return rows


def run_head(cfg, workers):
    subs = sorted({p.name.split("_acq-")[0] for p in MRART_PREP.iterdir() if p.is_dir()})[::3]
    with get_context("spawn").Pool(workers) as pool:
        rows = sum(pool.map(head_job, [(s, cfg["stage2"], i) for i, s in enumerate(subs)]), [])
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "head_check.csv", index=False)
    sl = {col: {a: slope(g, col) for a, g in df.groupby("artifact")} for col in ("d_efc", "d_efc_head")}
    ratio = {col: {a: round(v / sl[col]["motion"], 2) for a, v in d.items()} for col, d in sl.items()}
    (OUT / "head_check.json").write_text(json.dumps({"subjects": len(subs), "slope": sl, "ratio_to_motion": ratio},
                                                    indent=1) + "\n")
    print("entropy change per unit brain damage, relative to motion:", ratio)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["mrart", "fit", "head", "grid"])
    ap.add_argument("--config", default=str(REPO / "configs" / "dataset_v2.yaml"))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.step == "mrart":
        run_mrart(cfg, args.workers)
    elif args.step == "fit":
        run_fit()
    elif args.step == "head":
        run_head(cfg, args.workers)
    else:
        run_grid(cfg, args.workers)


if __name__ == "__main__":
    sys.exit(main())
