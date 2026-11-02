"""Task 3 CLI: run a fix tool on the dev volumes of blind-mri-v2 and score it (D115).

  uv run python scripts/score_fixes.py run --fix spike --workers 3   # → ../dataset/task3-results/stage1-spike/rows.jsonl
  uv run python scripts/score_fixes.py run --fix bias --workers 1    # HD-BET on the GPU: one process
  uv run python scripts/score_fixes.py run --fix noise --workers 4   # DRUNet: the workers take turns on the GPU
  uv run python scripts/score_fixes.py run --fix ghosting --workers 3  # FFTs on the CPU
  uv run python scripts/score_fixes.py score --fix spike             # → scores.json, rows.csv, the target table
The fix gets the input array alone, inside a guard that blocks refs, labels and the manifest (D99 #10); this script
reads them only after the fix has run. Each fix is scored against the volume without its artifact: the label replayed
without it, the same noise draw (before-doc §6), RMS over the reference's brain, WM = 1. Only dev sweep (s*) and dev
mixed (d*) are read; never a test volume (D72). `run` appends each row as it comes and skips the cases already there,
so a stopped run restarts where it was; the sorted rows.jsonl is written at the end. Output volumes are not kept.
Noise (Stage 2 §11) also reports, per volume: the split into tissue borders (top 25 % of the target's smoothed
gradient in the brain) and the rest, the two floors from the label's σ (the scan's own σn; σn·σa/√(σn² + σa²)), and
for every changed volume the error against the reference and the change in the brain (the false alarms).
Ghosting (Stage 3 §14) also reports the air error (≥ 8 voxels outside the target's head, as the pilot) and the motion
tool's answer on each changed output (it says "cannot tell" while ghosting is there).
"""
import argparse
import importlib
import json
import os
import time
from multiprocessing import Lock, Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")

import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

from bamr.config import REPO, load_config  # noqa: E402
from bamr.data.artifacts3d import replay3  # noqa: E402
from bamr.data.blindio import read_volume  # noqa: E402
from bamr.eval.guard import blind_guard  # noqa: E402
from bamr.eval.score_fixes import score_fix  # noqa: E402
from bamr.eval.score_noise import case_ids, dev_only  # noqa: E402
from bamr.tools.bias import LINE as BIAS_LINE  # noqa: E402
from bamr.tools.regions import head_mask  # noqa: E402

CFG = load_config(REPO / "configs" / "dataset_v2.yaml")
DS = CFG["paths"]["out"]
FIXES = {"spike": "stage1-spike", "bias": "stage1-bias", "noise": "stage2-noise", "ghosting": "stage3-ghosting"}
AIR_GAP = 8  # ghosting: the air error ≥ 8 voxels outside the target's head (the pilot's)
BORDER_TOP = 0.25  # noise: tissue borders = the top quarter of the target's gradient in the brain (as the pilot)
BLOCKED = (DS / "refs", DS / "labels", DS / "manifest.csv")  # closed to the fixes while they run (D99 #10)


def out_dir(fix: str):
    return DS.parent / "task3-results" / FIXES[fix]


def fix_fn(fix: str):
    """Imported when used: the bias fix loads HD-BET (torch), which the spike fix never needs."""
    return getattr(importlib.import_module(f"bamr.fixes.{fix}"), f"fix_{fix}")


def ids(man: pd.DataFrame) -> list[str]:
    return case_ids(man)


def _rms(a, b, m) -> float:
    return float(np.sqrt(np.mean((np.asarray(a, np.float64) - b)[m] ** 2)))


def _after_present(rep: dict, fix: str):
    """The Task 2 tool's yes/no on the output: the spike, noise and ghosting fixes ask their tool themselves (before,
    when the volume was left); the bias fix reads N4 again."""
    ev = rep["evidence"]
    if fix == "ghosting":
        return (ev.get("ghost_after") or {}).get("present", ev.get("present_before"))
    if fix == "noise":
        return ev.get("present_after", ev.get("present_before"))
    if fix == "spike":
        return (ev.get("spike_tool_after") or ev.get("spike_tool_before") or {}).get("present")
    r = ev.get("reading_after", ev.get("reading_before"))
    return None if r is None else bool(r >= BIAS_LINE)


def _error(rep: dict) -> bool:
    """The fix, or its Task 2 tool before or after it, gave an error instead of a reading (review I1)."""
    ev = rep["evidence"]
    after = " ".join((ev.get(k) or {}).get("why") or "" for k in ("spike_tool_after", "ghost_after", "motion_after"))
    after += ev.get("why_after") or ""
    return rep["why"].startswith("fix error") or "tool error" in rep["why"] or "tool error" in after


GPU_WAIT = [0.0]  # seconds this worker waited for its turn on the GPU, during the current fix call


def gpu_turns(fn, lock):
    """fn run while holding the shared GPU lock; the wait for the lock is added to GPU_WAIT, not to the fix's time."""
    def run(*a, **k):
        t = time.perf_counter()
        with lock:
            GPU_WAIT[0] += time.perf_counter() - t
            return fn(*a, **k)
    return run


def _init(lock):
    """Noise: the workers take turns on the GPU (DRUNet), so each fix's time is its own while the others read volumes,
    run the noise tool and score; the fix's code is unchanged, only the name it calls is wrapped in this process."""
    if lock is not None:
        import bamr.fixes.noise as fn
        fn.denoise3 = gpu_turns(fn.denoise3, lock)


def _job(args):
    cid, fix = args
    x = read_volume(DS / "inputs" / f"{dev_only(cid)}.nii.gz")
    GPU_WAIT[0] = 0.0
    t = time.perf_counter()
    with blind_guard(*BLOCKED):
        y, rep = fix_fn(fix)(x)
    sec = time.perf_counter() - t - GPU_WAIT[0]
    rep = json.loads(rep.to_json())
    # the scorer's side, after the fix has run: the answers
    lab = json.loads((DS / "labels" / f"{cid}.json").read_text())
    has = fix in lab["artifacts"]
    r = {"case_id": cid, "artifacts": "+".join(lab["artifacts"]) or "clean", "has": has, "changed": rep["changed"],
         "identical": bool(y.dtype == x.dtype and y.shape == x.shape and y.tobytes() == x.tobytes()),
         "seconds": round(sec, 3), "gpu_wait": round(GPU_WAIT[0], 3), "zeros_kept": bool((y[x == 0] == 0).all()),
         "change": rep["change"],
         "after_present": _after_present(rep, fix), "own_in": None, "own_out": None, "report": rep}
    if fix == "ghosting":
        m = rep["evidence"].get("motion_after") or {}
        r |= {"motion_after": m.get("present"), "motion_after_value": m.get("value"), "own_air_in": None,
              "own_air_out": None}
    if fix == "spike":
        r |= {"own_air_in": None, "own_air_out": None, "n_repaired": sum(len(q["k"]) for q in
                                                                         rep["evidence"].get("rounds", [])),
              "n_true": len(lab["params"]["spike"]["spikes_positions"]) if has else 0}
    if has or (fix == "noise" and rep["changed"]):
        ref = np.asarray(nib.load(DS / "refs" / cid / "reference.nii.gz").dataobj, np.float32)
        brain = np.asarray(nib.load(DS / "refs" / cid / "brain_mask.nii.gz").dataobj) > 0
    if fix == "noise" and rep["changed"]:
        r |= {"ref_in": _rms(x, ref, brain), "ref_out": _rms(y, ref, brain),
              "change_brain": _rms(y, x, brain)}
    if has:
        tgt = replay3(ref, dict(lab, artifacts=[a for a in lab["artifacts"] if a != fix]))
        r["own_in"], r["own_out"] = _rms(x, tgt, brain), _rms(y, tgt, brain)
        if fix == "spike":
            air = ~head_mask(x)
            r["own_air_in"], r["own_air_out"] = _rms(x, tgt, air), _rms(y, tgt, air)
        if fix == "ghosting":
            air = ndi.distance_transform_edt(~head_mask(tgt)) > AIR_GAP
            r["own_air_in"], r["own_air_out"] = _rms(x, tgt, air), _rms(y, tgt, air)
        if fix == "noise":
            g = ndi.gaussian_gradient_magnitude(tgt.astype(np.float64), 1.0)
            border = brain & (g >= np.quantile(g[brain], 1 - BORDER_TOP))
            flat = brain & ~border
            sn, sa = lab["native_sigma"], lab["params"]["noise"]["sigma_added"]
            r |= {"border_in": _rms(x, tgt, border), "border_out": _rms(y, tgt, border),
                  "flat_in": _rms(x, tgt, flat), "flat_out": _rms(y, tgt, flat),
                  "floor_clean": sn, "floor_best": sn * sa / float(np.hypot(sn, sa))}
    return r


def run(fix: str, workers: int) -> None:
    man = pd.read_csv(DS / "manifest.csv")
    out = out_dir(fix)
    out.mkdir(parents=True, exist_ok=True)
    part = out / "rows.partial.jsonl"
    done = {json.loads(line)["case_id"] for line in part.read_text().splitlines()} if part.exists() else set()
    jobs = [(c, fix) for c in ids(man) if c not in done]
    t0 = time.time()
    with Pool(workers, _init, (Lock() if fix == "noise" else None,)) as pool, open(part, "a") as fh:
        for k, r in enumerate(pool.imap_unordered(_job, jobs), 1):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            print(f"[{fix}] {k}/{len(jobs)} {r['case_id']} {r['seconds']:.1f} s changed={r['changed']}", flush=True)
    rows = [json.loads(line) for line in part.read_text().splitlines()]
    with open(out / "rows.jsonl", "w") as fh:
        for r in sorted(rows, key=lambda r: r["case_id"]):
            fh.write(json.dumps(r) + "\n")
    part.unlink()
    print(f"{fix}: {len(jobs)} volumes in {time.time() - t0:.0f} s ({len(done)} from an earlier run) → "
          f"{out / 'rows.jsonl'}")


def motion_summary(df: pd.DataFrame) -> dict:
    """Ghosting (§11), reported only: the motion tool's answer on the changed outputs, alone and mixed (ghosting with
    other artifacts); a changed volume without ghosting (a false alarm) is in neither."""
    ch = df[df.changed.astype(bool) & df.has.astype(bool)]
    return {g: {"yes": int(d.motion_after.eq(True).sum()), "no": int(d.motion_after.eq(False).sum()),
                "cannot_tell": int(d.motion_after.isna().sum())}
            for g, d in (("alone", ch[ch.artifacts == "ghosting"]), ("mixed", ch[ch.artifacts != "ghosting"]))}


def score(fix: str) -> None:
    out = out_dir(fix)
    rows = [json.loads(line) for line in (out / "rows.jsonl").read_text().splitlines()]
    df = pd.DataFrame(rows)
    df["own_in"], df["own_out"] = df.own_in.astype(float), df.own_out.astype(float)
    df["error"] = [_error(r["report"]) for r in rows]
    rep = score_fix(df, fix)
    rep["errors"] = sorted(df[df.error].case_id)
    if fix == "noise":  # reported only (§11): the floors, borders vs the rest, the false alarms' change
        a = df[df.has & (df.artifacts == "noise")]
        fa = df[~df.has & df.changed]
        rep["noise"] = {"floor_clean_removed_median": float((1 - a.floor_clean / a.own_in).median()),
                        "floor_best_removed_median": float((1 - a.floor_best / a.own_in).median()),
                        "border_out_median": float(a.border_out.median()),
                        "flat_out_median": float(a.flat_out.median()),
                        "false_alarm_change_median": float(fa.change_brain.median()) if len(fa) else None,
                        "gpu_wait_median": float(df.gpu_wait.median())}
    if fix == "ghosting":
        rep["ghosting"] = motion_summary(df)
    (out / "scores.json").write_text(json.dumps(rep, indent=1))
    df.drop(columns=["report"]).assign(what=[r["report"]["what"] for r in rows],
                                       why=[r["report"]["why"] for r in rows]).to_csv(out / "rows.csv", index=False)
    for name, t in rep["targets"].items():
        print(f"{name:20s} want {t['want']!s:8s} got {t['got']!s:22s} {'pass' if t['pass'] else 'FAIL'}")
    print("alone:", {k: round(v, 4) if isinstance(v, float) else v for k, v in rep["alone"].items()})
    print("mixed:", {k: round(v, 4) if isinstance(v, float) else v for k, v in rep["mixed"].items()})
    print("no harm:", rep["no_harm"], "missed:", rep["missed"], "errors:", rep["errors"], "→", out / "scores.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["run", "score"])
    ap.add_argument("--fix", choices=list(FIXES), required=True)
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    run(a.fix, a.workers) if a.step == "run" else score(a.fix)


if __name__ == "__main__":
    main()
