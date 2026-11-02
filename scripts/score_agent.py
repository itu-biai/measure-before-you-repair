"""Task 4 CLI: score the runs of scripts/run_agent.py (D132, D140), after they are done, in this separate process.

  uv run python scripts/score_agent.py                  # every system folder in ../dataset/task4-results/stage1-check
  uv run python scripts/score_agent.py --out stage1-check
  uv run python scripts/score_agent.py --out stage2-heldout --oracle   # first runs ⑤ on the folder's cases (D151)
  options: --workers 2 (⑤'s compute workers)

For each finished case (result.json): the input, the output, the reference with its motion replayed (label params,
bamr.data.artifacts3d.replay3), the reference's brain mask and the label; rules in bamr.eval.score_agent; PSNR, SSIM
and LPIPS of input and output against the clean reference (D158, bamr.eval.quality; the input's once per volume, an
output identical to the input reuses them). Writes
scores.json (one summary per system folder) and rows.csv next to the folders. Refs and labels are read here only, never
by the runner; test volumes (t*) only from Task 5 on. ⑤, the oracle, runs here because it reads the label: outside
blind_guard, with its own cache (task4-results/cache-oracle), after the blind runs.
"""
import argparse
import concurrent.futures as cf
import datetime
import json
import re
import shutil

import nibabel as nib
import numpy as np
import pandas as pd

from bamr.agent.baselines import run_chain
from bamr.agent.compute import Compute
from bamr.agent.loop import CaseLog
from bamr.config import REPO, load_config
from bamr.data.artifacts3d import replay3
from bamr.data.blindio import read_volume
from bamr.eval.quality import Lpips, quality
from bamr.eval.score_agent import ARTIFACTS, QUALITY, case_row, motion_target, summary

CFG = load_config(REPO / "configs" / "dataset_v2.yaml")
DS = CFG["paths"]["out"]
RESULTS = DS.parent / "task4-results"


def run_oracle(root, workers: int = 2, threads: int = 3) -> None:
    """⑤ on every case any system folder of this run was asked for (D151)."""
    ids = sorted({c for f in root.iterdir() if (f / "run.json").exists() and f.name != "oracle"
                  for c in json.loads((f / "run.json").read_text())["cases"]})
    assert all(re.fullmatch(r"[dst]\d{4}", c) for c in ids), ids  # test volumes from Task 5 on
    folder = root / "oracle"
    folder.mkdir(exist_ok=True)
    (folder / "run.json").write_text(json.dumps({"system": "oracle", "cases": ids, "workers": workers,
                                                 "started": datetime.datetime.now().isoformat(timespec="seconds")}))
    compute = Compute(RESULTS / "cache-oracle", workers=workers, blocked=())  # reads labels: outside the guard

    def one(cid):
        d = folder / cid
        if (d / "result.json").exists():
            return
        shutil.rmtree(d, ignore_errors=True)
        lab = json.loads((DS / "labels" / f"{cid}.json").read_text())
        work = RESULTS / "work" / "oracle" / cid
        shutil.rmtree(work, ignore_errors=True)
        res = run_chain(read_volume(DS / "inputs" / f"{cid}.nii.gz"), compute, work, CaseLog(d), mode="oracle",
                        label=lab)
        shutil.rmtree(work, ignore_errors=True)
        print(f"{cid} oracle: {res['version']} · {[s['fix'] for s in res['steps']]} · {res['seconds']:.0f} s", flush=True)
    try:
        with cf.ThreadPoolExecutor(threads) as ex:
            list(ex.map(one, ids))
    finally:
        compute.close()


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="stage1-check")
    p.add_argument("--oracle", action="store_true", help="run ⑤ first, on the cases of this run")
    p.add_argument("--workers", type=int, default=2, help="⑤'s compute workers (Task 5: 1, the 8 GB rule, D166)")
    return p


def main():
    a = parser().parse_args()
    root = RESULTS / a.out
    if a.oracle:
        run_oracle(root, workers=a.workers)
    folders = sorted(d for d in root.iterdir() if d.is_dir() and (d / "run.json").exists())
    done = {}  # case → the folders that finished it; each case's arrays are loaded once and dropped after
    for f in folders:
        for d in sorted(x for x in f.iterdir() if (x / "result.json").exists()):
            assert re.fullmatch(r"[dst]\d{4}", d.name), d.name  # test volumes from Task 5 on
            done.setdefault(d.name, []).append(f)
    rows, lp = [], Lpips()
    for cid in sorted(done):
        lab = json.loads((DS / "labels" / f"{cid}.json").read_text())
        ref = np.asarray(nib.load(DS / "refs" / cid / "reference.nii.gz").dataobj, np.float32)
        brain = np.asarray(nib.load(DS / "refs" / cid / "brain_mask.nii.gz").dataobj) > 0
        x, tgt = read_volume(DS / "inputs" / f"{cid}.nii.gz"), motion_target(ref, lab, replay3)
        q_in = quality(ref, x, brain, lp)
        for f in done[cid]:
            res = json.loads((f / cid / "result.json").read_text())
            y = read_volume(f / cid / "output.nii.gz")
            r = case_row(x, y, tgt, brain, lab, res)
            q_out = q_in if np.array_equal(y, x) else quality(ref, y, brain, lp)
            rows.append({"folder": f.name, "case_id": cid, **r, **{f"{k}_in": q_in[k] for k in QUALITY},
                         **{f"{k}_out": q_out[k] for k in QUALITY}})
            del y
        del x, tgt, ref, brain
    rows.sort(key=lambda r: (r["folder"], r["case_id"]))
    scores = {f.name: summary([r for r in rows if r["folder"] == f.name]) for f in folders}
    (root / "scores.json").write_text(json.dumps(scores, indent=1))
    flat = [{**{k: v for k, v in r.items() if not isinstance(v, (dict, list))},
             **{f"right_{k}": v for k, v in r["right"].items()}, **{f"said_{k}": v for k, v in r["said"].items()},
             "repairs": "+".join(r["repairs"])} for r in rows]
    pd.DataFrame(flat).to_csv(root / "rows.csv", index=False)
    print(f"{'folder':<22}{'n':>3}{'removed':>9}{'worse':>6}{'art':>7}{'rescan':>7}{'calls':>6}{'bad':>5}"
          f"{'fuse':>5}{'err':>4}{'sec':>7}{'$':>8}{'check':>8}")
    for f in folders:  # cases the run was asked for but did not finish: shown, never silently dropped
        want = json.loads((f / "run.json").read_text())["cases"]
        scores[f.name]["missing"] = [c for c in want if not (f / c / "result.json").exists()]
    (root / "scores.json").write_text(json.dumps(scores, indent=1))
    for k, s in scores.items():
        if not s["n"]:
            print(f"{k:<22} no finished case; missing {len(s['missing'])}")
            continue
        if s["missing"]:
            print(f"{k:<22} missing: {', '.join(s['missing'])}")
        rm = f"{s['removed_median']:.3f}" if s["removed_median"] is not None else "–"
        chk = f"{s['check_pass']['all']}/{s['checked']}" if s["checked"] else "–"
        print(f"{k:<22}{s['n']:>3}{rm:>9}{s['worse']:>6}{s['artifacts_right']:>4}/{s['artifacts_total']:<3}"
              f"{s['rescan_right']:>3}/{s['n']:<3}{s['calls']:>6}{s['bad_calls']:>5}{s['fuses']:>5}{s['errors']:>4}"
              f"{s['seconds_median']:>7.0f}{s['cost']:>8.3f}{chk:>8}")
    print("per artifact right:", {k: s["per_artifact_right"] for k, s in scores.items()})
    for k, s in scores.items():  # D158: mean over the volumes holding an artifact
        if s.get("quality"):
            q = s["quality"]
            print(f"{k:<22} n={q['n']:<3} " + "  ".join(
                f"{m.upper()} {q[m + '_in']['mean']:.4g} → {q[m + '_out']['mean']:.4g}" for m in QUALITY))
    print("→", root / "scores.json", root / "rows.csv", "· artifacts:", ", ".join(ARTIFACTS))


if __name__ == "__main__":
    main()
