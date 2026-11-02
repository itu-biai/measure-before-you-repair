"""Task 2 CLI: run a blind tool on the dev volumes of blind-mri-v2, calibrate its scale, score it (D72, D98).

  uv run python scripts/score_tools.py run --tool spike        # → ../dataset/task2-results/stage2-spike/findings.jsonl
  uv run python scripts/score_tools.py calibrate --tool spike  # prints the median raw/label on the 30 dev-sweep volumes
  uv run python scripts/score_tools.py score --tool spike      # → scores.json, findings.csv, and the target table
  uv run python scripts/score_tools.py promise                 # spike volumes: noise before / after removing the peaks
Tools: noise (Stage 1, the default), spike, ghosting (Stage 2), bias, motion (Stage 3). Only dev sweep (s*) and dev
mixed (d*) are read; never a test volume. The tool gets the input array alone, inside a guard that blocks refs and
labels (D99 #10): this script reads the labels, the tool does not. `run` also does the zero-air test (D69) on every
input with exact zeros. `run` appends each finding to findings.partial.jsonl as it comes and skips the cases already
there, so a stopped run restarts where it was (Stage 3 R7); the sorted findings.jsonl is written at the end.
"""
import argparse
import importlib
import json
import os
import time
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from bamr.config import REPO, load_config  # noqa: E402
from bamr.data.blindio import read_volume  # noqa: E402
from bamr.eval.guard import blind_guard  # noqa: E402
from bamr.eval.score_kspace import (DAMAGE_BANDS, despike, ghost_where_ok, label_spike_k,  # noqa: E402
                                    score_kspace, spike_where_ok)
from bamr.eval.score_noise import case_ids, dev_only, level_of, same_decision, score_noise  # noqa: E402
from bamr.eval.score_s3 import BIAS_BANDS, score_s3  # noqa: E402
from bamr.tools import ghosting as G  # noqa: E402
from bamr.tools.noise import noise_tool  # noqa: E402

CFG = load_config(REPO / "configs" / "dataset_v2.yaml")
DS = CFG["paths"]["out"]
TOOLS = {"noise": "stage1-noise", "spike": "stage2-spike", "ghosting": "stage2-ghosting", "bias": "stage3-bias",
         "motion": "stage3-motion"}
LABEL_RMS = {"spike": "spike_rms", "ghosting": "ghosting_rms", "bias": "target_maxmin", "motion": "motion_rms"}
BLOCKED = (DS / "refs", DS / "labels", DS / "manifest.csv")  # closed to the tools while they run (D99 #10)


def out_dir(tool: str):
    return DS.parent / "task2-results" / TOOLS[tool]


def tool_fn(tool: str):
    """Imported when used: the bias tool loads HD-BET (torch), which the other tools never need."""
    return getattr(importlib.import_module(f"bamr.tools.{tool}"), f"{tool}_tool")


def picture_cases(man: pd.DataFrame, tool: str) -> set[str]:
    """Evidence pictures for a fixed few. Noise (Stage 1): noise and spike at each level, borderline ghosting / bias /
    motion, the first three dev-mixed volumes with noise. Spike, ghosting: the tool's artifact at each level,
    borderline of every other artifact, the first three dev-mixed volumes with the artifact."""
    sw = man[man.set == "sweep"].sort_values("case_id")
    each = ("noise", "spike") if tool == "noise" else (tool,)
    pick = [sw[sw.severity == f"{a}:{lv}"].case_id.iloc[0] for a in each for lv in ("mild", "moderate", "borderline")]
    pick += [sw[sw.severity == f"{a}:borderline"].case_id.iloc[0]
             for a in ("noise", "spike", "ghosting", "bias", "motion") if a not in each]
    dv = man[(man.set == "dev") & man.artifacts.fillna("").str.contains(tool)].sort_values("case_id")
    return set(pick) | set(dv.case_id.iloc[:3])


def _blind(fn, x, **kw):
    """The tool call, with refs, labels and the manifest closed (D99 #10)."""
    with blind_guard(*BLOCKED):
        return fn(x, **kw)


def _job(args):
    cid, pic, tool = args
    fn = tool_fn(tool)
    x = read_volume(DS / "inputs" / f"{cid}.nii.gz")
    t = time.perf_counter()
    f = _blind(fn, x, picture=(out_dir(tool) / "pictures" / f"{cid}.png") if pic else None)
    sec = time.perf_counter() - t
    za = None
    z = x == 0  # the scorer may look at zeros; the tool may not (D69)
    if z.any():
        rng = np.random.default_rng([69, int(cid[1:]), ord(cid[0])])
        y = x.copy()
        y[z] = 0.005 * np.hypot(rng.standard_normal(z.sum()), rng.standard_normal(z.sum()))
        g = _blind(fn, y)
        za = {"zeros": int(z.sum()), "present": g.present, "value": g.value, "raw": g.evidence.get("raw")}
    return {"case_id": cid, "seconds": round(sec, 3), "finding": json.loads(f.to_json()), "zero_air": za}


def run(tool: str, workers: int) -> None:
    man = pd.read_csv(DS / "manifest.csv")
    man = man[man.set.isin(["sweep", "dev"])]  # drop the test rows on reading
    pics = picture_cases(man, tool)
    out = out_dir(tool)
    (out / "pictures").mkdir(parents=True, exist_ok=True)
    part = out / "findings.partial.jsonl"
    done = {json.loads(line)["case_id"] for line in part.read_text().splitlines()} if part.exists() else set()
    jobs = [(c, c in pics, tool) for c in case_ids(man) if c not in done]
    t0 = time.time()
    with Pool(workers) as pool, open(part, "a") as fh:
        for k, r in enumerate(pool.imap_unordered(_job, jobs), 1):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            print(f"[{tool}] {k}/{len(jobs)} {r['case_id']} {r['seconds']:.1f} s", flush=True)
    rows = [json.loads(line) for line in part.read_text().splitlines()]
    with open(out / "findings.jsonl", "w") as fh:
        for r in sorted(rows, key=lambda r: r["case_id"]):
            fh.write(json.dumps(r) + "\n")
    part.unlink()
    print(f"{tool}: {len(jobs)} volumes in {time.time() - t0:.0f} s ({len(done)} from an earlier run) → "
          f"{out / 'findings.jsonl'}")


def _label(cid: str) -> dict:
    return json.loads((DS / "labels" / f"{dev_only(cid)}.json").read_text())


def _findings(tool: str):
    for line in (out_dir(tool) / "findings.jsonl").read_text().splitlines():
        yield json.loads(line)


def table_noise() -> pd.DataFrame:
    """Noise findings joined with the labels (read here only)."""
    rows = []
    for r in _findings("noise"):
        f, lab = r["finding"], _label(r["case_id"])
        nz = lab["params"].get("noise")
        za = r["zero_air"] or {}
        rows.append(dict(case_id=r["case_id"], set=lab["set"], site=lab["site"], subject=lab["subject"],
                         artifacts="+".join(lab["artifacts"]) or "clean", noise=nz is not None,
                         noise_level=lab["severity"].get("noise", "none"),
                         sigma_true=nz["sigma_total"] if nz else lab["native_sigma"], present=f["present"],
                         level=level_of(f["value"]), value=f["value"], raw=f["evidence"].get("raw_sigma"),
                         fineness=f["evidence"].get("fineness"), tiles=f["evidence"].get("tiles"),
                         confidence=f["confidence"], seconds=r["seconds"], za_zeros=za.get("zeros"),
                         za_present=za.get("present"), za_level=level_of(za.get("value")), za_value=za.get("value")))
    return pd.DataFrame(rows)


def table_kspace(tool: str) -> pd.DataFrame:
    """Spike or ghosting findings joined with the labels (read here only)."""
    rows = []
    for r in _findings(tool):
        f, lab = r["finding"], _label(r["case_id"])
        p = lab["params"].get(tool)
        ev = f["evidence"]
        za = r["zero_air"] or {}
        row = dict(case_id=r["case_id"], set=lab["set"], site=lab["site"], subject=lab["subject"],
                   artifacts="+".join(lab["artifacts"]) or "clean", has=p is not None,
                   level_true=lab["severity"].get(tool, "none"), value_true=p[LABEL_RMS[tool]] if p else 0.0,
                   present=f["present"], value=f["value"], raw=ev.get("raw"), confidence=f["confidence"],
                   seconds=r["seconds"], za_zeros=za.get("zeros"), za_present=za.get("present"),
                   za_value=za.get("value"))
        if tool == "spike":
            ks = label_spike_k(p["spikes_positions"]) if p else []
            row |= dict(score=ev.get("best_z"), where_ok=spike_where_ok(ev.get("spikes", []), ks) if p else None,
                        n_found=len(ev.get("spikes", [])), n_echoes=len(ev.get("echoes", [])), n_label=len(ks),
                        top_k=json.dumps(ev["spikes"][0]["k"]) if ev.get("spikes") else None,
                        label_k=json.dumps(ks) if ks else None)
        else:
            n = p["num_ghosts"] if p else None
            score = (max(ev["peak"] / G.PEAK_LINE, ev["air_copy"]["prominence"] / G.AIR_LINE)
                     if ev.get("peak") is not None else None)
            row |= dict(score=score, where_ok=ghost_where_ok(ev.get("copy_mm"), n) if p else None, num_ghosts=n,
                        copy_mm=ev.get("copy_mm"), period_lines=ev.get("period_lines"), peak=ev.get("peak"),
                        air_prom=(ev.get("air_copy") or {}).get("prominence"), views="+".join(ev.get("views", [])))
        rows.append(row)
    return pd.DataFrame(rows)


def table_s3(tool: str) -> pd.DataFrame:
    """Bias or motion findings joined with the labels (read here only). score = the deciding number (evidence raw:
    the bias reading, the A–P leak), kept also when the tool cannot tell."""
    rows = []
    for r in _findings(tool):
        f, lab = r["finding"], _label(r["case_id"])
        p = lab["params"].get(tool)
        ev = f["evidence"]
        za = r["zero_air"] or {}
        row = dict(case_id=r["case_id"], set=lab["set"], site=lab["site"], subject=lab["subject"],
                   artifacts="+".join(lab["artifacts"]) or "clean", has=p is not None,
                   level_true=lab["severity"].get(tool, "none"), value_true=p[LABEL_RMS[tool]] if p else 0.0,
                   present=f["present"], value=f["value"], raw=ev.get("raw"), score=ev.get("raw"),
                   confidence=f["confidence"], why=f["why"], seconds=r["seconds"], za_zeros=za.get("zeros"),
                   za_present=za.get("present"), za_value=za.get("value"), za_raw=za.get("raw"))
        if tool == "bias":
            row |= dict(brain_ml=ev.get("brain_ml"), bright_side=ev.get("bright_side"), dark_side=ev.get("dark_side"))
        else:
            row |= dict(leak_lr=ev.get("leak_lr"), leak_si=ev.get("leak_si"), rhythm_peak=ev.get("rhythm_peak"),
                        noisy_air=ev.get("noisy_air"), noise_sigma=ev.get("noise_sigma"))
        rows.append(row)
    return pd.DataFrame(rows)


def calibrate(tool: str) -> None:
    if tool in ("bias", "motion"):
        d = table_s3(tool)
        d = d[(d.set == "sweep") & d.has & d.raw.notna() & d.present.notna()]
        r = d.raw / d.value_true
    elif tool == "noise":
        d = table_noise()
        d = d[(d.set == "sweep") & d.noise]
        r = d.raw / d.sigma_true
    else:
        d = table_kspace(tool)
        d = d[(d.set == "sweep") & d.has & d.raw.notna()]
        r = d.raw / d.value_true
    print(f"{tool}: SCALE = {float(np.median(r)):.3f}  (median raw/label over {len(d)} dev-sweep volumes; middle half "
          f"{r.quantile(0.25):.3f}–{r.quantile(0.75):.3f})")


def _zero_air(d: pd.DataFrame, level_cols: bool) -> dict:
    za = d[d.za_zeros.notna()]
    same = same_decision(za.present, za.za_present)
    if level_cols:
        same &= same_decision(za.level, za.za_level)
    if level_cols:  # noise: σ is read on every measured volume (Stage 1's definition)
        on, off = za[za.value.notna() & (za.value > 0)], za.iloc[:0]
    else:  # k-space tools: a value to read only where something was found (D98)
        on = za[za.present.eq(True) & za.value.notna() & (za.value > 0)]
        off = za[~za.present.eq(True) & za.value.notna()]
    return {"n": len(za), "same_finding": bool(same.all()), "changed": za[~same].case_id.tolist(),
            "max_move": float((on.za_value / on.value - 1).abs().max()) if len(on) else None,
            "max_move_nothing_found": float((off.za_value - off.value).abs().max()) if len(off) else None}


def _zero_air_s3(d: pd.DataFrame, tool: str) -> dict:
    """D104: no decision changes; bias: the value within 5 % where bias was found; motion: the leak within 0.002."""
    za = d[d.za_zeros.notna()]
    same = same_decision(za.present, za.za_present)
    rep = {"n": len(za), "same_finding": bool(same.all()), "changed": za[~same].case_id.tolist()}
    if tool == "bias":
        on = za[za.present.eq(True) & za.value.notna() & (za.value > 0)]
        rep["max_move"] = float((on.za_value / on.value - 1).abs().max()) if len(on) else None
    else:
        ok = za[za.raw.notna() & za.za_raw.notna()]
        rep["max_leak_move"] = float((ok.za_raw - ok.raw).abs().max()) if len(ok) else None
    return rep


def score(tool: str) -> None:
    out = out_dir(tool)
    if tool in ("bias", "motion"):
        d = table_s3(tool)
        rep = score_s3(d, BIAS_BANDS if tool == "bias" else DAMAGE_BANDS)
        rep["zero_air"] = _zero_air_s3(d, tool)
    elif tool == "noise":
        d = table_noise()
        rep = score_noise(d)
        rep["zero_air"] = _zero_air(d, level_cols=True)
    else:
        d = table_kspace(tool)
        rep = score_kspace(d)
        rep["zero_air"] = _zero_air(d, level_cols=False)
    rep["seconds"] = {"median": float(d.seconds.median()), "max": float(d.seconds.max())}
    rep["low_confidence"] = int((d.confidence == "low").sum())
    rep["could_not_measure"] = int(d.present.isna().sum())
    (out / "scores.json").write_text(json.dumps(rep, indent=1) + "\n")
    d.to_csv(out / "findings.csv", index=False)
    for s in ("sweep", "dev"):
        r = rep[s]
        if tool in ("bias", "motion"):
            print(f"{s}: {tool} {r['n_with']}, missed {r['missed']} {[c['case_id'] for c in r['missed_cases']]}, "
                  f"cannot tell {r['cannot_tell']}, false alarms {r['false_alarms']} "
                  f"{[c['case_id'] for c in r['false_alarm_cases']]}, AUC {r['auc']:.3f} (all {r['auc_all']:.3f}), "
                  f"median error {r['median_rel_err']:.1%}, ratio {r['median_ratio']:.3f}, within one "
                  f"{r['within_one']:.0%}, same level {r['same_level']:.0%}")
        elif tool == "noise":
            print(f"{s}: noise {r['n_noise']}, missed {r['missed']}, unmeasured {r['unmeasured']}, false alarms "
                  f"{r['false_alarms']}, AUC {r['auc']:.3f} (no spike {r['auc_no_spike']:.3f}), same level "
                  f"{r['same_level']:.0%}, within one {r['within_one']:.0%}, median error {r['median_rel_err']:.1%}, "
                  f"ratio {r['median_ratio']:.3f}")
        else:
            print(f"{s}: {tool} {r['n_with']}, missed {r['missed']} {[c['case_id'] for c in r['missed_cases']]}, "
                  f"unmeasured {r['unmeasured']}, false alarms {r['false_alarms']}, AUC {r['auc']:.3f}, "
                  f"median error {r['median_rel_err']:.1%}, ratio {r['median_ratio']:.3f}, within one "
                  f"{r['within_one']:.0%}, same level {r['same_level']:.0%}, where right {r['where_ok']:.0%}")
    print("zero-air:", rep["zero_air"], "seconds:", rep["seconds"], "low confidence:", rep["low_confidence"],
          "could not measure:", rep["could_not_measure"])


def _promise_job(args):
    cid, ks = args
    x = read_volume(DS / "inputs" / f"{cid}.nii.gz")
    a = _blind(noise_tool, x)
    b = _blind(noise_tool, despike(x, ks)) if ks else a
    return {"case_id": cid, "peaks_removed": len(ks), "noise_before": a.present, "sigma_before": a.value,
            "noise_after": b.present, "sigma_after": b.value}


def promise(workers: int) -> None:
    """The Stage 1 promise (D98): on every dev volume with a spike, remove the peaks the spike tool listed (spikes and
    echoes) and run the noise tool again. Labels pick the volumes and score; the repair uses the finding only."""
    jobs = []
    for r in _findings("spike"):
        if "spike" in _label(r["case_id"])["params"]:
            ev = r["finding"]["evidence"]
            jobs.append((r["case_id"], [p["k"] for p in ev.get("spikes", []) + ev.get("echoes", [])]))
    with Pool(workers) as pool:
        rows = pool.map(_promise_job, jobs, chunksize=1)
    df = pd.DataFrame(rows)
    df["noise_label"] = df.case_id.map(lambda c: "noise" in _label(c)["params"])
    df["set"] = df.case_id.map(lambda c: _label(c)["set"])
    df.to_csv(out_dir("spike") / "promise.csv", index=False)
    fa, nz = df[~df.noise_label], df[df.noise_label]
    print(f"{len(df)} spike volumes. Without noise ({len(fa)}): noise false alarms before "
          f"{int(fa.noise_before.eq(True).sum())}, after {int(fa.noise_after.eq(True).sum())} "
          f"{fa[fa.noise_after.eq(True)].case_id.tolist()}. With noise ({len(nz)}): found before "
          f"{int(nz.noise_before.eq(True).sum())}, after {int(nz.noise_after.eq(True).sum())}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["run", "calibrate", "score", "promise"])
    ap.add_argument("--tool", choices=list(TOOLS), default="noise")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    {"run": lambda: run(a.tool, a.workers), "calibrate": lambda: calibrate(a.tool), "score": lambda: score(a.tool),
     "promise": lambda: promise(a.workers)}[a.step]()


if __name__ == "__main__":
    main()
