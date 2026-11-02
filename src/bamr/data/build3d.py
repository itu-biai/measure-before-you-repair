"""blind-mri-v2 build: box each subject's volume, corrupt it in 3-D, write the blind folder, audit it (D80, D81).

Restartable: every finished case leaves a row file in `<out>.work/rows/`; a rerun skips cases whose row was written by
the same code commit. Files go to `<out>.partial/` and the folder is swapped in only when every case is done.
Files are written with one fixed modification time and gzip without a time stamp (D56, D81).
"""
import gzip
import json
import os
import re
import shutil
import time
from collections import Counter
from multiprocessing import get_context
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from bamr.data import artifacts3d as B
from bamr.data.assemble import Member, case_seed
from bamr.data.blindio import FILE_TIME, nifti_bytes3, read_volume, write_file
from bamr.data.metrics import masked_psnr, masked_ssim
from bamr.data.subjects import list_subjects, priority_order

INPUT_NAME = re.compile(r"^[tds]\d{4}\.nii\.gz$")


def all_candidates(cfg: dict) -> list[Member]:
    """Every IXI subject in the seeded priority order (D04) with its Stage 1 QC result and own noise σ."""
    order = priority_order(list_subjects(cfg["paths"]["raw"]), cfg["seed"], cfg["stage1"]["priority_quota"])
    out = []
    for rank, s in enumerate(order):
        p = cfg["paths"]["prep"] / s.id / "qc.json"
        q = json.loads(p.read_text()) if p.exists() else {"all_ok": False}
        sig = float(q.get("sigma_lstd_1mm_measured", np.nan))
        out.append(Member(s.id, s.site, rank, sig, bool(q["all_ok"]) and np.isfinite(sig)))
    return out


def load_boxed(prep: Path, sid: str):
    d = Path(prep) / sid
    t1 = nib.load(d / "t1_1mm.nii.gz").get_fdata(dtype=np.float32)
    mask = np.asarray(nib.load(d / "brain_mask_1mm.nii.gz").dataobj) > 0
    o = B.box_origin(mask)
    ref, brain = B.cut_box(t1, o), B.cut_box(mask, o)
    if brain.sum() != mask.sum():
        raise ValueError(f"{sid}: the brain does not fit in the box")
    return ref, brain, {"box_origin": list(o), "shape_1mm": list(t1.shape)}


def render_case(case: dict, ref, brain, cfg2: dict, seed: int, max_attempts: int):
    """Stage 2 in 3-D on one case; a failed sampling retries with the next seed (D58)."""
    for attempt in range(max_attempts):
        try:
            out, label = B.corrupt3(ref, brain, case["recipe"], case_seed(seed, case["set"], case["k"], attempt),
                                    cfg2, case["native_sigma"])
            return out, label, attempt
        except RuntimeError:
            continue
    raise RuntimeError(f"{case['case_id']}: no valid draw in {max_attempts} attempts")


def _subject_job(args):
    """All cases of one subject: load and box once, write each case's four files, then its row."""
    import torch
    torch.set_num_threads(2)
    sid, cases, cfg2, seed, max_attempts, prep, tmp, rows_dir, code, config_name = args
    ref, brain, meta = load_boxed(prep, sid)
    ref_b, brain_b = nifti_bytes3(ref, np.float32), nifti_bytes3(brain, np.uint8)
    done = []
    for c in cases:
        cid = c["case_id"]
        row_p = rows_dir / f"{cid}.json"
        if row_p.exists() and json.loads(row_p.read_text()).get("code") == code:
            continue
        t0 = time.time()
        img, lab, attempt = render_case(c, ref, brain, cfg2, seed, max_attempts)
        label = {"case_id": cid, "set": c["set"], "subject": sid, "site": c["site"], "volume": meta,
                 "native_sigma": c["native_sigma"], **lab, "attempt": attempt, "config": config_name, "code": code}
        write_file(tmp / "inputs" / f"{cid}.nii.gz", nifti_bytes3(img, np.float32))
        write_file(tmp / "refs" / cid / "reference.nii.gz", ref_b)
        write_file(tmp / "refs" / cid / "brain_mask.nii.gz", brain_b)
        write_file(tmp / "labels" / f"{cid}.json", (json.dumps(label, indent=1) + "\n").encode())
        names = lab["artifacts"]
        b = lab.get("budget", {})
        row = {"case_id": cid, "set": c["set"], "subject": sid, "site": c["site"], "card": c["card"],
               "n_artifacts": len(names), "artifacts": "+".join(names),
               "severity": ";".join(f"{n}:{lab['severity'][n]}" for n in names), "stress": len(names) >= 4,
               "attempt": attempt, "budget_level": b.get("level", ""), "budget_total": b.get("total", np.nan),
               "native_sigma": c["native_sigma"],
               "psnr": masked_psnr(ref, img, brain), "ssim": masked_ssim(ref, img, brain),
               "damage": float(np.sqrt(np.mean((img.astype(np.float64) - ref)[brain] ** 2))),
               "seconds": round(time.time() - t0, 1), "code": code}
        tmp_p = row_p.with_suffix(".tmp")
        tmp_p.write_text(json.dumps(row))
        os.replace(tmp_p, row_p)
        done.append(cid)
    return sid, done


def build_v2(cfg: dict, split: dict, cases: list[dict], out: Path, code: str, workers: int = 4,
             config_name: str = "dataset_v2.yaml", log=print) -> pd.DataFrame:
    out = Path(out)
    tmp, work = out.with_name(out.name + ".partial"), out.with_name(out.name + ".work")
    rows_dir = work / "rows"
    rows_dir.mkdir(parents=True, exist_ok=True)
    by_sub: dict[str, list] = {}
    for c in cases:
        by_sub.setdefault(c["subject"], []).append(c)
    jobs = [(sid, cs, cfg["stage2"], cfg["plan_seed"], cfg["stage3"]["max_attempts"], cfg["paths"]["prep"], tmp,
             rows_dir, code, config_name) for sid, cs in sorted(by_sub.items(), key=lambda kv: -len(kv[1]))]
    t0, n_done = time.time(), 0
    with get_context("spawn").Pool(workers) as pool:
        for sid, done in pool.imap_unordered(_subject_job, jobs):
            n_done += 1
            log(f"{n_done}/{len(jobs)} subjects · {sid} +{len(done)} cases · {time.time() - t0:.0f} s")
    rows = [json.loads((rows_dir / f"{c['case_id']}.json").read_text()) for c in cases]
    man = pd.DataFrame(rows).drop(columns=["seconds", "code"]).sort_values("case_id", ignore_index=True)
    man["card"] = man["card"].astype("Int64")
    write_file(tmp / "manifest.csv", man.to_csv(index=False, float_format="%.6g").encode())
    write_file(tmp / "splits.json", (json.dumps(splits_doc_v2(cfg, split), indent=1, ensure_ascii=False) + "\n").encode())
    for d in sorted((p for p in tmp.rglob("*") if p.is_dir()), reverse=True):
        os.utime(d, (FILE_TIME, FILE_TIME))
    if out.exists():
        shutil.rmtree(out)
    tmp.rename(out)
    shutil.rmtree(work)
    return man


def splits_doc_v2(cfg: dict, split: dict) -> dict:
    row = lambda m: {"id": m.id, "site": m.site, "rank": m.rank, "native_sigma": round(m.native_sigma, 5)}  # noqa: E731
    return {"seed": cfg["seed"], "plan_seed": cfg["plan_seed"], "rule": "D80, D51, D57",
            **{s: [row(m) for m in split[s]] for s in ("test", "dev", "reserve", "sweep")},
            "excluded": split["excluded"]}


# ---------------------------------------------------------------- audit
def _replay_job(args):
    import torch
    torch.set_num_threads(2)
    out, cid = args
    lab = json.loads((out / "labels" / f"{cid}.json").read_text())
    x = read_volume(out / "inputs" / f"{cid}.nii.gz")
    ref = read_volume(out / "refs" / cid / "reference.nii.gz")
    ok = bool(np.array_equal(B.replay3(ref, lab), x))
    clean = bool(np.array_equal(x, ref)) if not lab["artifacts"] else None
    return cid, ok, clean


def audit_v2(out: Path, split_doc: dict, v1_split: dict, calib_ids: set[str], workers: int = 4) -> dict:
    """Leak audit (D56, D81), replay check, and the one-subject-per-case rules of D80. Booleans must all be True."""
    out = Path(out)
    inp = sorted((out / "inputs").iterdir())
    gz = list(out.rglob("*.nii.gz"))
    labels = {p.stem: json.loads(p.read_text()) for p in sorted((out / "labels").glob("*.json"))}
    heads = set()
    for p in inp:
        with gzip.open(p, "rb") as f:
            heads.add(f.read(352))
    with get_context("spawn").Pool(workers) as pool:
        res = pool.map(_replay_job, [(out, cid) for cid in labels])
    balance: dict = {}
    subjects: dict = {}
    for lab in labels.values():
        size = len(lab["artifacts"])
        subjects.setdefault(lab["set"], []).append(lab["subject"])
        for a, lv in lab["severity"].items():
            balance.setdefault(lab["set"], {}).setdefault(str(size), {}).setdefault(a, Counter())[lv] += 1
    v1_dev = {r["id"] for r in v1_split["dev"]}
    v1_test = {r["id"] for r in v1_split["test"]}
    used = {s for v in subjects.values() for s in v}
    return {
        "n_by_set": dict(Counter(p.name[0] for p in inp)),
        "only_inputs": all(p.is_file() and INPUT_NAME.match(p.name) for p in inp),
        "headers_identical": len(heads) == 1,
        "shape_ok": all(read_volume(inp[i]).shape == B.BOX for i in (0, len(inp) // 2, len(inp) - 1)),
        "gzip_clean": all(b[4:8] == b"\0\0\0\0" and not b[3] & 0x08 for b in (p.open("rb").read(10) for p in gz)),
        "mtimes_fixed": all(p.stat().st_mtime == FILE_TIME for p in inp),
        "labels_match_inputs": set(labels) == {p.name[:-7] for p in inp},
        "replay_ok": all(r[1] for r in res),
        "clean_equal": all(r[2] for r in res if r[2] is not None),
        "one_subject_per_test_case": len(subjects["test"]) == len(set(subjects["test"])),
        "one_subject_per_dev_case": len(subjects["dev"]) == len(set(subjects["dev"])),
        "test_dev_disjoint": not (set(subjects["test"]) & (set(subjects["dev"]) | set(subjects["sweep"]))),
        "v1_dev_not_in_test": not (set(subjects["test"]) & v1_dev),
        "v1_test_in_test": v1_test <= set(subjects["test"]),
        "no_calibration_subject": not (used & calib_ids),
        "splits_match": {m["id"] for m in split_doc["test"]} == set(subjects["test"]),
        "replay_failed": [r[0] for r in res if not r[1]],
        "balance": {s: {n: {a: dict(c) for a, c in d.items()} for n, d in v.items()} for s, v in balance.items()},
    }
