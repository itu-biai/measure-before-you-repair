"""Stage 1 CLI: HD-BET (one MPS process) + CPU pool for N4 → σ → 1 mm RAS → WM = 1 → QC → manifest.

  uv run python scripts/prep_ixi.py [--limit N] [--workers K] [--no-hdbet] [--retry-errors]

Subjects run in the fixed priority order (60 priority first). Restartable: done = <prep>/<id>/qc.json.
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS", "ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"):
    os.environ[_v] = "1"  # 1 thread per pool worker (D18); inherited by spawned workers

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait  # noqa: E402

from bamr.config import load_config  # noqa: E402
from bamr.data.hdbet_batch import MASK_NAME  # noqa: E402
from bamr.data.subjects import list_subjects, priority_order  # noqa: E402
from bamr.data.worker import build_manifest, process_subject  # noqa: E402


def _init_worker() -> None:
    import SimpleITK as sitk
    import torch

    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)
    torch.set_num_threads(1)


def pool_size(cap: int) -> int:
    """k = min(cap, ncpu − ceil(load1) − 1), at least 1 (D18)."""
    return max(1, min(cap, os.cpu_count() - math.ceil(os.getloadavg()[0]) - 1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-hdbet", action="store_true", help="masks come from a separately started hdbet_batch")
    ap.add_argument("--retry-errors", action="store_true")
    ap.add_argument("--mask-timeout", type=float, default=900, help="give up after this long without a new mask")
    args = ap.parse_args()

    cfg = load_config()
    s1, prep = cfg["stage1"], cfg["paths"]["prep"]
    full = priority_order(list_subjects(cfg["paths"]["raw"]), cfg["seed"], s1["priority_quota"])
    order = full[: args.limit]
    n_prio = sum(s1["priority_quota"].values())
    (prep / "logs").mkdir(parents=True, exist_ok=True)

    if args.retry_errors:
        for s in order:
            q = prep / s.id / "qc.json"
            if q.exists() and json.loads(q.read_text()).get("error"):
                q.unlink()

    hd = None
    if not args.no_hdbet:
        cmd = [sys.executable, "-m", "bamr.data.hdbet_batch", "--device", args.device or s1["hdbet"]["device"]]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        hd = subprocess.Popen(cmd, stdout=open(prep / "logs" / "hdbet.log", "a"), stderr=subprocess.STDOUT,
                              env={**os.environ, "OMP_NUM_THREADS": "2"})

    pending = [s for s in order if not (prep / s.id / "qc.json").exists()]
    k = args.workers or pool_size(s1["pool_max_workers"])
    print(f"[prep] {len(pending)}/{len(order)} to do, {k} workers, load {os.getloadavg()[0]:.1f}", flush=True)
    t0, last_mask, n_done = time.time(), time.time(), 0
    with ProcessPoolExecutor(k, initializer=_init_worker) as ex:
        futs = {}
        while pending or futs:
            ready = [s for s in pending if (prep / s.id / MASK_NAME).exists()]
            if ready:
                last_mask = time.time()
            for s in ready[: max(0, 2 * k - len(futs))]:
                futs[ex.submit(process_subject, s, prep / s.id / MASK_NAME, prep / s.id, cfg)] = s
                pending.remove(s)
            if not futs:
                hd_dead = hd is not None and hd.poll() is not None
                if hd_dead or time.time() - last_mask > args.mask_timeout:
                    print(f"[prep] stopping: {len(pending)} subjects have no mask", flush=True)
                    break
                time.sleep(5)
                continue
            done, _ = wait(list(futs), timeout=5, return_when=FIRST_COMPLETED)
            for f in done:
                s = futs.pop(f)
                r = f.result()
                n_done += 1
                print(f"[prep] {n_done} {s.id} {s.site} {r.get('t_total', 0):.0f}s all_ok={r.get('all_ok')} "
                      f"err={r.get('error')} elapsed={(time.time() - t0) / 60:.1f}min", flush=True)
    df = build_manifest(prep, [s.id for s in full], n_prio)
    print(f"[prep] manifest: {len(df)} rows, all_ok {int(df['all_ok'].sum()) if len(df) else 0}", flush=True)
    if hd is not None:
        hd.wait()


if __name__ == "__main__":
    main()
