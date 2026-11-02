"""blind-mri-v2 CLI (D80–D82): split, reference sheet, build + audit.

  uv run python scripts/build_v2.py split    # → ../dataset/blind-mri-v2-split.json
  uv run python scripts/build_v2.py sheet    # → ../dataset/blind-mri-v2-sheet/*.png, three views of every subject (D57)
  uv run python scripts/build_v2.py build    # → ../dataset/blind-mri-v2/ + ../dataset/blind-mri-v2-audit.json
  uv run python scripts/build_v2.py audit    # audit an existing build again
Reads ../dataset/IXI-T1-prep (read-only); never reads or writes blind-mri-v1. The build refuses a dirty git tree (labels
record the commit) unless --allow-dirty is given, and refuses while stage3.frozen is true. Restartable.
"""
import argparse
import json
import os
import subprocess
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "2")

import numpy as np  # noqa: E402

from bamr.config import REPO, load_config  # noqa: E402
from bamr.data.build3d import all_candidates, audit_v2, build_v2, load_boxed, splits_doc_v2  # noqa: E402
from bamr.data.plan_v2 import plan_cases_v2, split_v2  # noqa: E402

CONFIG = REPO / "configs" / "dataset_v2.yaml"


def git_code() -> str:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True)
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True, check=True)
    return head.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")


def plan(cfg):
    v1 = json.loads(cfg["paths"]["v1_split"].read_text())
    calib = {s["id"] for s in json.loads((cfg["paths"]["calib"] / "subjects.json").read_text())}
    split = split_v2(all_candidates(cfg), v1, calib, cfg["stage3"])
    return v1, calib, split, plan_cases_v2(split, cfg["stage3"], cfg["plan_seed"], cfg["stage2"])


def sheet(cfg, split):
    """Mid-sagittal, axial and coronal view of every test, dev and reserve subject's boxed reference, 24 per page."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = cfg["paths"]["out"].with_name(cfg["paths"]["out"].name + "-sheet")
    out.mkdir(exist_ok=True)
    subs = [(s, m) for s in ("test", "dev", "reserve") for m in split[s]]
    per = 24
    for page in range(0, len(subs), per):
        chunk = subs[page:page + per]
        fig, axs = plt.subplots(6, 12, figsize=(24, 13.5))
        for ax in axs.ravel():
            ax.axis("off")
        for j, (s, m) in enumerate(chunk):
            ref, brain, _ = load_boxed(cfg["paths"]["prep"], m.id)
            c = [int(np.mean(np.flatnonzero(brain.any(axis=tuple(a for a in range(3) if a != ax))))) for ax in range(3)]
            views = [ref[c[0]].T[::-1], ref[:, :, c[2]].T[::-1, ::-1], ref[:, c[1], :].T[::-1, ::-1]]
            r, k = divmod(j, 4)
            for v, view in enumerate(views):
                ax = axs[r, 3 * k + v]
                ax.imshow(view, cmap="gray", vmin=0, vmax=1.4)
                if v == 0:
                    ax.set_title(f"{m.id} {m.site} · {s}", fontsize=8, loc="left")
        fig.subplots_adjust(0.005, 0.005, 0.995, 0.97, 0.02, 0.12)
        fig.savefig(out / f"sheet_{page // per + 1:02d}.png", dpi=55)
        plt.close(fig)
        print("page", page // per + 1, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["split", "sheet", "build", "audit"])
    ap.add_argument("--allow-dirty", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    cfg = load_config(CONFIG)
    out = cfg["paths"]["out"]
    v1, calib, split, cases = plan(cfg)
    if args.step == "split":
        doc = splits_doc_v2(cfg, split)
        p = out.with_name(out.name + "-split.json")
        p.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n")
        print({s: len(split[s]) for s in ("test", "dev", "reserve", "sweep")}, "cases", len(cases),
              "excluded", len(split["excluded"]), "→", p)
        return
    if args.step == "sheet":
        sheet(cfg, split)
        return
    if args.step == "build":
        if cfg["stage3"].get("frozen"):
            sys.exit(f"{out.name} is frozen: verify it with `shasum -a 256 -c {out.name}.sha256` in {out.parent}")
        code = git_code()
        if code.endswith("-dirty") and not args.allow_dirty:
            sys.exit(f"git tree is dirty ({code}); commit first or pass --allow-dirty")
        t0 = time.time()
        man = build_v2(cfg, split, cases, out, code, args.workers, log=lambda s: print(s, flush=True))
        print(f"built {len(man)} cases in {time.time() - t0:.0f} s", flush=True)
    t1 = time.time()
    rep = audit_v2(out, splits_doc_v2(cfg, split), v1, calib, args.workers)
    rep["audit_s"] = round(time.time() - t1, 1)
    out.with_name(out.name + "-audit.json").write_text(json.dumps(rep, indent=1) + "\n")
    flags = {k: v for k, v in rep.items() if isinstance(v, bool)}
    print("audit:", flags, rep["n_by_set"], flush=True)
    if not all(flags.values()):
        sys.exit("audit failed")


if __name__ == "__main__":
    main()
