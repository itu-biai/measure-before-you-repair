"""HD-BET v2 brain masks on the raw grid, model loaded once (spec Stage 1 step 1, D03).

Run:  uv run python -m bamr.data.hdbet_batch [--limit N] [--device mps|cpu]
Writes <prep>/<id>/brain_mask_native.nii.gz in priority order; existing masks are skipped.
"""
import argparse
import os
import time
from pathlib import Path

MASK_NAME = "brain_mask_native.nii.gz"


def run_hdbet(pairs: list[tuple[Path, Path]], device: str = "mps", tta: bool = False, threads: int = 2) -> None:
    import numpy as np
    import torch
    from HD_BET.checkpoint_download import maybe_download_parameters
    from HD_BET.hd_bet_prediction import get_hdbet_predictor
    from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

    todo = [(i, o) for i, o in pairs if not Path(o).exists()]
    if not todo:
        return
    torch.set_num_threads(threads)
    maybe_download_parameters()
    predictor = get_hdbet_predictor(use_tta=tta, device=torch.device(device))
    io = SimpleITKIO()
    for k, (src, dst) in enumerate(todo, 1):
        t0 = time.time()
        dst = Path(dst)
        dst.parent.mkdir(parents=True, exist_ok=True)
        data, props = io.read_images([str(src)])
        seg = predictor.predict_single_npy_array(data, props, None, None, False)
        tmp = dst.with_name(".tmp_" + dst.name)
        io.write_seg(seg.astype(np.uint8), str(tmp), props)
        os.replace(tmp, dst)
        print(f"[hdbet] {k}/{len(todo)} {Path(src).name} {time.time() - t0:.1f}s", flush=True)


def main() -> None:
    from bamr.config import load_config
    from bamr.data.subjects import list_subjects, priority_order

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    cfg = load_config()
    s1 = cfg["stage1"]
    order = priority_order(list_subjects(cfg["paths"]["raw"]), cfg["seed"], s1["priority_quota"])[: args.limit]
    pairs = [(s.path, cfg["paths"]["prep"] / s.id / MASK_NAME) for s in order]
    run_hdbet(pairs, device=args.device or s1["hdbet"]["device"], tta=s1["hdbet"]["tta"])


if __name__ == "__main__":
    main()
