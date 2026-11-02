"""Task 4 scoring rules (D132, D140): what a run of the chain, the router or the agent did to one volume.

Headline (D132): the error removed against the reference with its own motion replayed (motion has no repair, so the
best possible output still holds it; the same measure every fix was scored with, D115), RMS over the reference's brain,
WM = 1; removed = 1 − out / in, and worse when the output is further from that target than the input. Detection: each
artifact's yes/no in the report against the label. Rescan: right when it says yes exactly when the label's motion
damage (motion_rms) is ≥ RESCAN (D123); per artifact also sensitivity and specificity. Image quality (D158): PSNR,
SSIM and LPIPS of input and output against the clean reference (bamr.eval.quality; the scorer adds them to each row),
summarised over the volumes holding an artifact. A run with no report (fused, API error) answers no to everything. The
report check (D147: a repairs, b numbers, c findings; None without a report), the process counts and the cost
(OpenRouter's /generation record when there is one) come from result.json.
"""
import numpy as np

ARTIFACTS = ("noise", "spike", "ghosting", "bias", "motion")
RESCAN = 0.033
QUALITY = ("psnr", "ssim", "lpips")  # D158, bamr.eval.quality


def motion_target(ref: np.ndarray, label: dict, replay) -> np.ndarray:
    """The reference with only its motion replayed (replay = bamr.data.artifacts3d.replay3)."""
    if "motion" not in label["artifacts"]:
        return ref
    return replay(ref, dict(label, artifacts=["motion"]))


def _rms(a, b, m) -> float:
    d = (np.asarray(a, np.float64) - np.asarray(b, np.float64))[m]
    return float(np.sqrt(np.mean(d ** 2))) if d.size else 0.0


def case_row(x_in, y_out, target, brain, label: dict, result: dict) -> dict:
    e_in, e_out = _rms(x_in, target, brain), _rms(y_out, target, brain)
    rep = result.get("report")
    said = (rep or {}).get("artifacts") or {}
    right = {a: (said.get(a) == "yes") == (a in label["artifacts"]) for a in ARTIFACTS}
    m = (label.get("params") or {}).get("motion") or {}
    rescan_true = "motion" in label["artifacts"] and float(m.get("motion_rms", 0.0)) >= RESCAN
    rescan_said = (rep or {}).get("rescan") == "yes"
    u = result.get("usage") or {}
    gen = result.get("generation") or {}
    return {"system": result.get("system"), "model": result.get("model"), "artifacts": "+".join(label["artifacts"]),
            "n_artifacts": len(label["artifacts"]), "error_in": e_in, "error_out": e_out,
            "removed": 1 - e_out / e_in if e_in > 0 else None, "worse": bool(e_out > e_in * (1 + 1e-6) + 1e-9),
            "version": result.get("version"), "repairs": [r["repair"] for r in result.get("lineage", [])[1:]],
            "has_report": rep is not None, "right": right, "n_right": sum(right.values()),
            "said": {a: said.get(a) for a in ARTIFACTS}, "rescan_true": rescan_true, "rescan_said": rescan_said,
            "rescan_right": rescan_said == rescan_true, "motion_rms": float(m.get("motion_rms", 0.0)),
            "turns": result.get("turns", 0), "calls": result.get("calls", 0), "bad_calls": result.get("bad_calls", 0),
            "fused": bool(result.get("fused")), "error": result.get("error"), "seconds": result.get("seconds"),
            "prompt_tokens": u.get("prompt_tokens", 0), "completion_tokens": u.get("completion_tokens", 0),
            "cached_tokens": u.get("cached_tokens", 0), "reasoning_tokens": u.get("reasoning_tokens", 0),
            "cost": gen.get("cost", u.get("cost", 0.0)), "unparsed": result.get("unparsed"),
            **{f"check_{k}": (result.get("check") or {}).get("pass", {}).get(k) for k in "abc"}}


def detection(rows: list[dict]) -> dict:
    """Per artifact: said yes vs the label (D132). Sensitivity None without a volume holding it, specificity None
    without one free of it."""
    out = {}
    for a in ARTIFACTS:
        has = [a in r["artifacts"].split("+") for r in rows]
        yes = [r["said"][a] == "yes" for r in rows]
        tp, fn = sum(h and y for h, y in zip(has, yes)), sum(h and not y for h, y in zip(has, yes))
        fp, tn = sum(y and not h for h, y in zip(has, yes)), sum(not h and not y for h, y in zip(has, yes))
        out[a] = {"tp": tp, "fn": fn, "fp": fp, "tn": tn, "sensitivity": tp / (tp + fn) if tp + fn else None,
                  "specificity": tn / (tn + fp) if tn + fp else None}
    return out


def image_quality(rows: list[dict]) -> dict | None:
    """PSNR, SSIM, LPIPS of input and output against the clean reference (D158), over the volumes holding at least
    one artifact: a clean volume has nothing to restore (its PSNR is infinite when left alone); safety there is
    "made worse"."""
    q = [r for r in rows if r["n_artifacts"] and "psnr_out" in r]
    if not q:
        return None
    stat = lambda v: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else None,  # noqa: E731
                      "median": float(np.median(v))}
    return {"n": len(q), **{f"{k}_{s}": stat([r[f"{k}_{s}"] for r in q])
                            for k in QUALITY for s in ("in", "out")}}


def summary(rows: list[dict]) -> dict:
    rem = [r["removed"] for r in rows if r["removed"] is not None]
    tot = lambda k: sum(r[k] or 0 for r in rows)  # noqa: E731
    return {"n": len(rows), "removed_median": float(np.median(rem)) if rem else None,
            "removed_mean": float(np.mean(rem)) if rem else None, "worse": sum(r["worse"] for r in rows),
            "artifacts_right": tot("n_right"), "artifacts_total": len(ARTIFACTS) * len(rows),
            "per_artifact_right": {a: sum(r["right"][a] for r in rows) for a in ARTIFACTS},
            "detection": detection(rows), "quality": image_quality(rows),
            "rescan_right": sum(r["rescan_right"] for r in rows), "no_report": sum(not r["has_report"] for r in rows),
            "turns": tot("turns"), "calls": tot("calls"), "bad_calls": tot("bad_calls"),
            "fuses": sum(r["fused"] for r in rows), "errors": sum(r["error"] is not None for r in rows),
            "seconds_median": float(np.median([r["seconds"] for r in rows])) if rows else None,
            "seconds_total": tot("seconds"), "prompt_tokens": tot("prompt_tokens"),
            "completion_tokens": tot("completion_tokens"), "cached_tokens": tot("cached_tokens"),
            "reasoning_tokens": tot("reasoning_tokens"), "cost": tot("cost"),
            "checked": sum(r["check_a"] is not None for r in rows),
            "check_pass": {**{k: sum(r[f"check_{k}"] is True for r in rows) for k in "abc"},
                           "all": sum(all(r[f"check_{k}"] is True for k in "abc") for r in rows)}}
