"""D115 numbers for the Task 3 fixes: how much of its artifact a fix removes on its own single-artifact dev volumes,
no harm on the others, mixed volumes reported only.

`df` has one row per dev volume: case_id, artifacts ('+'-joined, 'clean' if none), has (the fix's artifact is in the
labels), changed, identical (the output's bytes equal the input's), own_in / own_out (brain RMS against the volume
without that artifact: the label replayed without it, the same noise draw; NaN without the artifact), after_present
(the Task 2 tool's yes/no on the output), seconds, zeros_kept (the input's exact zeros are still 0). Spike rows also
carry own_air_in, own_air_out (outside the head), n_true (spikes in the label) and n_repaired; ghosting rows
carry own_air_in, own_air_out (≥ 8 voxels outside the target's head). Optional: error (the fix
or its Task 2 tool reported an error instead of a reading).
A Task 2 answer of None afterwards (nothing measured) is never counted as a "no" (review I1).
The targets are the pilot's numbers rounded down (Stage 1 before-doc §10, D115; Stage 2 §11, D119; Stage 3 §14,
D124): the build must
reproduce the pilot, not beat it.
"""
import numpy as np
import pandas as pd

TARGETS = {"spike": {"removed_median": 0.95, "no_after": 29, "worse": 0, "seconds_max": 60.0},
           "bias": {"removed_median": 0.65, "no_after": 32, "worse": 0},
           "noise": {"removed_median": 0.40, "no_after": 32, "worse": 0, "seconds_max": 90.0},  # D119
           "ghosting": {"removed_median": 0.90, "no_after": 32, "worse": 0, "seconds_max": 30.0}}  # D124
ALONE_N = 32  # single-artifact dev volumes per artifact (blind-mri-v2's sweep)


def _group(d: pd.DataFrame, air: bool) -> dict:
    if d.empty:
        return {"n": 0}
    rem = 1 - d.own_out / d.own_in
    out = {"n": int(len(d)), "own_in_median": float(d.own_in.median()), "own_out_median": float(d.own_out.median()),
           "removed_median": float(rem.median()), "removed_min": float(rem.min()),
           "worse": int((d.own_out > d.own_in).sum()), "still_yes": int((d.after_present == True).sum()),  # noqa: E712
           "unchanged": int((~d.changed.astype(bool)).sum()),
           "seconds_median": float(d.seconds.median()), "seconds_max": float(d.seconds.max())}
    if air:
        out |= {"own_air_in_median": float(d.own_air_in.median()), "own_air_out_median": float(d.own_air_out.median())}
    return out


def _target(want, got, ok: bool) -> dict:
    return {"want": want, "got": got, "pass": bool(ok)}


def score_fix(df: pd.DataFrame, fix: str) -> dict:
    t = TARGETS[fix]
    has = df.has.astype(bool)
    alone = df[has & (df.artifacts == fix)]
    mixed = df[has & (df.artifacts != fix)]
    none = df[~has]
    still = df.changed.astype(bool)
    air = fix in ("spike", "ghosting")
    rep = {"alone": _group(alone, air), "mixed": _group(mixed, air),
           "no_harm": {"n": int(len(none)), "identical": int(none.identical.sum()),
                       "changed": int(none.changed.astype(bool).sum())}}
    miss = has & ~still
    if fix == "spike":
        miss = miss | (has & (df.n_true > df.n_repaired))
    rep["missed"] = sorted(df[miss].case_id)
    rep["false_alarms"] = sorted(df[~has & still].case_id)  # changed without the artifact: by name
    a = rep["alone"]
    no_after = int(alone.after_present.eq(False).sum())  # None (nothing measured) is not a "no"
    err = df.error.astype(bool) if "error" in df else pd.Series(False, index=df.index)
    # no harm: spike, noise and ghosting, the volumes their tool passes (the fix left them); bias, those without bias
    keep = df[~still] if fix in ("spike", "noise", "ghosting") else none
    tg = {"alone_n": _target(ALONE_N, a["n"], a["n"] == ALONE_N),
          "removed_median": _target(t["removed_median"], a.get("removed_median"),
                                    a["n"] and a["removed_median"] >= t["removed_median"]),
          "no_after": _target(t["no_after"], no_after, no_after >= t["no_after"]),
          "worse": _target(t["worse"], a.get("worse"), a["n"] and a["worse"] <= t["worse"]),
          "no_harm": _target(len(keep), int(keep.identical.sum()), bool(keep.identical.all())),
          "errors": _target(0, int(err.sum()), not err.any()),
          "unchanged_identical": _target("all", int(df[~still].identical.sum()), bool(df[~still].identical.all())),
          "zeros_kept": _target("all", int(df.zeros_kept.sum()), bool(df.zeros_kept.all()))}
    if "seconds_max" in t:
        sec = float(alone.seconds.max()) if len(alone) else float("nan")  # §10: on its own volumes
        tg["seconds_max"] = _target(t["seconds_max"], sec, np.isfinite(sec) and sec <= t["seconds_max"])
    rep["targets"] = tg
    return rep
