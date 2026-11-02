"""D104 numbers for the Stage 3 tools (bias, motion): detection, false alarms, "cannot tell", value, level, AUC.

`df` has one row per case: case_id, set, site, artifacts ('+'-joined, 'clean' if none), has (the tool's artifact is
in the labels), level_true, value_true (target_maxmin / motion_rms; 0 without the artifact), present (True, False, or
None = cannot tell), value, score (the deciding number: the bias reading, the A–P leak; the motion tool reports it
even when it cannot tell). "Cannot tell" is counted apart, never as a miss or a false alarm (D104, D105). The level is
the band of the value, worked out here, never in the tools (D95).
"""
import numpy as np
import pandas as pd

from bamr.eval.score_noise import ORDER, auc

BIAS_BANDS = ((0.15, 0.25), (0.35, 0.45), (0.55, 0.70))  # = configs/dataset_v2.yaml stage2.bias.target_maxmin


def level_gapped(value, bands) -> str | None:
    """Band of a value; between two bands the nearer one, split at the gap's middle (R6); below the first band
    'none', at or past the last band's top 'above'. For touching bands this is score_noise.level_of."""
    if value is None or not np.isfinite(value):
        return None
    if value < bands[0][0]:
        return "none"
    if value >= bands[-1][1]:
        return "above"
    tops = [(hi + lo2) / 2 for (_, hi), (lo2, _) in zip(bands[:-1], bands[1:])] + [bands[-1][1]]
    return next(name for name, top in zip(ORDER[1:4], tops) if value < top)


def _levels(d: pd.DataFrame) -> tuple[float, float]:
    got = [ORDER.index(lv) if lv in ORDER else None for lv in d.level]
    want = [ORDER.index("none" if lv == "faint" else lv) for lv in d.level_true]  # faint: a D65 share below mild
    same = np.mean([g == w for g, w in zip(got, want)])
    within = np.mean([g is not None and abs(g - w) <= 1 for g, w in zip(got, want)])
    return float(same), float(within)


def score_s3(df: pd.DataFrame, bands) -> dict:
    out = {}
    df = df.assign(level=df.value.map(lambda v: level_gapped(v, bands)))
    for s, d in df.groupby("set"):
        yes, no, unsure = d.present.eq(True), d.present.eq(False), d.present.isna()
        w, o = d[d.has], d[~d.has]
        m = w[w.value.notna()]  # value error over the cases with a value
        told = d[~unsure]
        same, within = _levels(m) if len(m) else (float("nan"), float("nan"))
        pos, neg = told[told.has].score.dropna(), told[~told.has].score.dropna()
        pos_all, neg_all = w.score.dropna(), o.score.dropna()
        out[s] = {
            "n": len(d), "n_with": int(len(w)), "missed": int(no[w.index].sum()),
            "cannot_tell": {"with": int(unsure[w.index].sum()), "without": int(unsure[o.index].sum())},
            "false_alarms": {a: int(yes[o.index][o.artifacts == a].sum()) for a in o.artifacts.unique()},
            "false_alarm_cases": o[yes[o.index]][["case_id", "artifacts"]].to_dict("records"),
            "missed_cases": w[no[w.index]][["case_id", "artifacts", "level_true"]].to_dict("records"),
            "cannot_tell_cases": d[unsure][["case_id", "artifacts"]].to_dict("records"),
            "auc": auc(pos, neg), "auc_n": [int(len(pos)), int(len(neg))],
            "auc_all": auc(pos_all, neg_all), "auc_all_n": [int(len(pos_all)), int(len(neg_all))],
            "median_rel_err": float(np.median(np.abs(m.value - m.value_true) / m.value_true)) if len(m) else None,
            "median_ratio": float(np.median(m.value / m.value_true)) if len(m) else None,
            "same_level": same, "within_one": within,
        }
    return out
