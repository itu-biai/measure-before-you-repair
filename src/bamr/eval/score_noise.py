"""D72 numbers for the noise tool: detection, false alarms, AUC, level agreement, error, per site.

`df` has one row per case: case_id, set, site, artifacts ('+'-joined, 'clean' if none), noise (bool), noise_level,
sigma_true (label: own ⊕ added for noise cases, own σ otherwise), present, value, raw. The tool reports σ and yes/no
only; the level is the band of its value, worked out here (D95).
"""
import numpy as np
import pandas as pd

ORDER = ["none", "mild", "moderate", "borderline", "above"]
NOISE_BANDS = ((0.030, 0.038), (0.038, 0.048), (0.048, 0.061))  # = configs/dataset_v2.yaml stage2.noise.sigma_total


def level_of(value, bands=NOISE_BANDS) -> str | None:
    """Band of a value: below the first band 'none', at or past the last band's top 'above'; lower edges inclusive."""
    if value is None or not np.isfinite(value):
        return None
    if value < bands[0][0]:
        return "none"
    for name, (lo, hi) in zip(ORDER[1:4], bands):
        if lo <= value < hi:
            return name
    return "above"


def dev_only(case_id: str) -> str:
    """The guard on every case the scorer touches: dev sweep (s*) or dev mixed (d*), never test (D72)."""
    if not (len(case_id) > 1 and case_id[0] in "sd" and case_id[1:].isdigit()):
        raise ValueError(f"not a dev case: {case_id!r}")
    return case_id


def case_ids(manifest: pd.DataFrame) -> list[str]:
    """Dev sweep + dev mixed only; the test set is never read in Task 2 (D72)."""
    return sorted(dev_only(c) for c in manifest[manifest.set.isin(["sweep", "dev"])].case_id)


def auc(pos, neg) -> float:
    """Mann–Whitney: chance a noise case reads higher than a case without noise (ties count ½)."""
    p, n = np.asarray(pos, float)[:, None], np.asarray(neg, float)[None, :]
    return float(((p > n) + 0.5 * (p == n)).mean()) if p.size and n.size else float("nan")


def same_decision(a: pd.Series, b: pd.Series) -> pd.Series:
    """Equal, or missing in both (could not measure twice is the same finding, D99 #5)."""
    a, b = a.astype(object), b.astype(object)
    return (a.isna() & b.isna()) | (a.notna() & b.notna() & (a == b))


def _levels(d: pd.DataFrame) -> tuple[float, float]:
    got = [ORDER.index(lv) if lv in ORDER else None for lv in d.level]
    want = [ORDER.index(lv) for lv in d.noise_level]
    same = np.mean([g == w for g, w in zip(got, want)])
    within = np.mean([g is not None and abs(g - w) <= 1 for g, w in zip(got, want)])
    return float(same), float(within)


def score_noise(df: pd.DataFrame) -> dict:
    out = {}
    df = df.assign(level=df.value.map(level_of))
    for s, d in df.groupby("set"):
        hit = d.present.eq(True)  # None (could not measure) counts as not found
        nz, other = d[d.noise], d[~d.noise]
        m = nz[nz.value.notna()]  # the error is over measured cases; the unmeasured are counted (D99 #4)
        same, within = _levels(nz) if len(nz) else (float("nan"), float("nan"))
        out[s] = {
            "n": len(d), "n_noise": int(len(nz)), "missed": int((~hit[d.noise]).sum()),
            "unmeasured": int(d.value.isna().sum()),
            "false_alarms": {a: int(hit[other.index][other.artifacts == a].sum()) for a in other.artifacts.unique()},
            "false_alarm_cases": other[hit[other.index]][["case_id", "artifacts"]].to_dict("records"),
            "auc": auc(nz.raw.dropna(), other.raw.dropna()),
            "auc_no_spike": auc(nz.raw.dropna(), other[~other.artifacts.str.contains("spike")].raw.dropna()),
            "same_level": same, "within_one": within,
            "median_rel_err": float(np.median(np.abs(m.value - m.sigma_true) / m.sigma_true)) if len(m) else None,
            "median_ratio": float(np.median(m.value / m.sigma_true)) if len(m) else None,
            "by_site": {site: {"same_level": _levels(g)[0],
                               "median_ratio": float(np.median(g.value / g.sigma_true)) if g.value.notna().any() else None}
                        for site, g in nz.groupby("site")},
        }
    return out
