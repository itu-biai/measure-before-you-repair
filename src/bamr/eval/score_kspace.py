"""D98 numbers for the k-space tools (spike, ghosting): detection, false alarms, AUC, value, level, where.

`df` has one row per case: case_id, set, site, artifacts ('+'-joined, 'clean' if none), has (the tool's artifact is
in the labels), level_true, value_true (spike_rms / ghosting_rms; 0 without the artifact), present, value, score (the
tool's deciding number, for the AUC), where_ok (bool, or None when not applicable). The level is the band of the value
on the shared damage scale, worked out here, never in the tools (D95).
"""
import numpy as np
import pandas as pd
import scipy.fft as sfft

from bamr.eval.score_noise import ORDER, auc, level_of
from bamr.tools.kspace import spectrum

DAMAGE_BANDS = ((0.0194, 0.0283), (0.0283, 0.0333), (0.0333, 0.0553))  # = configs/dataset_v2.yaml damage_band (D90)
T_SHAPE = (192, 240, 240)  # the simulator's TorchIO view (LR, SI, AP) of a blind-mri-v2 volume


def label_spike_k(positions, shape=T_SHAPE) -> list[tuple[int, int, int]]:
    """TorchIO spike positions in [0, 1)³ of the (LR, SI, AP) spectrum → k in the array's (LR, AP, SI) order."""
    s = np.array(shape)
    return [tuple(int(v) for v in (np.floor(np.array(p) * s).astype(int) - s // 2)[[0, 2, 1]]) for p in positions]


def _close(a, b, tol=1) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    return bool(np.abs(a - b).max() <= tol or np.abs(a + b).max() <= tol)


def spike_where_ok(spikes, label_ks) -> bool:
    """The top listed spike within 1 sample of a label spike (either sign: a spike shows at ±k)."""
    return bool(spikes) and any(_close(spikes[0]["k"], k) for k in label_ks)


def ghost_where_ok(copy_mm, n, fov_mm: float = 240.0) -> bool:
    """Copy spacing within 2 mm of FOV ÷ n (D98)."""
    return copy_mm is not None and n is not None and abs(copy_mm - fov_mm / n) <= 2


def _levels(d: pd.DataFrame) -> tuple[float, float]:
    got = [ORDER.index(lv) if lv in ORDER else None for lv in d.level]
    want = [ORDER.index("none" if lv == "faint" else lv) for lv in d.level_true]  # faint: a D65 share below mild
    same = np.mean([g == w for g, w in zip(got, want)])
    within = np.mean([g is not None and abs(g - w) <= 1 for g, w in zip(got, want)])
    return float(same), float(within)


def score_kspace(df: pd.DataFrame) -> dict:
    out = {}
    df = df.assign(level=df.value.map(lambda v: level_of(v, DAMAGE_BANDS)))
    for s, d in df.groupby("set"):
        hit = d.present.eq(True)  # None (could not measure) counts as not found
        w, o = d[d.has], d[~d.has]
        m = w[w.value.notna()]  # value error over measured cases; the unmeasured are counted (D99 #4)
        found = w[hit[w.index]]
        pos, neg = w.score.dropna(), o.score.dropna()
        same, within = _levels(w) if len(w) else (float("nan"), float("nan"))
        out[s] = {
            "n": len(d), "n_with": int(len(w)), "missed": int((~hit[w.index]).sum()),
            "unmeasured": int(d.value.isna().sum()),
            "false_alarms": {a: int(hit[o.index][o.artifacts == a].sum()) for a in o.artifacts.unique()},
            "false_alarm_cases": o[hit[o.index]][["case_id", "artifacts"]].to_dict("records"),
            "missed_cases": w[~hit[w.index]][["case_id", "artifacts", "level_true"]].to_dict("records"),
            "auc": auc(pos, neg), "auc_n": [int(len(pos)), int(len(neg))],
            "median_rel_err": float(np.median(np.abs(m.value - m.value_true) / m.value_true)) if len(m) else None,
            "median_ratio": float(np.median(m.value / m.value_true)) if len(m) else None,
            "same_level": same, "within_one": within,
            "where_ok": float(found.where_ok.astype(bool).mean()) if len(found) else None,
        }
    return out


def despike(x: np.ndarray, ks) -> np.ndarray:
    """A k-space clamp (ours; a check, not kao2000's method, D116), to test that the spike tool's evidence is enough to
    fix: each 3×3×3 at ±k scaled down to the power of the 7³ ring around it, phase kept; back to the image. Not a fix
    tool."""
    K = spectrum(x)
    c = np.array(K.shape) // 2
    for k in ks:
        for j in (c + np.asarray(k), c - np.asarray(k)):
            b3 = tuple(slice(v - 1, v + 2) for v in j)
            P7 = np.abs(K[tuple(slice(v - 3, v + 4) for v in j)]).astype(np.float64) ** 2
            ring = (P7.sum() - P7[2:5, 2:5, 2:5].sum()) / 316
            K[b3] *= np.minimum(np.sqrt(ring / np.maximum(np.abs(K[b3]).astype(np.float64) ** 2, 1e-30)), 1.0)
    return np.clip(np.real(sfft.ifftn(np.fft.ifftshift(K), workers=1)), 0, None).astype(np.float32)
