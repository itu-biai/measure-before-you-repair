"""The bias fix (D114 A): the smooth shading N4 finds, divided out, only when the bias tool would call it a bias.

The bias tool's own steps (D102): the brain by HD-BET on the input (isensee2019), N4's B-spline field inside it
(tustison2010; sled1998's N3 made robust), and its reading e^(max − min of the log field over the brain) − 1. Under the
bias tool's line 0.083 the volume comes back as given. At or above it, the input is divided by the field, centred so
its log averages 0 over the brain, which keeps the brain's brightness (WM ≈ 1): correction by division (vovk2007).
N4 is then read again on the output. What N4 leaves is as smooth as anatomy, so part of a real field stays (D102: N4
sees about 0.6 of it). Exact-zero voxels are left out of N4's mask and stay 0 under the division (D69).
"""
import numpy as np

from bamr.fixes.report import FixReport, head_change, unchanged
from bamr.tools.bias import LINE, MIN_BRAIN_ML, MIN_SIDE, SCALE, WORDS, brain_mask, n4_logfield, sides
from bamr.tools.finding import PROTOCOL_V2, Protocol, by_line, check_volume
from bamr.tools.regions import head_mask

TOOL, VERSION, ARTIFACT = "fix_bias", 1, "bias"


def _reading(logf: np.ndarray, brain: np.ndarray) -> float:
    v = logf[brain]
    return round(float(np.exp(v.max() - v.min()) - 1), 6)


def fix_bias(x: np.ndarray, protocol: Protocol = PROTOCOL_V2) -> tuple[np.ndarray, FixReport]:
    try:
        a = np.asarray(x)
        why = check_volume(a, protocol)
        if not why and min(a.shape) < MIN_SIDE:
            why = f"volume {a.shape} is too small for N4's field (need ≥ {MIN_SIDE} samples per side)"
        if why:
            return unchanged(x, TOOL, VERSION, ARTIFACT, why)
        return _fix(x, a.astype(np.float32, copy=False), protocol)
    except Exception as e:  # no crash inside an agent loop (D68): the error becomes the reason
        return unchanged(x, TOOL, VERSION, ARTIFACT, f"fix error: {type(e).__name__}: {e}")


def _fix(x, a: np.ndarray, protocol: Protocol) -> tuple[np.ndarray, FixReport]:
    brain = brain_mask(a, protocol)
    ml = float(brain.sum()) * protocol.voxel_mm ** 3 / 1000
    if not brain.any():
        return unchanged(x, TOOL, VERSION, ARTIFACT, "no brain found (HD-BET), so no shading over it to measure")
    if ml < MIN_BRAIN_ML:
        return unchanged(x, TOOL, VERSION, ARTIFACT, f"HD-BET found only {ml:.0f} ml of brain (need ≥ "
                         f"{MIN_BRAIN_ML} ml for N4's field)", {"brain_ml": ml})
    m = brain & (a > 0)
    logf = n4_logfield(a, m, protocol.voxel_mm)
    r0 = _reading(logf, brain)
    rt = by_line(r0, LINE, 3)  # never printed on the wrong side of the line
    if not r0 >= LINE:  # the bias tool's own decision (present = reading ≥ line); a NaN reading is a no
        return unchanged(x, TOOL, VERSION, ARTIFACT, f"no shading beyond what a clean brain has: N4 reads {rt} "
                         f"(line {LINE})", {"reading_before": r0, "line": LINE, "brain_ml": ml})
    y = (a / np.exp(logf - logf[m].mean())).astype(np.float32)
    r1 = _reading(n4_logfield(y, m & (y > 0), protocol.voxel_mm), brain)
    sd = sides(logf, brain, protocol)
    b, d = WORDS[sd["bright_side"]], WORDS[sd["dark_side"]]
    why = (f"smooth shading over the brain, about {r0 / SCALE:.2f} max/min − 1 (N4 reads {rt}, line {LINE}): "
           f"brightest at the {b}, darkest at the {d}")
    what = f"smooth shading divided out: N4 reading {r0:.3f} → {r1:.3f}"
    ev = {"brightest": b, "darkest": d, "reading_before": r0, "reading_after": r1, "line": LINE, "brain_ml": ml}
    return y, FixReport(TOOL, VERSION, ARTIFACT, True, what, why, head_change(y, a, head_mask(a)), ev)
