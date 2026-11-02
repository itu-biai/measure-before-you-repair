"""The motion tool (D103 = A, D105): head signal leaking into the air along the slow phase-encode axis.

A head that moves between blocks of A–P k-space lines mixes two poses in k-space; in the image the head smears along
whole A–P lines (zaitsev2015), so the air in front of and behind it carries a faint copy of it at every shift.
Leak: the Stage 2 air view's g(Δ) (the air regressed on the head shifted by Δ along A–P, `ghosting.air_curve`), its
median over Δ = 20–120 voxels. Reading artifacts in the air is Mortamet's idea (mortamet2009, MRIQC's QI1); counting
only along the slow phase-encode axis is ours. The same leak along L–R and S–I is reported as a control.
Two blind spots, kept visible (D105): ghosting leaks the same way, so when the ghosting tool's period view hears a
rhythm (peak ≥ 150) the answer is "cannot tell"; noisy air hides a weak leak (|s + n| barely moves its mean when
s ≪ σ, gudbjartsson1995), so when the noise tool says noisy, or cannot measure the noise, the confidence is low.
Exact zeros are left out of the air (D107 = B): a zero is no measurement (D69), and zeros spread unevenly around the
head would read as a smear.
Value: leak ÷ one scale factor, in the label's unit (whole-brain damage), never below 0.
"""
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from bamr.tools import ghosting as G
from bamr.tools.finding import PROTOCOL_V2, Finding, Protocol, by_line, check_volume
from bamr.tools.kspace import spectrum
from bamr.tools.noise import noise_tool
from bamr.tools.regions import head_mask

VERSION = 2  # D107: exact-zero air left out
UNIT = "whole-brain damage, fraction of WM"
LINE = 0.0038  # D103: between the dev sweep's highest leak without motion and lowest with (pilot)
RHYTHM_LINE = G.PEAK_LINE  # D103: the ghosting tool's period-view line (150)
SHIFT_MIN, SHIFT_MAX = 20, 120
NEAR = 0.10  # leak within ±10 % of the line → low confidence
MIN_SIDE = 32
SCALE = 0.230  # D103: median leak/motion_rms on the 30 dev-sweep motion volumes (pilot 1/4.34; score_tools calibrate)


def _cannot(why: str, evidence: dict | None = None) -> Finding:
    return Finding("motion", VERSION, "motion", None, None, UNIT, "low", why, evidence or {}, None)


def leak(gs: np.ndarray) -> float:
    """Median of g over shifts 20–120 (or to half the axis, past which the symmetric curve repeats)."""
    d = np.arange(gs.size)
    w = gs[(d >= SHIFT_MIN) & (d <= min(SHIFT_MAX, gs.size // 2))]
    return float(np.nanmedian(w)) if np.isfinite(w).any() else float("nan")


def motion_tool(x: np.ndarray, protocol: Protocol = PROTOCOL_V2, picture: Path | None = None) -> Finding:
    x = np.asarray(x)
    why = check_volume(x, protocol)
    if not why and min(x.shape) < MIN_SIDE:
        why = f"volume {x.shape} is too small for the air view (need ≥ {MIN_SIDE} samples per side)"
    if why:
        return _cannot(why)
    try:
        return _measure(x.astype(np.float32, copy=False), protocol, picture)
    except Exception as e:  # no crash inside an agent loop (D68): the error becomes the reason
        return _cannot(f"tool error: {type(e).__name__}: {e}")


def _measure(x: np.ndarray, protocol: Protocol, picture: Path | None) -> Finding:
    head = head_mask(x)
    if not head.any():
        return _cannot("no head found, so no head signal to follow into the air")
    air = ndi.distance_transform_edt(~head) > G.AIR_GAP
    if not air.any():
        return _cannot(f"the head fills the box: no air more than {G.AIR_GAP} voxels from it to look in")
    air &= x != 0  # D107 = B: a zero is no measurement (D69)
    if not air.any():
        return _cannot("the air is all exact zeros (blanked, not measured), so there is no air to read a leak in")
    axes = {a: protocol.array_axes.index(a) for a in ("AP", "LR", "SI")}
    curves = {a: G.air_curve(x, head, i, air) for a, i in axes.items()}
    lk = {a: round(leak(g), 6) for a, g in curves.items()}
    ap = protocol.phase_slow
    if not np.isfinite(lk[ap]):
        return _cannot("the shifted head never reaches the air along A–P, so no leak can be read")
    peak = round(float(G.period_view(spectrum(x), axes[ap])["peak"]), 6)
    nf = noise_tool(x, protocol)
    noisy, sigma = nf.present, nf.value  # None: the noise could not be measured
    ev = {"leak_ap": lk["AP"], "leak_lr": lk["LR"], "leak_si": lk["SI"], "raw": lk[ap], "line": LINE, "scale": SCALE,
          "rhythm_peak": peak, "rhythm_line": RHYTHM_LINE, "noisy_air": noisy, "noise_sigma": sigma}
    name = f"{ap[0]}–{ap[1]}"
    lt = by_line(lk[ap], LINE, 4)  # never printed on the wrong side of the line
    noise_txt = f"σ {100 * sigma:.1f} % of WM" if sigma is not None else "σ not measured"
    if peak >= RHYTHM_LINE:
        why = (f"ghosting found ({name} rhythm, peak {by_line(peak, RHYTHM_LINE, 0)}, line {RHYTHM_LINE:.0f}): it "
               f"leaks along {name} the same way, so motion cannot be told apart here (leak {lt}, line {LINE})")
        f = Finding("motion", VERSION, "motion", None, None, UNIT, "low", why, ev)
    else:
        present = bool(lk[ap] >= LINE)
        others = ", ".join(f"{a[0]}–{a[1]} {lk[a]:.4f}" for a in ("LR", "SI"))
        if present:
            why = f"head signal leaks into the air along {name} ({lt}, line {LINE}); {others}"
            why += f"; noisy air ({noise_txt}) can make it read weaker" if noisy else ""
            why += "; the noise could not be measured, and noise can make it read weaker" if noisy is None else ""
        else:
            why = f"no leak of head signal into the air along {name} ({lt}, line {LINE})"
            why += f"; the air is noisy ({noise_txt}), which can hide a weak leak" if noisy else ""
            why += "; the noise could not be measured, and noise can hide a weak leak" if noisy is None else ""
        low = abs(lk[ap] / LINE - 1) < NEAR or noisy is not False
        f = Finding("motion", VERSION, "motion", present, round(max(lk[ap], 0.0) / SCALE, 6), UNIT,
                    "low" if low else "high", why, ev)
    if picture is not None:
        from bamr.tools.picture import motion_picture
        motion_picture(x, curves, protocol, LINE, f"motion: {f.why}", Path(picture))
        f.picture = Path(picture).name
    return f
