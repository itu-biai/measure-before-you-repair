"""The ghosting tool, option C (D97): a rhythm in the A–P line energies, or a sharp copy of the head in the air.

Period view (wood1985, fisher1929): a periodic change over the slow phase-encode lines repeats in their energy. The
typical energy of each A–P plane of k-space (central disk), its fast wiggle, and the periodogram of that; a ghost if
the strongest peak is ≥ 150 × the typical bin. The copies' spacing is FOV ÷ period, where the period is the slowest
rhythm with ≥ 30 % of the strongest peak (every-4th-line dimming peaks as tall at every 2nd: an overtone).
Air view (giannelli2010, esteban2017): the air's intensity regressed on the head shifted by Δ along A–P, for every Δ
at once; a ghost if one shift stands out of its neighbours 6–15 voxels away by ≥ 0.0056.
Strength: the periodic share of the line energy (Parseval) × the head's RMS ÷ one scale factor.
"""
import warnings
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from bamr.tools.finding import PROTOCOL_V2, Finding, Protocol, by_line, check_volume
from bamr.tools.kspace import line_energy, periodogram, spectrum
from bamr.tools.regions import head_mask

VERSION = 1
UNIT = "ghost RMS in the head, fraction of WM"
PEAK_LINE = 150.0  # D97: the middle of the dev sweep's gap
AIR_LINE = 0.0056  # D97: 2 × the most any dev-sweep volume without ghosting shows
SUB_SHARE = 0.30  # D97: a slower rhythm counts if its peak is ≥ 30 % of the strongest
F_MIN = 0.1  # periods ≤ 10 lines: copies ≥ 24 mm from the head
SIG = 10.0  # bins above 10 × the typical one carry the periodic share
DISK = 64
AIR_GAP = 8
SHIFT_MIN, SHIFT_MAX = 20, 120
NB = np.r_[-15:-5, 6:16]  # neighbours a copy is compared with
NEAR = 0.10  # the deciding score within ±10 % of its line → low confidence
MIN_SIDE = 32
SCALE = 1.400  # D97: median raw/ghosting_rms on the 30 dev-sweep ghosting volumes, 24 Oct (score_tools.py calibrate)


def _cannot(why: str, evidence: dict | None = None) -> Finding:
    return Finding("ghosting", VERSION, "ghosting", None, None, UNIT, "low", why, evidence or {}, None)


def _decide(peak: float, prom: float) -> tuple[bool, list[str], str]:
    views = (["period"] if peak >= PEAK_LINE else []) + (["air"] if prom >= AIR_LINE else [])
    score = max(peak / PEAK_LINE, prom / AIR_LINE)
    return bool(views), views, "low" if views == ["air"] or abs(score - 1) < NEAR else "high"


def period_view(K: np.ndarray, axis: int) -> dict:
    E = line_energy(K, axis, DISK)
    f, pw, rz = periodogram(E)
    band = f >= F_MIN
    med = float(np.median(pw[band]))
    rel = pw / med if med > 0 else np.zeros_like(pw)
    j = int(np.argmax(np.where(band, rel, -1)))
    best = j
    for m in range(2, 6):  # the slowest rhythm with a real peak: the fundamental, not an overtone
        if f[j] / m < F_MIN - 1e-9:
            break
        b = int(np.argmin(np.abs(f - f[j] / m)))
        if rel[b] >= SUB_SHARE * rel[j]:
            best = b
    on = band & (pw > SIG * med)
    w = np.where(np.isclose(f, 0.5), 1.0, 2.0)  # every bin but the last stands for ±f
    V = float(np.sum((w * (pw - med))[on]) / rz.size ** 2)
    return {"peak": float(rel[j]), "f_star": float(f[j]), "period_lines": float(1 / f[best]) if f[best] > 0 else None,
            "spec_amp": float(np.sqrt(max(V, 0.0)) / 2), "rz": rz, "f": f, "rel": rel}


def _measured(den: np.ndarray) -> np.ndarray:
    """Shifts where the shifted head reaches the air at all: the variance is above round-off (1e-6 × the largest)."""
    top = float(np.max(den)) if den.size else 0.0
    return den > 1e-6 * top if top > 0 else np.zeros(den.shape, bool)


def _symmetric(g: np.ndarray) -> np.ndarray:
    """A copy sits at +Δ and at −Δ: the mean of the two, or the one that could be measured (review I1)."""
    pair = np.stack([g, np.roll(g[::-1], 1)])
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # both sides unmeasured: NaN, on purpose
        return np.nanmean(pair, axis=0)


def air_curve(x: np.ndarray, head: np.ndarray, axis: int, air: np.ndarray | None = None) -> np.ndarray:
    """Ghost-to-signal g(Δ) for every shift Δ: the air's intensity regressed on the head image shifted by Δ, via FFTs;
    symmetric (a copy sits at ±Δ), NaN where the shifted head never reaches the air. Uses the air's intensities
    relative to each other, so exact-zero air reads 0 like faint noise does (D69). The motion tool reads it too, over
the air without its exact zeros (D107 = B)."""
    air = ndi.distance_transform_edt(~head) > AIR_GAP if air is None else air
    N = x.shape[axis]
    if not air.any():
        return np.full(N, np.nan)
    s = np.where(head, x, 0).astype(np.float32)
    other = tuple(a for a in range(3) if a != axis)
    F = lambda a: np.fft.rfft(a, axis=axis)  # noqa: E731
    corr = lambda fa, fb: np.fft.irfft(fa * np.conj(fb), n=N, axis=axis).sum(axis=other)  # Σ a(y)·b(y − Δ)
    fa, fs = F(air.astype(np.float32)), F(s)
    A, Sx = float(air.sum()), float(x[air].sum())
    Sxs, Ss, Sss = corr(F(np.where(air, x, 0).astype(np.float32)), fs), corr(fa, fs), corr(fa, F(s * s))
    den = Sss / A - (Ss / A) ** 2
    ok = _measured(den)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.where(ok, (Sxs / A - Sx / A * Ss / A) / np.where(ok, den, 1.0), np.nan)
    return _symmetric(g)


def air_view(x: np.ndarray, head: np.ndarray, axis: int) -> dict:
    """The copy: the shift whose g stands out most from its neighbours 6–15 voxels away."""
    gs = air_curve(x, head, axis)
    N = gs.size
    if not np.isfinite(gs).any():
        return {"prominence": 0.0, "shift": 0, "share": 0.0}
    with np.errstate(invalid="ignore"):
        prom = np.array([gs[a] - np.nanmedian(gs[(a + NB) % N]) if np.isfinite(gs[(a + NB) % N]).any() else np.nan
                         for a in range(N)])
    d = np.arange(N)
    ok = (d >= SHIFT_MIN) & (d <= min(SHIFT_MAX, N // 2)) & np.isfinite(prom)
    if not ok.any():
        return {"prominence": 0.0, "shift": 0, "share": 0.0}
    jp = int(np.argmax(np.where(ok, prom, -np.inf)))
    return {"prominence": float(prom[jp]), "shift": jp, "share": float(gs[jp])}


def ghosting_tool(x: np.ndarray, protocol: Protocol = PROTOCOL_V2, picture: Path | None = None) -> Finding:
    x = np.asarray(x)
    why = check_volume(x, protocol)
    if not why and min(x.shape) < MIN_SIDE:
        why = f"volume {x.shape} is too small for the line-energy rhythm (need ≥ {MIN_SIDE} samples per side)"
    if why:
        return _cannot(why)
    try:
        return _measure(x.astype(np.float32, copy=False), protocol, picture)
    except Exception as e:  # no crash inside an agent loop (D68): the error becomes the reason
        return _cannot(f"tool error: {type(e).__name__}: {e}")


def _measure(x: np.ndarray, protocol: Protocol, picture: Path | None) -> Finding:
    head = head_mask(x)
    if not head.any():
        return _cannot("no head found, so no copies of it to look for")
    ap = protocol.array_axes.index(protocol.phase_slow)
    name = f"{protocol.phase_slow[0]}–{protocol.phase_slow[1]}"
    pv, av = period_view(spectrum(x), ap), air_view(x, head, ap)
    peak, prom = round(pv["peak"], 6), round(av["prominence"], 6)
    present, views, confidence = _decide(peak, prom)
    head_rms = float(np.sqrt(np.mean(x[head].astype(np.float64) ** 2)))
    raw = pv["spec_amp"] * head_rms
    fov = x.shape[ap] * protocol.voxel_mm
    air_mm = av["shift"] * protocol.voxel_mm
    period = pv["period_lines"] if "period" in views else None
    copy_mm = fov / period if period else (air_mm if "air" in views else None)
    pt, at = by_line(peak, PEAK_LINE, 0), by_line(prom, AIR_LINE, 4)  # never printed on the wrong side of a line
    air_txt = f"a copy of the head in the air {air_mm:.0f} mm along {name} (prominence {at}, line {AIR_LINE})"
    if period:
        why = (f"{name} line energy repeats every {period:.1f} lines (peak {pt}, line {PEAK_LINE:.0f}): "
               f"copies every {copy_mm:.0f} mm along {name}") + (f"; also {air_txt}" if "air" in views else "")
    elif views:
        why = f"{air_txt}; the line energy barely repeats (peak {pt}), so the strength reads low"
    else:
        why = (f"no repeat in the {name} line energy (peak {pt}, line {PEAK_LINE:.0f}) and no copy in the air "
               f"({at}, line {AIR_LINE})")
    ev = {"period_lines": period, "copy_mm": copy_mm, "peak": peak, "peak_line": PEAK_LINE,
          "air_copy": {"shift_mm": air_mm, "share": av["share"], "prominence": prom}, "air_line": AIR_LINE,
          "views": views, "spec_amp": pv["spec_amp"], "head_rms": head_rms, "raw": raw, "scale": SCALE}
    f = Finding("ghosting", VERSION, "ghosting", present, round(raw / SCALE, 6), UNIT, confidence, why, ev)
    if picture is not None:
        from bamr.tools.picture import ghost_picture
        ghost_picture(x, ap, pv["rz"], pv["f"], pv["rel"], period, copy_mm, f"ghosting: {why}", Path(picture))
        f.picture = Path(picture).name
    return f
