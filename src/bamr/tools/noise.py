"""The noise tool, option A (D71): σ from the flat tissue alone; fineness is logged, never used.

Tile meter (aja2008, coupe2010): every 5 × 5 tile in each axial plane that lies fully in the flat tissue, its best
plane a + b·row + c·col removed; σ = sqrt(median tile variance), in % of WM (WM = 1).
Neighbour meter (immerkaer1996): median |3 × 3 Laplacian| / 0.6745 over the head; fineness = neighbour ÷ tile.
"""
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as swv
from scipy import ndimage as ndi

from bamr.tools.finding import PROTOCOL_V2, Finding, Protocol, check_volume
from bamr.tools.regions import INSET, TOL, flat_tissue, head_mask, inset

VERSION = 2  # D95: no band name in the finding
UNIT = "sigma, fraction of WM"
LINE = 0.030  # clean/noisy line: between clinical 'good' (2.1 %) and 'medium' (3.5 %) noise (D45, D51)
SCALE = 0.991  # D71: median raw/σ on the 30 dev-sweep noise volumes, 23 Oct (scripts/score_tools.py calibrate)
PATCH = 5
MIN_TILES = 1000
NEAR = 0.0015  # within 0.15 points of the line → low confidence
LAP = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], float) / 6.0  # unit gain for white noise


def _plane_residual(p: int) -> np.ndarray:
    yy, xx = np.mgrid[0:p, 0:p]
    a = np.c_[np.ones(p * p), yy.ravel(), xx.ravel()]
    return np.eye(p * p) - a @ np.linalg.pinv(a)


def tile_sigma(x: np.ndarray, region: np.ndarray, plane_axes: tuple[int, int], p: int = PATCH) -> tuple[float, int]:
    """Raw σ and the number of tiles, one plane at a time (keeps memory flat on a whole volume)."""
    other = 3 - sum(plane_axes)
    r = _plane_residual(p)
    var = []
    for xi, ri in zip(np.moveaxis(x, other, 0), np.moveaxis(region, other, 0)):
        if not ri.any():
            continue
        keep = swv(ri, (p, p)).all(axis=(-1, -2))
        if keep.any():
            t = swv(xi, (p, p))[keep].reshape(-1, p * p).astype(np.float64) @ r
            var.append((t ** 2).sum(axis=1) / (p * p - 3))
    if not var:
        return float("nan"), 0
    v = np.concatenate(var)
    return float(np.sqrt(np.median(v))), int(v.size)


def neighbour_sigma(x: np.ndarray, head: np.ndarray, plane_axes: tuple[int, int]) -> float:
    region = inset(head) >= 3
    if not region.any():
        return float("nan")
    k = np.expand_dims(LAP, 3 - sum(plane_axes))
    return float(np.median(np.abs(ndi.convolve(x.astype(np.float64), k)[region])) / 0.6745)


def _cannot(why: str, evidence: dict | None = None) -> Finding:
    return Finding("noise", VERSION, "noise", None, None, UNIT, "low", why, evidence or {}, None)


def _check(x: np.ndarray, protocol: Protocol) -> str | None:
    """Why this input cannot be measured, or None: the shared check, then room for one tile."""
    why = check_volume(x, protocol)
    if why:
        return why
    plane = [x.shape[i] for i, a in enumerate(protocol.array_axes) if a != "SI"]
    if x.size == 0 or min(plane) < PATCH:
        return f"volume {x.shape} is too small for one {PATCH} × {PATCH} tile"
    return None


def _too_few(x, head, flat, n, plane, ev) -> Finding:
    """Too little flat tissue: no head, a small head, or a big head whose inside is far from WM = 1."""
    need = f"only {n} tiles of flat tissue (need {MIN_TILES})"
    if not head.any():
        return _cannot("no head found, so no flat tissue to measure in", ev)
    inner = inset(head) >= INSET
    if inner.sum() < 10 * MIN_TILES:
        return _cannot(f"{need}: the head is too small", ev)
    ev["inner_median"] = float(np.median(x[inner]))
    ev["neighbour_sigma"] = neighbour_sigma(x, head, plane)
    if abs(ev["inner_median"] - 1) > 0.5:
        return _cannot(f"{need}: the head's inside has median {ev['inner_median']:.3g}, so white matter is not at 1 "
                       "(scale the input)", ev)
    return _cannot(f"{need}: just {flat.sum() / inner.sum():.0%} of the head's inside is within ±{TOL:.0%} of white "
                   f"matter, so the noise is very strong (neighbour meter: {100 * ev['neighbour_sigma']:.1f} % of WM)",
                   ev)


def noise_tool(x: np.ndarray, protocol: Protocol = PROTOCOL_V2, picture: Path | None = None) -> Finding:
    x = np.asarray(x)
    why = _check(x, protocol)
    if why:
        return _cannot(why)
    try:
        return _measure(x.astype(np.float32, copy=False), protocol, picture)
    except Exception as e:  # no crash inside an agent loop (D68): the error becomes the reason
        return _cannot(f"tool error: {type(e).__name__}: {e}")


def _measure(x: np.ndarray, protocol: Protocol, picture: Path | None) -> Finding:
    plane = tuple(i for i, a in enumerate(protocol.array_axes) if a != "SI")
    head = head_mask(x)
    flat = flat_tissue(x, head)
    raw, n = tile_sigma(x, flat, plane)
    ev = {"raw_sigma": raw if n else None, "scale": SCALE, "tiles": n, "flat_voxels": int(flat.sum()),
          "head_voxels": int(head.sum())}
    if n < MIN_TILES:
        return _too_few(x, head, flat, n, plane, ev)
    value = round(raw / SCALE, 6)  # D99 #9: decide on the value as the agent reads it
    ev["fineness"] = neighbour_sigma(x, head, plane) / raw if raw > 0 else None
    side = "above" if value >= LINE else "below"
    f = Finding("noise", VERSION, "noise", bool(value >= LINE), value, UNIT, "low" if abs(value - LINE) < NEAR else "high",
                f"σ {100 * value:.2f} % of WM, {100 * abs(value - LINE):.2f} points {side} the 3.0 % line", ev, None)
    if picture is not None:
        from bamr.tools.picture import evidence_picture
        evidence_picture(x, head, flat, f"noise: σ = {100 * value:.2f} % of WM, {side} the 3.0 % line", Path(picture))
        f.picture = Path(picture).name
    return f
