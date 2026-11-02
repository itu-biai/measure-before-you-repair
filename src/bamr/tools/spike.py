"""The spike tool (D96): bright points in 3-D k-space, each one a stripe wave across the image.

Search (foo1994, zhang2001): every sample 5–96 from the k-space centre, axis tubes and the mirror half left out. A
compact peak's 3³ power against the 7³ ring around it, as a robust z within its distance shell; a spike if z ≥ 7.
Strength (bracewell2003): a stripe a·cos(k·r) over the head puts (a/2)² × the head's own low-frequency power into
the 3³ around ±k, so a = 2·√(excess / that power), measured in the spectrum of the head alone (plan R1).
Echoes: a magnitude image makes two spikes beat at k1 − k2, a stripe that lives in the air, and a strong spike
leaves weak waves at 2k and k1 ± k2 in the head; both are listed apart, never counted as spikes.
"""
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi

from bamr.tools.finding import PROTOCOL_V2, Finding, Protocol, by_line, check_volume
from bamr.tools.kspace import excess, kgrid, low_freq_power, peak_contrast, shell_z, spectrum
from bamr.tools.regions import head_mask

VERSION = 1
UNIT = "stripe RMS in the head, fraction of WM"
LINE = 7.0  # D96: geometric middle of the dev sweep's gap (highest z without a spike 5.9, lowest with 8.3)
R_MIN, R_MAX = 5, 96  # outside the centre, inside the sphere the 240-sample axes hold
MARGIN = 5  # samples kept from each face: room for the 7³ ring and the one-sample sharpening (plan R2)
MAX_PEAKS = 4
AIR_GAP = 8  # air: more than 8 voxels from the head
ECHO_SHARE = 0.25  # a product of found spikes weaker than this share of the strongest is an echo
NEAR = 0.10  # best z within ±10 % of the line → low confidence
MIN_SIDE = 32
SCALE = 0.977  # D96/R1: median raw/spike_rms on the 30 dev-sweep spike volumes, 24 Oct (score_tools.py calibrate)


def _cannot(why: str, evidence: dict | None = None) -> Finding:
    return Finding("spike", VERSION, "spike", None, None, UNIT, "low", why, evidence or {}, None)


def _confidence(best_z: float, n_spikes: int, n_echoes: int) -> str:
    return "low" if abs(best_z - LINE) < NEAR * LINE or (n_echoes and not n_spikes) else "high"


def find_peaks(K: np.ndarray) -> tuple[float, list[dict]]:
    """Best z, and up to MAX_PEAKS peaks with z ≥ LINE (highest first): index, k and z of each."""
    P = np.abs(K).astype(np.float32) ** 2
    kk = kgrid(K.shape)
    r = np.sqrt(sum(k.astype(np.float64) ** 2 for k in kk))
    tube = sum((np.abs(k) <= 2).astype(np.int8) for k in kk) >= 2  # the box's edges leave ridges along the axes
    inside = np.ones(K.shape, bool)
    for k, n in zip(kk, K.shape):
        inside = inside & (np.abs(k) <= n // 2 - MARGIN)
    keep = (r >= R_MIN) & (r <= R_MAX) & inside & ~tube & (kk[2] >= 0)  # each spike shows at ±k: one half
    d, _, _ = peak_contrast(P)
    z = shell_z(d, np.round(r).astype(np.int32), keep)
    c = np.array(K.shape) // 2
    best = float(z.max()) if np.isfinite(z.max()) else float("nan")
    peaks = []
    for _ in range(MAX_PEAKS):
        i = np.array(np.unravel_index(int(np.argmax(z)), z.shape))
        zi = float(z[tuple(i)])
        if not round(zi, 6) >= LINE:
            break
        blk = P[tuple(slice(j - 1, j + 2) for j in i)]  # R2: the brightest sample of the 3³ is the peak
        j = i + np.array(np.unravel_index(int(np.argmax(blk)), blk.shape)) - 1
        peaks.append({"idx": tuple(int(v) for v in j), "k": [int(v) for v in j - c], "z": zi})
        for m in (i, 2 * c - i):  # set aside the 5³ around the peak and around its mirror
            z[tuple(slice(max(a - 2, 0), a + 3) for a in m)] = -np.inf
    return best, peaks


def _near(k, others, tol=1) -> bool:
    k = np.asarray(k)
    return any(np.abs(k - o).max() <= tol or np.abs(k + o).max() <= tol for o in others)


def _classify(peaks, x, head, voxel_mm) -> tuple[list[dict], list[dict]]:
    """Spikes and echoes. Echo: the stripe is stronger per voxel in the air than in the head (a beat), or the peak
    sits on 2k_i or k_i ± k_j of spikes already found and is weak next to them (a second-order product)."""
    air = ndi.distance_transform_edt(~head) > AIR_GAP
    KH = spectrum(np.where(head, x, 0))
    KA = spectrum(np.where(air, x, 0)) if air.any() else None
    h3, nh, na = low_freq_power(head), int(head.sum()), int(air.sum())
    spikes, echoes = [], []
    for p in peaks:
        eh = excess(KH, p["idx"])
        ea = excess(KA, p["idx"]) if KA is not None else 0.0
        a_head_vox, a_air = 2 * np.sqrt(eh) / nh, 2 * np.sqrt(ea) / max(na, 1)  # per voxel of each region
        f = np.array(p["k"]) / np.array(x.shape)
        q = {"k": p["k"], "z": p["z"], "stripe_mm": voxel_mm / float(np.sqrt((f ** 2).sum())),
             "direction": (f / np.sqrt((f ** 2).sum())).tolist(), "amplitude": 2 * np.sqrt(eh / h3),
             "air_share": a_air / a_head_vox if a_head_vox > 0 else (float("inf") if a_air > 0 else 0.0)}
        ks = [np.array(s["k"]) for s in spikes]
        products = [2 * a for a in ks] + [a + b for a in ks for b in ks] + [a - b for a in ks for b in ks]
        if q["air_share"] >= 1:
            echoes.append(q | {"reason": "stripe in the air"})
        elif spikes and _near(q["k"], products) and q["amplitude"] < ECHO_SHARE * max(s["amplitude"] for s in spikes):
            echoes.append(q | {"reason": "product of found spikes"})
        else:
            spikes.append(q)
    return spikes, echoes


def spike_tool(x: np.ndarray, protocol: Protocol = PROTOCOL_V2, picture: Path | None = None) -> Finding:
    x = np.asarray(x)
    why = check_volume(x, protocol)
    if not why and min(x.shape) < MIN_SIDE:
        why = f"volume {x.shape} is too small for the k-space search (need ≥ {MIN_SIDE} samples per side)"
    if why:
        return _cannot(why)
    try:
        return _measure(x.astype(np.float32, copy=False), protocol, picture)
    except Exception as e:  # no crash inside an agent loop (D68): the error becomes the reason
        return _cannot(f"tool error: {type(e).__name__}: {e}")


def _measure(x: np.ndarray, protocol: Protocol, picture: Path | None) -> Finding:
    head = head_mask(x)
    if not head.any():
        return _cannot("no head found, so no stripe strength to read")
    K = spectrum(x)
    best, peaks = find_peaks(K)
    if not np.isfinite(best):
        return _cannot("the spectrum has no spread to compare peaks against (a flat volume?)")
    spikes, echoes = _classify(peaks, x, head, protocol.voxel_mm) if peaks else ([], [])
    if spikes:
        raw = float(np.sqrt(sum(s["amplitude"] ** 2 for s in spikes) / 2))
    else:  # plan R3: only echoes found: the top one stands in for the strength, with low confidence
        raw = float(echoes[0]["amplitude"] / np.sqrt(2)) if echoes else 0.0
    best = round(best, 6)
    present = bool(best >= LINE)
    value = round(raw / SCALE, 6)
    zt = by_line(best, LINE, 1)
    if spikes:
        mm = " and ".join(f"{s['stripe_mm']:.0f}" for s in spikes)
        why = (f"{len(spikes)} bright k-space point{'s' if len(spikes) > 1 else ''}, z {zt} (line {LINE}): "
               f"stripes {mm} mm apart")
        if echoes:
            why += f"; {len(echoes)} echo{'es' if len(echoes) > 1 else ''} set aside"
    elif echoes:
        why = f"bright k-space point z {zt} (line {LINE}), but its stripe sits in the air"
    else:
        why = f"no bright k-space point: best z {zt} (line {LINE})"
    ev = {"spikes": spikes, "echoes": echoes, "line": LINE, "best_z": best, "raw": raw, "scale": SCALE,
          "head_voxels": int(head.sum())}
    f = Finding("spike", VERSION, "spike", present, value, UNIT, _confidence(best, len(spikes), len(echoes)), why, ev)
    if picture is not None:
        from bamr.tools.picture import spike_picture
        c = np.array(K.shape) // 2
        shown = [tuple(np.array(s["k"]) + c) for s in spikes] or [p["idx"] for p in peaks[:1]]
        spike_picture(x, K, shown, f"spike: {why}", Path(picture))
        f.picture = Path(picture).name
    return f
