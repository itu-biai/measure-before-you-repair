"""The spike fix (SE, D113): each stripe the spike tool lists, taken out of the magnitude image.

Our files hold magnitudes: input = |image + Z|, Z = Σ C_i·e^(i2πk_i·r), one complex wave per bad k-space sample
(bracewell2003; how TorchIO and a scanner make it). Kao & MacFall estimate a spike analytically from a region of known
signal and warn against patching a sample from its neighbours (kao2000); SE does that on the image side:
1. Each found stripe's C fitted over the head, where it adds to the image (x ≈ image + Re Z), each with the others
   taken out, twice. Which way each stripe turns (+k or −k) is chosen by the air.
2. The far air (> 8 voxels from the head): its noise σ by Rician maximum likelihood with ν = |Z|, the top 1 % left out
   (gudbjartsson1995). Quiet (σ ≤ 0.2 |Z|): the true air is ≈ 0, so x ≈ |Z| there pins the stripes' common size and
   relative phases (robust soft-L1 fit; the common phase is invisible in |Z|, the head's is kept). Noisy: the head fit.
3. The magnitude inverted: x² = (image + Re Z)² + (Im Z)² has two roots. In the head, the larger. Elsewhere, where both
   are ≥ 0 (within 0.1 |Z|), the one nearer the 3³ median, 3 passes. In noisy air instead x × Rician mean(0, σ) ÷
   Rician mean(|Z|, σ): the air's expected brightness no longer follows Z.
4. The spike tool is asked again; a spike it now lists that was not yet removed (a weaker one the first was hiding) is
   repaired too, at most 3 rounds.
Limit: this assumes the file is |image + stripe|, a single-coil magnitude without background phase. On multi-coil
scans with phase the stripe spreads; there the repair belongs in the raw complex k-space. Exact-zero voxels of the
input are never read and stay 0 (D69).
"""
import itertools

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import least_squares, minimize_scalar
from scipy.special import i0e, i1e

from bamr.fixes.report import FixReport, head_change, unchanged
from bamr.tools.finding import PROTOCOL_V2, Protocol, by_line
from bamr.tools.regions import head_mask
from bamr.tools.spike import AIR_GAP, LINE, spike_tool

TOOL, VERSION, ARTIFACT = "fix_spike", 1, "spike"
TAU = 0.1  # a root within 0.1 × |Z| below 0 counts as ≥ 0: the fitted Z is off by ~2 %, a true 0 must stay a choice
QUIET = 0.2  # far-air σ ÷ the stripes' RMS |Z| at or below this: the air is "known ≈ 0" (kao2000's known region)
MAX_ROUNDS = 3
MEDIAN_PASSES = 3
FAR_MIN = 1000  # far-air voxels needed to read σ
SAMPLE = 200_000  # far-air voxels used for σ and the quiet fit (a fixed draw)
TOP_PCT = 99  # the brightest 1 % of the far air left out of σ: structures, not noise
F_SCALE = 0.05  # soft-L1 scale, in units of |Z|: bright structures in the air do not pull the fit
SIZE_BOUNDS = (0.5, 2.0)
NEAR_TOL = 1  # a spike within 1 sample of one already removed (or its mirror) is that one
AIR_WORDS = {"quiet": "far air quiet", "noisy": "far air noisy", "no far air": "no far air"}


def rice_mean(nu, sigma):
    """Mean of a Rician magnitude |ν + noise|, noise σ per channel (gudbjartsson1995): σ√(π/2)·L½(−ν²/2σ²)."""
    t = np.asarray(nu, np.float64) ** 2 / (2 * sigma ** 2)
    return sigma * np.sqrt(np.pi / 2) * ((1 + t) * i0e(t / 2) + t * i1e(t / 2))


def rician_sigma(xs, nu) -> float:
    """Maximum-likelihood σ of Rician samples xs with known ν per sample; the top 1 % left out."""
    keep = xs <= np.percentile(xs, TOP_PCT)
    xs, nu = xs[keep].astype(np.float64), nu[keep].astype(np.float64)

    def nll(ls):
        s2 = np.exp(2 * ls)
        a = xs * nu / s2
        return -np.sum(np.log(np.maximum(xs, 1e-12)) - 2 * ls - (xs ** 2 + nu ** 2) / (2 * s2) + np.log(i0e(a)) + a)
    return float(np.exp(minimize_scalar(nll, bounds=(np.log(1e-5), np.log(0.5)), method="bounded").x))


def _waves(shape, ks) -> list[np.ndarray]:
    g = np.ogrid[tuple(slice(0, s) for s in shape)]
    return [np.exp(1j * sum((2 * np.pi * k[a] / shape[a]) * g[a] for a in range(3))).astype(np.complex64) for k in ks]


def _stripes(A, sg, E) -> tuple[np.ndarray, np.ndarray]:
    """Re Z and Im Z of Z = Σ 2A_i·e_i, stripe i's imaginary part flipped where sg_i = −1 (a stripe at −k)."""
    return (sum(np.real(2 * a * e) for a, e in zip(A, E)).astype(np.float32),
            sum(s * np.imag(2 * a * e) for a, e, s in zip(A, E, sg)).astype(np.float32))


def se_repair(x: np.ndarray, ks, head: np.ndarray, blank: np.ndarray) -> tuple[np.ndarray, dict]:
    """One round of SE on the listed ks; `blank` = the input's exact zeros (never read, kept 0). Returns the volume
    and what was done: mode (quiet · noisy · no far air), σ ÷ |Z|, σ, the size factor, the phase shifts in degrees."""
    x = np.asarray(x, np.float32)
    E = _waves(x.shape, ks)
    n = len(ks)
    A, Zr, H = np.zeros(n, complex), np.zeros(x.shape, np.float32), float(head.sum())
    for _ in range(2):  # (1) the head fit, each stripe with the others taken out, twice
        for i, e in enumerate(E):
            Zr -= 2 * np.real(A[i] * e)
            A[i] = complex(((x - Zr) * np.conj(e))[head].sum() / H)
            Zr += 2 * np.real(A[i] * e)
    air = ~head & ~blank
    far = (ndi.distance_transform_edt(~head) > AIR_GAP) & ~blank
    best = None
    for signs in itertools.product((1, -1), repeat=max(n - 1, 0)):  # one stripe: the sign does not matter
        sg = (1,) + signs
        zr, zi = _stripes(A, sg, E)
        err = float(np.mean((x[air] - np.sqrt(zr[air] ** 2 + zi[air] ** 2)) ** 2))
        if best is None or err < best[0]:
            best = (err, sg)
    sg = best[1]
    info = {"mode": "no far air", "sigma_ratio": None, "sigma": None, "size": 1.0, "shifts_deg": [0.0] * (n - 1)}
    if far.sum() > FAR_MIN:  # (2) the far air: quiet (≈ 0) or noisy?
        idx = np.random.default_rng(0).choice(np.flatnonzero(far.ravel()), min(SAMPLE, int(far.sum())), replace=False)
        xs, es = x.ravel()[idx], [e.ravel()[idx] for e in E]
        zs = np.hypot(*_stripes(A, sg, es))
        zrms = float(np.sqrt(np.mean(zs ** 2)))
        sigma = rician_sigma(xs, zs)
        info |= {"sigma_ratio": sigma / zrms, "sigma": sigma}
        if sigma / zrms <= QUIET:  # quiet: x ≈ |Z| pins the size and the stripes' relative phases
            lo, hi = np.r_[SIZE_BOUNDS[0], -np.full(n - 1, np.pi)], np.r_[SIZE_BOUNDS[1], np.full(n - 1, np.pi)]

            def resid(p):
                return (np.hypot(*_stripes(A * p[0] * np.exp(1j * np.r_[0.0, p[1:]] * np.array(sg)), sg, es)) - xs) / zrms
            p = least_squares(resid, np.r_[1.0, np.zeros(n - 1)], loss="soft_l1", f_scale=F_SCALE, bounds=(lo, hi)).x
            psi = np.r_[0.0, p[1:]]
            psi -= psi.mean()  # the common phase is invisible in |Z|: the head's mean phase kept
            A = A * p[0] * np.exp(1j * psi * np.array(sg))
            info |= {"mode": "quiet", "size": float(p[0]), "shifts_deg": np.degrees(p[1:]).round(2).tolist()}
        else:
            info["mode"] = "noisy"
    zr, zi = _stripes(A, sg, E)
    s = np.sqrt(np.maximum(x ** 2 - zi ** 2, 0))  # (3) the two roots of x² = (image + Re Z)² + (Im Z)²
    hi, lo = s - zr, -s - zr
    amb = lo >= -TAU * np.sqrt(zr ** 2 + zi ** 2)
    hi, lo = np.maximum(hi, 0), np.maximum(lo, 0)
    y = np.where(head, hi, np.where(np.abs(lo) < np.abs(hi), lo, hi)).astype(np.float32)
    for _ in range(MEDIAN_PASSES):
        m = ndi.median_filter(y, size=3)
        y = np.where(amb & (np.abs(lo - m) < np.abs(hi - m)), lo, hi).astype(np.float32)
    if info["mode"] == "noisy":  # the air is noise of random phase: no single root; flatten its expected brightness
        z = np.hypot(zr, zi)
        y = np.where(head, y, x * rice_mean(0.0, info["sigma"]) / rice_mean(z, info["sigma"])).astype(np.float32)
    y[blank] = 0
    return y, info


def _near_any(k, seen, tol=NEAR_TOL) -> bool:
    k = np.asarray(k)
    return any(np.abs(k - np.asarray(s)).max() <= tol or np.abs(k + np.asarray(s)).max() <= tol for s in seen)


def fix_spike(x: np.ndarray, protocol: Protocol = PROTOCOL_V2) -> tuple[np.ndarray, FixReport]:
    try:
        f = spike_tool(x, protocol)
        before = {"present": f.present, "best_z": f.evidence.get("best_z"), "spikes": len(f.evidence.get("spikes", [])),
                  "echoes": len(f.evidence.get("echoes", []))}
        if f.present is not True:
            return unchanged(x, TOOL, VERSION, ARTIFACT, f.why, {"spike_tool_before": before})
        if not f.evidence.get("spikes"):
            return unchanged(x, TOOL, VERSION, ARTIFACT, f.why + "; only echoes (stripes in the air), nothing repaired",
                             {"spike_tool_before": before})
        return _repair(np.asarray(x, np.float32), f, before, protocol)
    except Exception as e:  # no crash inside an agent loop (D68): the error becomes the reason
        return unchanged(x, TOOL, VERSION, ARTIFACT, f"fix error: {type(e).__name__}: {e}")


def _repair(x: np.ndarray, f, before: dict, protocol: Protocol) -> tuple[np.ndarray, FixReport]:
    blank = x == 0
    y, g, seen, mm, rounds = x, f, [], [], []
    todo = f.evidence["spikes"]
    while todo and len(rounds) < MAX_ROUNDS:
        ks = [s["k"] for s in todo]
        y, info = se_repair(y, ks, head_mask(y), blank)
        seen += ks
        mm += [s["stripe_mm"] for s in todo]
        g = spike_tool(y, protocol)
        rounds.append({"k": ks, "air": info["mode"], "size": info["size"], "shift_deg": info["shifts_deg"],
                       "sigma_ratio": info["sigma_ratio"], "best_z_after": g.evidence.get("best_z")})
        todo = [s for s in g.evidence.get("spikes", []) if not _near_any(s["k"], seen)] if g.present else []
    n, r = len(seen), len(rounds)
    airs = " then ".join(dict.fromkeys(AIR_WORDS[q["air"]] for q in rounds))
    za = g.evidence.get("best_z")
    after = "no reading" if za is None else f"best z {by_line(za, LINE, 1)} (line {LINE})"
    what = (f"{n} stripe{'s' if n > 1 else ''} removed ({' and '.join(f'{m:.0f}' for m in mm)} mm apart), "
            f"{r} round{'s' if r > 1 else ''}; {airs}; spike tool after: {after}")
    ev = {"rounds": rounds, "spike_tool_before": before,
          "spike_tool_after": {"present": g.present, "best_z": za, "why": g.why}}
    return y, FixReport(TOOL, VERSION, ARTIFACT, True, what, f.why, head_change(y, x, head_mask(x)), ev)
