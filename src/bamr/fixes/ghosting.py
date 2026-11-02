"""The ghosting fix (D122 A, GF): the planes our ghosting dimmed, put back from the magnitude image.

The model (wood1985; TorchIO's Ghosting, our simulator): every n-th A–P plane of k-space dimmed by (1 − I), the centre
plane kept. In the image that subtracts I × the mean of n shifted copies: the head dimmed by I/n and n − 1 negative
copies every FOV/n along A–P. Our files hold the magnitude, which turns the copies in the air positive (before-doc §2).
1. The ghosting tool (Task 2, D97) reads the period. "No", "cannot tell", no period (a copy in the air only) or a period
   further than 0.1 from a whole number (not whole planes, so not this model; before-doc §7): the volume comes back as
   given, with the reason.
2. I is read in the air (> 4 voxels from the head), where the copies are: x² = I²h² + 2σ², h = the copies per unit I
   that the head predicts, the noise as the intercept (least squares); 3 rounds, h from the current estimate.
3. The sign the magnitude lost, by error reduction with the head (dilated 3 voxels) as the support: outside it, the
   sign the estimate predicts (fienup1982; lee2002's POCS ghost correction uses the object's support the same way).
4. On the dimmed planes only the missing signal I·S is put back, S by a Wiener weight from the two neighbouring planes'
   power over the k-space corners' noise floor: the noise came after the ghost, so ÷ (1 − I) would lift it on every
   n-th plane, a rhythm the ghosting tool hears.
5. Exact-zero voxels of the input stay 0, and I is never read from them (D69). The ghosting and motion tools are asked
   again on the output, for the report.
Limits (before-doc §7, §12): it inverts our simulator's model, whole planes dimmed in a real image with the centre plane
kept. With a scanner's image phase it removes less; a period off a whole number it declines. The dimmed planes are taken
as k ≡ 0 (mod n) and the copies as N/n apart, as TorchIO's when n divides N/2 (n = 2, 3, 4 on our 240 A–P planes).
The sign step still sees exact zeros as "no signal": a scanner that blanks the air near the head is outside this fix.
"""
import numpy as np
import scipy.fft as sfft
from scipy import ndimage as ndi

from bamr.fixes.report import FixReport, head_change, unchanged
from bamr.tools.finding import PROTOCOL_V2, Protocol
from bamr.tools.ghosting import ghosting_tool
from bamr.tools.motion import motion_tool
from bamr.tools.regions import head_mask

TOOL, VERSION, ARTIFACT = "fix_ghosting", 1, "ghosting"
METHOD = "GF: I read in the air, sign by error reduction, missing signal put back (Wiener); no weights"
PERIOD_TOL = 0.1  # a period further than this from a whole number is not whole planes dimmed (before-doc §7)
AIR_OUT = 4  # the air where I is read: > 4 voxels from the head
AIR_MIN = 1000  # air voxels (not exact zeros) needed to read I
SUPPORT = 3  # the head dilated by 3 voxels: where the image may be anything; outside it, ≥ 0 is enforced
ROUNDS, ITERS_FIT, ITERS_FINAL = 3, 3, 5
I_MAX = 0.95  # 1 − I is divided by: never let it reach 0
WIENER = (5, 1, 5)  # the neighbouring planes' power averaged over 5 × 5 samples in the plane
CORNER = 0.4  # k-space corners (|k| > 0.4 N on every axis): the noise floor
AP = 1  # the work is done with A–P moved to array axis 1
WORD = {True: "yes", False: "no", None: "cannot tell"}


def ordinal(n: int) -> str:
    return "th" if n % 100 in (11, 12, 13) else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def kgain(N: int, n: int, I: float) -> np.ndarray:
    """Gain of each A–P plane (FFT order): (1 − I) on the planes k ≡ 0 (mod n), except the centre; 1 elsewhere."""
    k = np.round(np.fft.fftfreq(N) * N).astype(int)
    return np.where((k % n == 0) & (k != 0), 1 - I, 1.0)


def _along(v: np.ndarray, m: np.ndarray, inv: bool = False) -> np.ndarray:
    """v with its A–P planes × m (÷ m when inv). m is even in k, so the real FFT is exact."""
    F = sfft.rfft(v, axis=AP, workers=1)
    F *= (1 / m if inv else m)[: F.shape[AP]].reshape(1, -1, 1)
    return sfft.irfft(F, n=v.shape[AP], axis=AP, workers=1).astype(np.float32)


def _er(x: np.ndarray, m: np.ndarray, supp: np.ndarray, iters: int) -> tuple[np.ndarray, np.ndarray]:
    """Error reduction (fienup1982): outside the support the ghosted image takes the sign the current estimate predicts.
    → (the input with that sign, the estimate)."""
    y, f = x, _along(x, m, True)
    for _ in range(iters):
        g = _along(np.where(supp, np.maximum(f, 0), 0), m)
        y = np.where(supp | (g >= 0), x, -x)
        f = _along(y, m, True)
    return y, f


def _copies(fs: np.ndarray, n: int) -> np.ndarray:
    """The signed air pattern per unit I: the A–P line mean (the centre plane kept) minus the mean of the n shifted
    copies."""
    N = fs.shape[AP]
    return fs.mean(axis=AP, keepdims=True) - sum(np.roll(fs, j * N // n, axis=AP) for j in range(n)) / n


def _fit(x: np.ndarray, h: np.ndarray, air: np.ndarray) -> tuple[float, float]:
    """x² = I²·h² + 2σ² over the air (least squares with an intercept: the noise floor) → (I, σ)."""
    a2, y2 = h[air].astype(np.float64) ** 2, x[air].astype(np.float64) ** 2
    (s, c), *_ = np.linalg.lstsq(np.stack([a2, np.ones_like(a2)], 1), y2, rcond=None)
    return float(np.sqrt(max(s, 0))), float(np.sqrt(max(c, 0) / 2))


def strength(x: np.ndarray, n: int, head: np.ndarray, air: np.ndarray) -> tuple[float, float]:
    """(I, σ in the air): first with the ghosted head standing in for the clean one (first-order corrected), then 3
    rounds with the copies the current estimate predicts."""
    I, sig = _fit(x, _copies(np.where(head, x, 0).astype(np.float32), n), air)
    I = I / (1 + (n - 1) * I / n)
    supp = ndi.binary_dilation(head, iterations=SUPPORT)
    for _ in range(ROUNDS):
        _, f = _er(x, kgain(x.shape[AP], n, min(I, I_MAX)), supp, ITERS_FIT)
        I, sig = _fit(x, _copies(np.where(supp, np.maximum(f, 0), 0), n), air)
    return float(np.clip(I, 0, I_MAX)), sig


def addback(y: np.ndarray, n: int, I: float) -> np.ndarray:
    """The noise after the ghosting was not dimmed: on the dimmed samples put back I·S only, S by a Wiener weight from
    the two neighbouring planes' power over the noise floor (the k-space corners)."""
    K = sfft.fftn(y, workers=1)
    N = y.shape[AP]
    k = np.round(np.fft.fftfreq(N) * N).astype(int)
    dim = np.flatnonzero((k % n == 0) & (k != 0))
    P = np.abs(K) ** 2
    c = [np.abs(np.round(np.fft.fftfreq(m) * m)) > CORNER * m for m in y.shape]
    F = float(np.median(P[np.ix_(*c)]))
    Pn = 0.5 * (P[:, (dim - 1) % N, :] + P[:, (dim + 1) % N, :])
    a = (1 - I) ** 2 * np.maximum(ndi.uniform_filter(Pn, WIENER) - F, 0)
    W = a / (a + F) if F > 0 else np.ones_like(a)
    K[:, dim, :] *= 1 + I / (1 - I) * W
    return np.real(sfft.ifftn(K, workers=1)).astype(np.float32)


def fix_ghosting(x: np.ndarray, protocol: Protocol = PROTOCOL_V2) -> tuple[np.ndarray, FixReport]:
    try:
        f = ghosting_tool(x, protocol)
        per = f.evidence.get("period_lines")
        ev = {"present_before": f.present, "value_before": f.value, "period_lines": per}
        if f.present is not True:  # no ghost, or the tool cannot measure: its own reason
            return unchanged(x, TOOL, VERSION, ARTIFACT, f.why, ev)
        if per is None:
            return unchanged(x, TOOL, VERSION, ARTIFACT, f"{f.why}; no period, so no planes to undo", ev)
        n = int(round(per))
        if abs(per - n) > PERIOD_TOL:
            return unchanged(x, TOOL, VERSION, ARTIFACT, f"{f.why}; the period {per:.2f} lines is not a whole number "
                             f"(± {PERIOD_TOL}): not every n-th plane dimmed, the model this fix undoes", ev)
        return _fix(x, n, f.why, ev, protocol)
    except Exception as e:  # no crash inside an agent loop (D68): the error becomes the reason
        return unchanged(x, TOOL, VERSION, ARTIFACT, f"fix error: {type(e).__name__}: {e}")


def _fix(x, n: int, why: str, ev: dict, protocol: Protocol) -> tuple[np.ndarray, FixReport]:
    ap = protocol.array_axes.index(protocol.phase_slow)
    name = f"{protocol.phase_slow[0]}–{protocol.phase_slow[1]}"
    a = np.moveaxis(np.asarray(x).astype(np.float32, copy=False), ap, AP)
    head = head_mask(a)
    air = (ndi.distance_transform_edt(~head) > AIR_OUT) & (a != 0)
    if air.sum() < AIR_MIN:
        return unchanged(x, TOOL, VERSION, ARTIFACT, f"{why}; the air holds {int(air.sum())} voxels that are not exact "
                         f"zeros (need ≥ {AIR_MIN}): the ghost's strength is read in the air, and exact zeros are not "
                         f"a reading", ev)
    I, sig = strength(a, n, head, air)
    y, _ = _er(a, kgain(a.shape[AP], n, I), ndi.binary_dilation(head, iterations=SUPPORT), ITERS_FINAL)
    out = np.where(a == 0, 0, np.abs(addback(y, n, I))).astype(np.float32)
    out = np.ascontiguousarray(np.moveaxis(out, AP, ap))
    g, m = ghosting_tool(out, protocol), motion_tool(out, protocol)
    after = f"{WORD[g.present]} ({g.value:.4f})" if g.value is not None else f"cannot tell ({g.why})"
    what = (f"every {n}{ordinal(n)} {name} plane: the missing signal put back (I {I:.3f}, read in the air), sign put "
            f"back by error reduction; ghosting tool after: {after}")
    ev |= {"n": n, "planes": f"every {n} from the centre", "I": I, "sigma_air": sig, "air_voxels": int(air.sum()),
           "ghost_after": {"present": g.present, "value": g.value, "why": g.why},
           "motion_after": {"present": m.present, "value": m.value, "why": m.why}, "method": METHOD}
    xs = np.asarray(x).astype(np.float32, copy=False)
    return out, FixReport(TOOL, VERSION, ARTIFACT, True, what, why, head_change(out, xs, head_mask(xs)), ev)
