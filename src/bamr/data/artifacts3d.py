"""Stage 2 in 3-D (blind-mri-v2, D73–D86): one clean boxed volume + recipe + seed → corrupted magnitude volume + label.

Arrays are RAS (LR, AP, SI) at 1 mm in the 192 × 240 × 240 box (D75). TorchIO gets the volume as (1, LR, SI, AP): A–P,
the slow phase-encode axis of a sagittal 3-D MPRAGE (D74), is the last axis, so Motion splits k-space in time along A–P
and Ghosting dims every n-th A–P plane (axis 2). The transform classes, the Rician noise and the physical order are v1's
(bamr.data.artifacts, D08, D35); only the geometry is 3-D.

Levels (D83–D86): noise on the clinical noise grades (total σ, unchanged); motion, ghosting and spike each have their own
whole-brain damage band, set by the clinical calibration so each level looks like real MR-ART motion of that grade
(entropy, D84, D85). Their shared budget (D65) works in grade-matched units: an artifact's damage × its `equiv` factor
= the motion damage that looks the same. Every level's target is drawn uniformly from its band and hit exactly: ghosting
and spike by scaling their knob (D60), motion by scaling the amplitude of a movement drawn from the gentle prior (D88).
"""
import numpy as np
import torch
import torchio as tio

from bamr.data.artifacts import (LEVELS, ORDER, RicianNoise, build, centred_affine, ghosting_params, sample_budget,
                                 sample_noise, share_level, spike_params)

BOX = (192, 240, 240)  # (LR, AP, SI) at 1 mm; N/2 = 96, 120, 120 are multiples of 2, 3 and 4 (D30, D75)
N_BIAS_COEFFICIENTS = 20  # order-3 polynomial in 3-D: all 20 terms act on a volume (D78)
AXES = {"readout": "SI", "phase_slow": "AP", "phase_fast": "LR"}  # D74


# ---------------------------------------------------------------- the box (D75)
def box_origin(mask: np.ndarray, box=BOX) -> tuple[int, int, int]:
    """Start index of a box centred on the brain's 3-D bounding box (D39 extended to S–I). May be negative."""
    out = []
    for ax, n in enumerate(box):
        idx = np.flatnonzero(mask.any(axis=tuple(a for a in range(3) if a != ax)))
        out.append(int(round((idx[0] + idx[-1] + 1) / 2 - n / 2)))
    return out[0], out[1], out[2]


def cut_box(vol: np.ndarray, origin, box=BOX) -> np.ndarray:
    """The box at `origin` cut from a RAS volume; zeros where the box runs past the volume."""
    out = np.zeros(box, vol.dtype)
    src, dst = [], []
    for o, n, size in zip(origin, box, vol.shape):
        lo, hi = max(o, 0), min(o + n, size)
        src.append(slice(lo, hi))
        dst.append(slice(lo - o, hi - o))
    out[tuple(dst)] = vol[tuple(src)]
    return out


# ---------------------------------------------------------------- TorchIO view: A–P last
def to_t(x: np.ndarray) -> np.ndarray:
    """(LR, AP, SI) → (LR, SI, AP)."""
    return np.ascontiguousarray(np.transpose(x, (0, 2, 1)))


def from_t(x: np.ndarray) -> np.ndarray:
    """(LR, SI, AP) → (LR, AP, SI)."""
    return np.transpose(x, (0, 2, 1))


def as_image3(x: np.ndarray) -> tio.ScalarImage:
    """(LR, AP, SI) volume, real or complex → ScalarImage (1, LR, SI, AP) with a centred 1 mm affine (D31)."""
    t = to_t(np.asarray(x))
    t = t.astype(np.complex64 if np.iscomplexobj(t) else np.float32)
    return tio.ScalarImage(tensor=torch.from_numpy(t)[None], affine=centred_affine(t.shape))


def apply3(x: np.ndarray, name: str, p: dict) -> np.ndarray:
    return from_t(build(name, p)(as_image3(x)).data[0].numpy())


def damage3(ref: np.ndarray, brain: np.ndarray, name: str, p: dict) -> float:
    """Whole-brain RMS of |artifact(ref)| − ref on the clean reference, WM = 1 (D41, D59 in 3-D)."""
    out = np.abs(apply3(ref, name, p))
    return float(np.sqrt(np.mean((out - ref)[brain] ** 2)))


# ---------------------------------------------------------------- bias (D78, D86)
def _log_field3(coefficients, shape_t) -> np.ndarray:
    probe = torch.zeros((1,) + tuple(shape_t))
    return np.log(tio.BiasField.generate_bias_field(probe, 3, list(coefficients)).astype(np.float64))


def sample_bias3(rng: np.random.Generator, band, brain: np.ndarray, order: int = 3) -> dict:
    """A random order-3 field shape (all 20 terms), scaled so max/min − 1 over the whole brain equals a target drawn
    from the band; mean log-gain 0 in the brain, so the WM = 1 scale is kept (as v1)."""
    bt = to_t(brain)
    r = np.zeros(N_BIAS_COEFFICIENTS)
    r[1:] = rng.uniform(-1, 1, N_BIAS_COEFFICIENTS - 1)
    log_f = _log_field3(r, bt.shape)[bt]
    span = float(log_f.max() - log_f.min())
    t = float(rng.uniform(*band))
    sc = np.log1p(t) / span
    c = sc * r
    c[0] = -sc * float(log_f.mean())
    return {"coefficients": c.tolist(), "order": order, "target_maxmin": t}


# ---------------------------------------------------------------- motion (D76, D88)
def draw_motion3(rng: np.random.Generator, mcfg: dict) -> dict:
    """A movement from v1's gentle prior on all six axes (D62, D76): 1–2 moves, each rotation ±0–2°, each shift
    ±0–2 mm; times as RandomMotion's model (k/(n+1) ± 0.3/(n+1))."""
    lo, hi = mcfg["moves"]
    n = int(rng.integers(lo, hi + 1))
    deg = rng.choice([-1.0, 1.0], (n, 3)) * rng.uniform(*mcfg["degrees"], (n, 3))
    tr = rng.choice([-1.0, 1.0], (n, 3)) * rng.uniform(*mcfg["translation_mm"], (n, 3))
    step = 1 / (n + 1)
    times = np.arange(1, n + 1) * step + rng.uniform(-0.3 * step, 0.3 * step, n)
    return {"degrees": deg.tolist(), "translation": tr.tolist(), "times": times.tolist(),
            "image_interpolation": mcfg["image_interpolation"]}


def scale_motion(shape: dict, s: float) -> dict:
    return shape | {"degrees": (s * np.asarray(shape["degrees"])).tolist(),
                    "translation": (s * np.asarray(shape["translation"])).tolist(), "amplitude_scale": float(s)}


def sample_motion3(rng: np.random.Generator, band, mcfg: dict, ref: np.ndarray, brain: np.ndarray) -> dict:
    """Draw a target from the band and a movement from the prior, then scale the movement's amplitude (all rotations
    and shifts by one factor s ≤ 1, so it stays inside the prior) until its whole-brain damage is within `rtol` of the
    target. Damage grows as a power of s (about s^0.7 on IXI volumes), so s is updated on a log–log secant through
    the last two tries and converges in 2–4 steps. A movement that cannot reach the target with s ≤ 1 is replaced
    by a new draw."""
    t = float(rng.uniform(*band))
    rtol, steps = float(mcfg["rtol"]), int(mcfg["max_steps"])
    for draw in range(int(mcfg["max_draws"])):
        shape = draw_motion3(rng, mcfg)
        s, power, last = min(1.0, (t / MOTION_FULL) ** (1 / MOTION_POWER)), MOTION_POWER, None
        for step in range(steps):
            p = scale_motion(shape, s)
            e = damage3(ref, brain, "motion", p)
            if abs(e - t) <= rtol * t:
                return p | {"target_rms": t, "motion_rms": e, "draws": draw + 1, "steps": step + 1}
            if e <= 0 or (s >= 1.0 and e < t):
                break  # this movement is too gentle even at full amplitude
            if last is not None and last[0] != s and last[1] > 0 and last[1] != e:
                power = float(np.clip(np.log(e / last[1]) / np.log(s / last[0]), 0.3, 2.0))
            last = (s, e)
            s = min(1.0, s * (t / e) ** (1 / power))
    raise RuntimeError(f"motion: no movement reached damage {t:.4f} within {rtol:.0%}")


MOTION_POWER = 0.7  # first guess of d log(damage) / d log(s)
MOTION_FULL = 0.055  # typical whole-brain damage of a prior draw at full amplitude (s = 1): the first guess of s


# ---------------------------------------------------------------- ghosting and spike (D77, D60)
TARGET_RTOL = 1e-3
GHOST_PROBE, SPIKE_PROBE = 0.1, 0.05


def _hit3(ref, brain, name: str, make, k: float, target: float, max_steps: int = 8) -> tuple[dict, float]:
    """Knob k ← k · target / damage(k) until the damage is within 0.1 % of the target (v1's D60 rule, in 3-D)."""
    for _ in range(max_steps):
        p = make(k)
        e = damage3(ref, brain, name, p)
        if abs(e - target) <= TARGET_RTOL * target:
            return p, e
        if e <= 0:
            raise ValueError(f"{name}: no damage inside the brain mask, cannot set its level")
        k *= target / e
    raise RuntimeError(f"{name}: damage {e:.4f} did not reach the target {target:.4f} in {max_steps} steps")


def sample_ghosting3(rng: np.random.Generator, band, gcfg: dict, ref: np.ndarray, brain: np.ndarray) -> dict:
    """n ghosts along A–P (axis 2 of the TorchIO view); ghost ratio g set so the damage hits a target from the band."""
    n = int(rng.integers(gcfg["num_ghosts"][0], gcfg["num_ghosts"][1] + 1))
    t = float(rng.uniform(*band))
    p, e = _hit3(ref, brain, "ghosting", lambda g: ghosting_params(n, g, gcfg["restore"]), GHOST_PROBE, t)
    if p["intensity"] >= 1:
        raise RuntimeError(f"ghosting: target {t:.4f} needs n·g = {p['intensity']:.2f} ≥ 1")
    return p | {"target_rms": t, "ghosting_rms": e}


def spike_positions3(rng: np.random.Generator, scfg: dict, shape_t, n_spikes: int | None = None) -> list:
    """n distinct spikes at whole-sample offsets 10–60 from the k-space centre in a random 3-D direction (D77), as
    TorchIO positions in [0, 1)³ of the (LR, SI, AP) spectrum."""
    s = np.array(shape_t)
    if n_spikes is None:
        n_spikes = int(rng.integers(scfg["n_spikes"][0], scfg["n_spikes"][1] + 1))
    offsets: list[tuple[int, int, int]] = []
    while len(offsets) < n_spikes:
        v = rng.normal(size=3)
        o = tuple(int(x) for x in np.round(rng.uniform(*scfg["k_offset"]) * v / np.linalg.norm(v)))
        if o not in offsets and any(o):
            offsets.append(o)
    return ((s // 2 + np.array(offsets) + 0.5) / s).tolist()


def sample_spike3(rng: np.random.Generator, band, scfg: dict, x: np.ndarray, ref: np.ndarray,
                  brain: np.ndarray) -> dict:
    """Spikes' stripe amplitude a set so their damage on the clean volume hits a target from the band; the same a is
    then applied to the current input x (D34, D60)."""
    positions = spike_positions3(rng, scfg, to_t(ref).shape)
    t = float(rng.uniform(*band))
    p, e = _hit3(ref, brain, "spike", lambda a: spike_params(positions, a, ref), SPIKE_PROBE, t)
    return spike_params(positions, p["stripe_amplitude"], x) | {"target_rms": t, "spike_rms": e}


# ---------------------------------------------------------------- the chain
OWN_DAMAGE = {"motion": "motion_rms", "ghosting": "ghosting_rms", "spike": "spike_rms"}


def corrupt3(ref: np.ndarray, brain: np.ndarray, recipe: dict[str, str], seed: int, cfg2: dict,
             native_sigma: float | None = None) -> tuple[np.ndarray, dict]:
    """Apply the recipe's artifacts in physical order (D08) on the boxed volume; returns (magnitude, label)."""
    unknown = set(recipe) - set(ORDER)
    if unknown:
        raise ValueError(f"unknown artifacts {sorted(unknown)}")
    rng = np.random.default_rng(seed)
    ref = np.asarray(ref, np.float32)
    names = [n for n in ORDER if n in recipe]
    bcfg = cfg2["budget"]
    pool = [n for n in names if n in bcfg["pool"]]
    budget = None
    if len(pool) >= 2:  # D65 in grade-matched units (D85)
        levels = {recipe[n] for n in pool}
        if len(levels) != 1:
            raise ValueError(f"motion-like artifacts {pool} share one budget and need one level, got {sorted(levels)}")
        blevel = levels.pop()
        band = cfg2["damage_band"][LEVELS.index(blevel)]
        budget = {"level": blevel, "units": "motion-equivalent whole-brain damage"} | \
            sample_budget(rng, band, pool, bcfg["min_share"])
        budget["own_targets"] = {a: u / bcfg["equiv"][a] for a, u in budget["shares"].items()}
    x, params = ref, {}
    for name in names:
        level, c = recipe[name], cfg2[name]
        i = LEVELS.index(level)
        own = budget["own_targets"].get(name) if budget else None
        if name == "bias":
            p = sample_bias3(rng, c["target_maxmin"][i], brain, c["order"])
        elif name == "motion":
            mband = [own, own] if own is not None else c["rms_band"][i]
            p = sample_motion3(rng, mband, c, ref, brain)
        elif name == "ghosting":
            p = sample_ghosting3(rng, [own, own] if own is not None else c["rms_band"][i], c, ref, brain)
        elif name == "spike":
            p = sample_spike3(rng, [own, own] if own is not None else c["rms_band"][i], c, x, ref, brain)
        else:
            p = sample_noise(rng, c["sigma_total"][i], native_sigma)
        x = apply3(x, name, p)
        params[name] = p
    severity = {n: recipe[n] for n in names}
    if budget:
        eq = {n: params[n][OWN_DAMAGE[n]] * bcfg["equiv"][n] for n in pool}
        total = float(np.sqrt(sum(v ** 2 for v in eq.values())))
        if not band[0] <= total <= band[1]:
            raise RuntimeError(f"budget {blevel}: grade-matched total {total:.4f} outside {list(band)} (D65); redraw")
        y = ref  # the pooled artifacts alone on the clean volume: their joint physical damage, for the record
        for n in pool:
            y = apply3(y, n, params[n])
        budget |= {"equivalent": eq, "total": total,
                   "joint_rms": float(np.sqrt(np.mean((np.abs(y) - ref)[brain] ** 2)))}
        severity |= {n: share_level(eq[n], cfg2["damage_band"]) for n in pool}
    label = {"artifacts": names, "severity": severity, "params": params, "box": list(ref.shape),
             "axes": dict(AXES), "seed": int(seed), "torchio": tio.__version__}
    if budget:
        label["budget"] = budget
    return np.abs(x).astype(np.float32), label


def replay3(ref: np.ndarray, label: dict) -> np.ndarray:
    """Re-apply a label's stored parameters; bit-for-bit equal to corrupt3()'s output."""
    x = np.asarray(ref, np.float32)
    for name in label["artifacts"]:
        x = apply3(x, name, label["params"][name])
    return np.abs(x).astype(np.float32)


__all__ = ["BOX", "AXES", "box_origin", "cut_box", "to_t", "from_t", "as_image3", "apply3", "damage3", "sample_bias3",
           "draw_motion3", "scale_motion", "sample_motion3", "sample_ghosting3", "spike_positions3", "sample_spike3",
           "corrupt3", "replay3", "RicianNoise"]
