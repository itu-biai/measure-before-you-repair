"""Stage 2: one clean canvas slice + recipe + seed → corrupted magnitude slice + replayable label.

Physical order (D08): bias → motion → ghosting → spike → Rician noise + magnitude. The slice is a TorchIO
image (1, 1, AP, LR) with a centred affine (D31), so PE = L–R = the last axis (D05, D09). Motion, ghosting
and spike keep the complex inverse FFT (D35); the magnitude is taken once, at the end (D10).
Every knob comes from one numpy Generator per case and is stored under TorchIO's argument names (D36).
Levels (D59, D64): mild / moderate / borderline, all below the clinical rescan line. Motion, ghosting and spike share
one damage scale, the brain RMS change the artifact alone does to the clean slice (D41, D60). Two or more of them in one
case share one damage budget, so their joint damage stays in one band (D65); their own shares can be "faint".
"""
import numpy as np
import SimpleITK as sitk
import torch
import torchio as tio
from torchio.transforms.intensity_transform import IntensityTransform

ORDER = ("bias", "motion", "ghosting", "spike", "noise")
LEVELS = ("mild", "moderate", "borderline")  # "borderline" was "severe" until 19 Oct (D64)
N_BIAS_COEFFICIENTS = 20  # order 3 in 3-D; on a (1, AP, LR) slice only c0–c9 act


def centred_affine(spatial_shape) -> np.ndarray:
    """Identity spacing, world origin at the grid centre: TorchIO's Motion pivots about world (0, 0, 0) (D31)."""
    a = np.eye(4)
    a[:3, 3] = -(np.asarray(spatial_shape, float) - 1) / 2
    return a


def as_image(x: np.ndarray) -> tio.ScalarImage:
    """(AP, LR) slice, real or complex → ScalarImage (1, 1, AP, LR) with a centred affine."""
    x = np.asarray(x)
    x = x.astype(np.complex64 if np.iscomplexobj(x) else np.float32)
    t = torch.from_numpy(np.ascontiguousarray(x))[None, None]
    return tio.ScalarImage(tensor=t, affine=centred_affine(t.shape[1:]))


# ---------------------------------------------------------------- complex versions of the TorchIO 1.2.1 transforms
# Each add_artifact is TorchIO's own body (random_motion.py, random_ghosting.py, random_spike.py) without the final
# `.real.float()`; tests check that the real part equals TorchIO's output.
class ComplexMotion(tio.Motion):
    def add_artifact(self, image, transforms, times, interpolation):
        images = self.resample_images(image, transforms, interpolation)
        spectra = [self.fourier_transform(torch.from_numpy(sitk.GetArrayFromImage(im).transpose())) for im in images]
        self.sort_spectra(spectra, times)
        result_spectrum = torch.empty_like(spectra[0])
        last_index = result_spectrum.shape[2]
        indices = [int(v) for v in (last_index * times).astype(int).tolist()] + [last_index]
        ini = 0
        for spectrum, fin in zip(spectra, indices, strict=True):
            result_spectrum[..., ini:fin] = spectrum[..., ini:fin]
            ini = fin
        return self.inv_fourier_transform(result_spectrum).to(torch.complex64)


class ComplexGhosting(tio.Ghosting):
    def add_artifact(self, tensor, num_ghosts, axis, intensity, restore_center):
        if not num_ghosts or not intensity:
            return tensor
        spectrum = self.fourier_transform(tensor)
        planes = self._get_planes_to_modify(spectrum, axis, num_ghosts)
        tensor_restore, slices = self._get_slices_to_restore(spectrum, axis, restore_center)
        tensor_restore = tensor_restore.clone()
        planes *= 1 - intensity
        spectrum[slices] = tensor_restore
        return self.inv_fourier_transform(spectrum).to(torch.complex64)


class ComplexSpike(tio.Spike):
    def add_artifact(self, tensor, spikes_positions, intensity_factor):
        if intensity_factor == 0 or len(spikes_positions) == 0:
            return tensor
        spectrum = self.fourier_transform(tensor)
        shape = np.array(spectrum.shape)
        mid_shape = shape // 2
        indices = np.floor(spikes_positions * shape).astype(int)
        for index in indices:
            i, j, k = mid_shape + (index - mid_shape)
            artifact = spectrum.cpu().abs().max() * intensity_factor
            if self.invert_transform:
                spectrum[i, j, k] -= artifact
            else:
                spectrum[i, j, k] += artifact
        return self.inv_fourier_transform(spectrum).to(torch.complex64)


class RicianNoise(IntensityTransform):
    """|x + σ(n₁ + i·n₂)| with white n ~ N(0, 1) from np.random.default_rng(seed) (D02, D27). σ = 0 → |x|."""

    def __init__(self, sigma: float, seed: int, **kwargs):
        super().__init__(**kwargs)
        self.sigma, self.seed = float(sigma), int(seed)
        self.args_names = ["sigma", "seed"]

    def apply_transform(self, subject):
        for image in self.get_images(subject):
            x = image.data.numpy()
            if self.sigma > 0:
                rng = np.random.default_rng(self.seed)
                x = x + self.sigma * (rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape))
            image.set_data(torch.from_numpy(np.abs(x).astype(np.float32)))
        return subject


def build(name: str, p: dict) -> tio.Transform:
    """TorchIO-named parameters (as stored in a label) → transform. Our knob keys are ignored."""
    if name == "bias":
        return tio.BiasField(coefficients=[float(c) for c in p["coefficients"]], order=int(p["order"]))
    if name == "motion":
        return ComplexMotion(degrees=np.asarray(p["degrees"], float), translation=np.asarray(p["translation"], float),
                             times=np.asarray(p["times"], float), image_interpolation=p["image_interpolation"])
    if name == "ghosting":
        return ComplexGhosting(num_ghosts=int(p["num_ghosts"]), axis=int(p["axis"]), intensity=float(p["intensity"]),
                               restore=p["restore"])
    if name == "spike":
        return ComplexSpike(spikes_positions=np.asarray(p["spikes_positions"], float), intensity=float(p["intensity"]))
    if name == "noise":
        return RicianNoise(sigma=p["sigma"], seed=p["seed"])
    raise ValueError(f"unknown artifact {name!r}")


def _apply(x: np.ndarray, name: str, p: dict) -> np.ndarray:
    return build(name, p)(as_image(x)).data[0, 0].numpy()


# ---------------------------------------------------------------- severity knobs → TorchIO arguments
def _log_field(coefficients, shape) -> np.ndarray:
    probe = torch.zeros((1, 1) + tuple(shape))
    return np.log(tio.BiasField.generate_bias_field(probe, 3, list(coefficients))[0].astype(np.float64))


def bias_coefficients(shape_c: np.ndarray, brain: np.ndarray, target: float) -> list[float]:
    """Scale a random field shape so that max/min − 1 inside `brain` equals `target`; mean log-gain 0 in the brain."""
    r = np.asarray(shape_c, float).copy()
    r[0] = 0.0
    log_f = _log_field(r, brain.shape)[brain]
    span = float(log_f.max() - log_f.min())
    sc = np.log1p(target) / span if span > 0 else 0.0
    c = sc * r
    c[0] = -sc * float(log_f.mean())
    return c.tolist()


def sample_bias(rng: np.random.Generator, band, brain: np.ndarray, order: int = 3) -> dict:
    shape_c = np.zeros(N_BIAS_COEFFICIENTS)
    shape_c[1:10] = rng.uniform(-1, 1, 9)
    t = float(rng.uniform(*band))
    return {"coefficients": bias_coefficients(shape_c, brain, t), "order": order, "target_maxmin": t}


def _draw_motion(rng, moves, degrees, translation, interpolation) -> dict:
    """In-plane only (D31): degrees[:, 0] = rotation about the slice normal, translation[:, 1:] = (AP, LR) mm.
    Times follow RandomMotion's model (random_motion.py: k/(n+1) ± 0.3/(n+1))."""
    n = int(rng.integers(moves[0], moves[1] + 1))
    deg = np.zeros((n, 3))
    deg[:, 0] = rng.choice([-1.0, 1.0], n) * rng.uniform(degrees[0], degrees[1], n)
    tr = np.zeros((n, 3))
    tr[:, 1:] = rng.choice([-1.0, 1.0], (n, 2)) * rng.uniform(translation[0], translation[1], (n, 2))
    step = 1 / (n + 1)
    times = np.arange(1, n + 1) * step + rng.uniform(-0.3 * step, 0.3 * step, n)
    return {"degrees": deg.tolist(), "translation": tr.tolist(), "times": times.tolist(),
            "image_interpolation": interpolation}


def damage(ref: np.ndarray, brain: np.ndarray, name: str, params: dict) -> float:
    """Brain RMS of |artifact(ref)| − ref on the clean reference: the shared damage scale (D41, D59)."""
    out = np.abs(_apply(ref, name, params))
    return float(np.sqrt(np.mean((out - ref)[brain] ** 2)))


def motion_rms(ref: np.ndarray, brain: np.ndarray, params: dict) -> float:
    return damage(ref, brain, "motion", params)


TARGET_RTOL = 1e-3  # the realised damage is within 0.1 % of the drawn target (D60)


def _hit_target(ref, brain, name: str, make, k: float, target: float, max_steps: int = 8) -> tuple[dict, float]:
    """Damage grows almost linearly with the knob (ghost ratio g, stripe amplitude a), so k ← k · target / damage(k)
    converges in 2–4 steps. `make(k)` returns the transform's parameters for knob k on the clean slice."""
    for _ in range(max_steps):
        p = make(k)
        e = damage(ref, brain, name, p)
        if abs(e - target) <= TARGET_RTOL * target:
            return p, e
        if e <= 0:
            raise ValueError(f"{name}: no damage inside the brain mask, cannot set its level")
        k *= target / e
    raise RuntimeError(f"{name}: damage {e:.4f} did not reach the target {target:.4f} in {max_steps} steps")


def sample_motion(rng: np.random.Generator, level: str, mcfg: dict, ref: np.ndarray, brain: np.ndarray) -> dict:
    """Draw from the level's prior until the brain RMS error lands in the level's band (D32, D41)."""
    i = LEVELS.index(level)
    keys = ("moves", "degrees", "translation_mm")
    band = mcfg.get("rms_band")
    priors = (("level", [mcfg[k][i] for k in keys]),
              ("union", [(min(b[0] for b in mcfg[k]), max(b[1] for b in mcfg[k])) for k in keys]))
    for prior_name, prior in priors:
        for t in range(int(mcfg["max_tries"])):
            p = _draw_motion(rng, *prior, mcfg["image_interpolation"])
            e = motion_rms(ref, brain, p)
            if band is None or band[i][0] <= e < band[i][1]:
                return p | {"motion_rms": e, "prior": prior_name, "tries": t + 1}
    raise RuntimeError(f"motion {level}: no draw with brain RMS in {band[i]} after {2 * mcfg['max_tries']} tries")


GHOST_PROBE, SPIKE_PROBE = 0.1, 0.05  # first knob values tried; damage per unit is ≈ 0.4–0.5 (g) and 0.7–1 (a)


def ghosting_params(n: int, g: float, restore) -> dict:
    """n ghosts of ratio g each: TorchIO's intensity = n · g (D33)."""
    return {"num_ghosts": n, "axis": 2, "intensity": n * g, "restore": restore, "ghost_ratio": g}


def sample_ghosting(rng: np.random.Generator, band, gcfg: dict, ref: np.ndarray, brain: np.ndarray) -> dict:
    """The level is the brain damage on the clean slice (D60): draw n and a target from the band, then find the
    ghost ratio g that reaches it on this slice (the same g does different damage at different sites)."""
    n = int(rng.integers(gcfg["num_ghosts"][0], gcfg["num_ghosts"][1] + 1))
    t = float(rng.uniform(*band))
    p, e = _hit_target(ref, brain, "ghosting", lambda g: ghosting_params(n, g, gcfg["restore"]), GHOST_PROBE, t)
    if p["intensity"] >= 1:
        raise RuntimeError(f"ghosting: target {t:.4f} needs n·g = {p['intensity']:.2f} ≥ 1")
    return p | {"target_rms": t, "ghosting_rms": e}


def spike_positions(rng: np.random.Generator, scfg: dict, shape, n_spikes: int | None = None) -> list:
    """n distinct spikes at whole-sample k-space offsets from the centre, as TorchIO positions in [0, 1)."""
    shape = np.array((1,) + tuple(shape))
    if n_spikes is None:
        n_spikes = int(rng.integers(scfg["n_spikes"][0], scfg["n_spikes"][1] + 1))
    offsets: list[tuple[int, int, int]] = []
    while len(offsets) < n_spikes:
        r, th = rng.uniform(*scfg["k_offset"]), rng.uniform(0, 2 * np.pi)
        o = (0, int(round(r * np.sin(th))), int(round(r * np.cos(th))))
        if o not in offsets:
            offsets.append(o)
    return ((shape // 2 + np.array(offsets) + 0.5) / shape).tolist()


def spike_params(positions, a: float, x: np.ndarray) -> dict:
    """Intensity chosen so each stripe has amplitude a (WM units) on the input x (D34)."""
    k_max = float(np.abs(np.fft.fftn(x)).max())  # TorchIO's max|K| (fftshift does not change it)
    return {"spikes_positions": positions, "intensity": a * x.size / k_max, "stripe_amplitude": a}


def sample_spike(rng: np.random.Generator, band, scfg: dict, x: np.ndarray, ref: np.ndarray,
                 brain: np.ndarray) -> dict:
    """The level is the brain damage the spikes alone do to the clean slice (D60): draw the spikes and a target, find
    the stripe amplitude a that reaches it on `ref`, then apply that same a to the current input x."""
    positions = spike_positions(rng, scfg, x.shape)
    t = float(rng.uniform(*band))
    p, e = _hit_target(ref, brain, "spike", lambda a: spike_params(positions, a, ref), SPIKE_PROBE, t)
    return spike_params(positions, p["stripe_amplitude"], x) | {"target_rms": t, "spike_rms": e}


def share_level(d: float, bands) -> str:
    """Name of the band a damage value falls in; below the mild band it is "faint" (D65)."""
    if d < bands[0][0]:
        return "faint"
    return next((lv for lv, (_, hi) in zip(LEVELS, bands) if d < hi), LEVELS[-1])


def sample_budget(rng: np.random.Generator, band, pool: list[str], floor: float) -> dict:
    """D65: the motion-like artifacts of one case share one damage budget. Draw the total T from the band (at least
    floor·√k, so every share can reach the floor), then split it like independent errors add:
    d_i² = floor² + w_i·(T² − k·floor²), w ~ Dirichlet(1), so every share ≥ floor and Σ d_i² = T²."""
    k = len(pool)
    lo = max(band[0], floor * np.sqrt(k))
    if lo >= band[1]:
        raise ValueError(f"budget {list(band)} cannot hold {k} shares of at least {floor:.4f} each (D65)")
    t = float(rng.uniform(lo, band[1]))
    w = rng.dirichlet(np.ones(k))
    d = np.sqrt(floor**2 + w * (t**2 - k * floor**2))
    return {"pool": list(pool), "target": t, "floor": float(floor), "shares": {a: float(x) for a, x in zip(pool, d)}}


MIN_ADDED_OVER_NATIVE = 0.5  # added σ ≥ half the native σ, so the noise always gets visibly worse (D49)


def sample_noise(rng: np.random.Generator, band, native_sigma: float | None) -> dict:
    """The level is the total σ the image ends up with, native ⊕ added (D45): draw the total from the band, then
    add the Rician σ that reaches it. A scan whose own noise sits near the band's lower end gets a raised lower end
    (D49)."""
    if native_sigma is None or not np.isfinite(native_sigma):
        raise ValueError("noise needs the scan's native σ: its level is the total σ (D45)")
    native = float(native_sigma)
    lo = max(band[0], native * float(np.hypot(1, MIN_ADDED_OVER_NATIVE)))
    if lo >= band[1]:
        raise ValueError(f"native σ {native:.4f} is too high for the noise band {list(band)} (D49)")
    total = float(rng.uniform(lo, band[1]))
    s = float(np.sqrt(total**2 - native**2))
    return {"sigma": s, "seed": int(rng.integers(2**31 - 1)), "sigma_added": s, "sigma_native": native,
            "sigma_total": total}


# ---------------------------------------------------------------- the chain
def corrupt(ref: np.ndarray, brain: np.ndarray, recipe: dict[str, str], seed: int, cfg2: dict,
            native_sigma: float | None = None) -> tuple[np.ndarray, dict]:
    """Apply the recipe's artifacts in physical order; knobs are sampled stepwise on the current image."""
    unknown = set(recipe) - set(ORDER)
    if unknown:
        raise ValueError(f"unknown artifacts {sorted(unknown)}")
    rng = np.random.default_rng(seed)
    ref = np.asarray(ref, np.float32)
    names = [n for n in ORDER if n in recipe]
    bcfg = cfg2.get("budget")
    pool = [n for n in names if bcfg and n in bcfg["pool"]]
    budget = None
    if len(pool) >= 2:  # D65: one damage budget for the motion-like artifacts of this case
        levels = {recipe[n] for n in pool}
        if len(levels) != 1:
            raise ValueError(f"motion-like artifacts {pool} share one budget and need one level, got {sorted(levels)}")
        blevel = levels.pop()
        own = float(native_sigma) if native_sigma is not None and np.isfinite(native_sigma) else 0.0
        band = cfg2["damage_band"][LEVELS.index(blevel)]
        budget = {"level": blevel} | sample_budget(rng, band, pool, max(own, bcfg["min_share"]))
    x, params = ref, {}
    for name in names:
        level, c = recipe[name], cfg2[name]
        i = LEVELS.index(level)
        share = budget["shares"].get(name) if budget else None
        if name == "bias":
            p = sample_bias(rng, c["target_maxmin"][i], brain, c["order"])
        elif name == "motion":
            if share is not None:
                c = c | {"rms_band": [[share * (1 - bcfg["motion_tol"]), share * (1 + bcfg["motion_tol"])]] * len(LEVELS),
                         "max_tries": bcfg["motion_max_tries"]}
            p = sample_motion(rng, level, c, ref, brain)
        elif name == "ghosting":
            p = sample_ghosting(rng, [share, share] if share is not None else c["rms_band"][i], c, ref, brain)
        elif name == "spike":
            p = sample_spike(rng, [share, share] if share is not None else c["rms_band"][i], c, x, ref, brain)
        else:
            p = sample_noise(rng, c["sigma_total"][i], native_sigma)
        x = _apply(x, name, p)
        params[name] = p
    severity = {n: recipe[n] for n in names}
    if budget:
        y = ref  # the pooled artifacts alone on the clean slice: the case's motion-like damage
        for n in pool:
            y = _apply(y, n, params[n])
        e = float(np.sqrt(np.mean((np.abs(y) - ref)[brain] ** 2)))
        if not band[0] <= e <= band[1]:
            raise RuntimeError(f"budget {blevel}: motion-like damage {e:.4f} outside {list(band)} (D65); redraw")
        budget["damage"] = e
        own_damage = {"motion": "motion_rms", "ghosting": "ghosting_rms", "spike": "spike_rms"}
        severity |= {n: share_level(params[n][own_damage[n]], cfg2["damage_band"]) for n in pool}
    label = {"artifacts": names, "severity": severity, "params": params,
             "canvas": [int(ref.shape[1]), int(ref.shape[0])], "pe_axis": "LR", "seed": int(seed),
             "torchio": tio.__version__}
    if budget:
        label["budget"] = budget
    return np.abs(x).astype(np.float32), label


def replay(ref: np.ndarray, label: dict) -> np.ndarray:
    """Re-apply a label's stored parameters; bit-for-bit equal to corrupt()'s output."""
    x = np.asarray(ref, np.float32)
    for name in label["artifacts"]:
        x = _apply(x, name, label["params"][name])
    return np.abs(x).astype(np.float32)
