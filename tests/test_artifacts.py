import copy
import itertools
import json

import numpy as np
import pytest
import SimpleITK as sitk
import torchio as tio
from torchio.data.io import nib_to_sitk

from bamr.config import load_config
from bamr.data import artifacts as A

AP, LR = 240, 192
CFG2 = load_config()["stage2"]


def phantom():
    """Head-like slice on the canvas: scalp 0.6, GM 0.8, WM 1, a ventricle and a bar; brain = GM + WM."""
    y, x = np.mgrid[:AP, :LR].astype(float)
    cy, cx = (AP - 1) / 2, (LR - 1) / 2
    r = np.sqrt(((y - cy) / 95) ** 2 + ((x - cx) / 75) ** 2)
    img = np.zeros((AP, LR), np.float32)
    img[r < 1.0] = 0.6
    brain = r < 0.88
    img[brain] = 0.8
    img[r < 0.75] = 1.0
    img[np.sqrt(((y - cy) / 25) ** 2 + ((x - cx - 8) / 6) ** 2) < 1] = 0.3
    img[(np.abs(y - cy - 40) < 3) & (np.abs(x - cx) < 30)] = 0.5
    return img, brain


def disk(radius=20.0, value=1.0):
    y, x = np.mgrid[:AP, :LR].astype(float)
    return np.where(np.hypot(y - (AP - 1) / 2, x - (LR - 1) / 2) < radius, value, 0.0).astype(np.float32)


def run(transform, x):
    return transform(A.as_image(x)).data[0, 0].numpy()


def corrupt_retry(ref, brain, recipe, seed, cfg2=CFG2, native_sigma=0.02, attempts=10):
    """As the Stage 3 build does (D58): a case whose motion-like budget misses its band is redrawn with the next seed."""
    for k in range(attempts):
        try:
            return A.corrupt(ref, brain, recipe, seed=1000 * seed + k, cfg2=cfg2, native_sigma=native_sigma)
        except RuntimeError:
            continue
    raise RuntimeError(f"no valid draw in {attempts} attempts")


def brain_rms(a, b, m):
    return float(np.sqrt(np.mean((np.abs(a) - b)[m] ** 2)))


MOTION_P = dict(degrees=np.array([[4.0, 0, 0], [-3.0, 0, 0]]), translation=np.array([[0, 3.0, -2.0], [0, -2.0, 4.0]]),
                times=np.array([0.3, 0.6]), image_interpolation="bspline")


# ---------------------------------------------------------------- Task 2: complex transforms + Rician noise
def _poses(t4, affine, p):
    img = nib_to_sitk(t4, affine, force_3d=True)
    mo = tio.Motion(**p)
    Ts = mo.get_rigid_transforms(p["degrees"], p["translation"], img)
    return [sitk.GetArrayFromImage(i).transpose() for i in mo.resample_images(img, Ts, p["image_interpolation"])]


def test_motion_default_layout_is_rigid():
    """D09: with TorchIO's 2-D layout (C, W, H, 1) Motion splits k-space along the length-1 axis -> one rigid pose."""
    ref, brain = phantom()
    t4 = ref[None, :, :, None]
    aff = A.centred_affine(t4.shape[1:])
    p = dict(degrees=np.array([[0, 0, 4.0], [0, 0, -3.0]]), translation=np.array([[3.0, -2.0, 0], [-2.0, 4.0, 0]]),
             times=np.array([0.3, 0.6]), image_interpolation="bspline")
    out = tio.Motion(**p)(tio.ScalarImage(tensor=t4, affine=aff)).data[0, :, :, 0].numpy()
    best = min(brain_rms(out, q[:, :, 0], brain) for q in _poses(t4, aff, p))
    assert best < 1e-5
    ours = run(A.ComplexMotion(**MOTION_P), ref)
    poses = _poses(ref[None, None], A.centred_affine((1, AP, LR)), MOTION_P)
    assert min(brain_rms(ours, q[0], brain) for q in poses) > 0.01


def test_motion_pivot_centred():
    """D31: TorchIO drops the rotation centre, so the pivot is world (0,0,0); a centred affine puts it mid-slice."""
    x = disk(40.0)[None, None]
    p = dict(degrees=np.array([[10.0, 0, 0]]), translation=np.zeros((1, 3)), times=np.array([0.5]),
             image_interpolation="bspline")

    def centroid_shift(aff):
        moved = _poses(x, aff, p)[1][0]
        jj, kk = np.indices(moved.shape)
        c = lambda a: np.array([(a * jj).sum() / a.sum(), (a * kk).sum() / a.sum()])
        return float(np.linalg.norm(c(np.clip(moved, 0, None)) - c(x[0, 0])))

    assert centroid_shift(A.centred_affine((1, AP, LR))) < 1.0
    assert centroid_shift(np.eye(4)) > 10.0


@pytest.mark.parametrize("name", ["motion", "ghosting", "spike"])
def test_real_part_equals_torchio(name):
    ref, _ = phantom()
    pos = np.array([[0.5, (AP // 2 + 17 + 0.5) / AP, (LR // 2 - 23 + 0.5) / LR]])
    pairs = {"motion": (A.ComplexMotion(**MOTION_P), tio.Motion(**MOTION_P)),
             "ghosting": (A.ComplexGhosting(num_ghosts=3, axis=2, intensity=0.3, restore=None),
                          tio.Ghosting(num_ghosts=3, axis=2, intensity=0.3, restore=None)),
             "spike": (A.ComplexSpike(spikes_positions=pos, intensity=0.4), tio.Spike(spikes_positions=pos, intensity=0.4))}
    ours, theirs = pairs[name]
    o, t = run(ours, ref), run(theirs, ref)
    assert np.iscomplexobj(o)
    np.testing.assert_allclose(o.real, t, atol=1e-5)


def test_rician_constant():
    sigma = 0.1
    out = run(A.RicianNoise(sigma=sigma, seed=1), np.ones((AP, LR), np.float32))
    assert out.dtype == np.float32 and out.std() == pytest.approx(sigma, rel=0.05)
    zero = run(A.RicianNoise(sigma=sigma, seed=2), np.zeros((AP, LR), np.float32))
    assert zero.mean() == pytest.approx(sigma * np.sqrt(np.pi / 2), rel=0.03)
    ref, _ = phantom()
    np.testing.assert_array_equal(run(A.RicianNoise(sigma=0.0, seed=3), ref), np.abs(ref))


# ---------------------------------------------------------------- Task 3: knobs, chain, labels
@pytest.mark.parametrize("target", [0.2, 0.4, 0.62])
def test_bias_target_exact(target):
    _, brain = phantom()
    shape_c = np.zeros(20)
    shape_c[1:10] = np.random.default_rng(4).uniform(-1, 1, 9)
    c = A.bias_coefficients(shape_c, brain, target)
    field = run(A.build("bias", {"coefficients": c, "order": 3}), np.ones((AP, LR), np.float32))
    f = field[brain]
    assert f.max() / f.min() - 1 == pytest.approx(target, abs=1e-4)


def test_ghost_ratio_measured():
    """Replica weight = intensity / n (D33). TorchIO's restored k-space centre line adds a separate term that is
    constant along each row (s × row mean, as in L10), so subtract the value at a ghost-free column of the same rows."""
    x = disk(10.0)
    obj = x > 0
    out = run(A.build("ghosting", {"num_ghosts": 3, "axis": 2, "intensity": 3 * 0.1, "restore": None}), x)
    ghost, free = np.roll(obj, LR // 3, axis=1), np.roll(obj, LR // 6, axis=1)
    assert np.abs(out[ghost] - out[free]).mean() / x[obj].mean() == pytest.approx(0.1, abs=0.005)


@pytest.mark.parametrize("ghosted", [False, True])
def test_spike_stripe_amplitude(ghosted):
    ref, brain = phantom()
    x = run(A.ComplexGhosting(num_ghosts=2, axis=2, intensity=0.5, restore=None), ref) if ghosted else ref
    a = 0.15
    pos = A.spike_positions(np.random.default_rng(9), CFG2["spike"], x.shape, n_spikes=1)
    p = A.spike_params(pos, a, x)
    out = run(A.build("spike", p), x)
    d = np.abs(out - x)
    assert p["stripe_amplitude"] == pytest.approx(a)
    np.testing.assert_allclose(d, a, rtol=0.01)


def shrunk_phantom():
    """A second 'subject': the phantom at 80 % size, so the same knob does different damage."""
    ref, brain = phantom()
    y, x = np.mgrid[:AP, :LR].astype(float)
    cy, cx = (AP - 1) / 2, (LR - 1) / 2
    yy, xx = (cy + (y - cy) / 0.8).round().astype(int), (cx + (x - cx) / 0.8).round().astype(int)
    ok = (yy >= 0) & (yy < AP) & (xx >= 0) & (xx < LR)
    out, br = np.zeros_like(ref), np.zeros_like(brain)
    out[ok], br[ok] = ref[yy[ok], xx[ok]], brain[yy[ok], xx[ok]]
    return out, br


KNOB = {"ghosting": "ghost_ratio", "spike": "stripe_amplitude"}


@pytest.mark.parametrize("level", A.LEVELS)
@pytest.mark.parametrize("name", ["ghosting", "spike"])
def test_targeted_damage(name, level):
    """D60: the level is the brain damage the artifact alone does to the clean slice, drawn from the shared band."""
    ref, brain = phantom()
    lo, hi = CFG2[name]["rms_band"][A.LEVELS.index(level)]
    for seed in range(4):
        out, lab = A.corrupt(ref, brain, {name: level}, seed=seed, cfg2=CFG2, native_sigma=0.02)
        p = lab["params"][name]
        assert lo <= p["target_rms"] < hi
        assert p[f"{name}_rms"] == pytest.approx(p["target_rms"], rel=1e-3)
        assert brain_rms(out, ref, brain) == pytest.approx(p[f"{name}_rms"], rel=1e-6)


def test_ghost_ratio_adapts_to_image():
    """D60: same seed → same target on two different slices; the ghost ratio differs, the damage does not."""
    (r1, b1), (r2, b2) = phantom(), shrunk_phantom()
    _, l1 = A.corrupt(r1, b1, {"ghosting": "moderate"}, seed=3, cfg2=CFG2)
    _, l2 = A.corrupt(r2, b2, {"ghosting": "moderate"}, seed=3, cfg2=CFG2)
    p1, p2 = l1["params"]["ghosting"], l2["params"]["ghosting"]
    assert p1["target_rms"] == p2["target_rms"]
    assert p1["ghosting_rms"] == pytest.approx(p2["ghosting_rms"], rel=2e-3)
    assert abs(p1["ghost_ratio"] / p2["ghost_ratio"] - 1) > 0.02


def test_stripe_amplitude_adapts_to_spike_count():
    """D60: one stripe changes the brain by ≈ a/√2; two stripes by more, so the same target needs a smaller a."""
    ref, brain = phantom()
    a = {}
    for n in (1, 2):
        cfg = copy.deepcopy(CFG2)
        cfg["spike"]["n_spikes"] = [n, n]
        _, lab = A.corrupt(ref, brain, {"spike": "moderate"}, seed=3, cfg2=cfg)
        p = lab["params"]["spike"]
        assert len(p["spikes_positions"]) == n and p["spike_rms"] == pytest.approx(p["target_rms"], rel=1e-3)
        a[n] = p["stripe_amplitude"] / p["target_rms"]
    assert a[1] == pytest.approx(np.sqrt(2), rel=0.05)
    assert a[2] < a[1] / 1.15


def test_spike_in_chain_keeps_stripe_amplitude():
    """In a chain the spike's stripe amplitude (set on the clean slice) is kept on the current, already damaged input."""
    ref, brain = phantom()
    _, lab = A.corrupt(ref, brain, {"bias": "moderate", "spike": "mild"}, seed=4, cfg2=CFG2, native_sigma=0.02)
    p = lab["params"]["spike"]
    x = A.replay(ref, {"artifacts": ["bias"], "params": lab["params"]})
    assert p["intensity"] == pytest.approx(p["stripe_amplitude"] * x.size / np.abs(np.fft.fftn(x)).max(), rel=1e-5)


RESCAN = 0.061  # median MR-ART "bad" scan, rated unusable (D59)


def test_bands_recoverable():
    """D59, D61: every level of every artifact stays at or below the rescan line; levels are contiguous."""
    bands = {n: CFG2[n]["rms_band"] for n in ("motion", "ghosting", "spike")} | {"noise": CFG2["noise"]["sigma_total"]}
    for name, b in bands.items():
        assert len(b) == len(A.LEVELS) == 3
        assert b[-1][1] <= RESCAN, name
        assert all(b[i][1] == b[i + 1][0] for i in range(2)) and all(lo < hi for lo, hi in b), name
    assert A.LEVELS == ("mild", "moderate", "borderline")


ZERO = {"bias": {"coefficients": [0.0] * 20, "order": 3},
        "motion": {"degrees": [[0.0, 0, 0]], "translation": [[0.0, 0, 0]], "times": [0.5], "image_interpolation": "bspline"},
        "ghosting": {"num_ghosts": 2, "axis": 2, "intensity": 0.0, "restore": None},
        "spike": {"spikes_positions": [[0.5, 0.6, 0.6]], "intensity": 0.0},
        "noise": {"sigma": 0.0, "seed": 0}}


@pytest.mark.parametrize("name", A.ORDER)
def test_zero_severity_identity(name):
    ref, _ = phantom()
    np.testing.assert_allclose(np.abs(run(A.build(name, ZERO[name]), ref)), ref, atol=1e-5)


@pytest.mark.parametrize("name", A.ORDER)
def test_changes_image(name):
    ref, brain = phantom()
    out, lab = A.corrupt(ref, brain, {name: "mild"}, seed=1, cfg2=CFG2, native_sigma=0.02)
    assert lab["artifacts"] == [name]
    assert brain_rms(out, ref, brain) > 1e-3


def test_same_seed_identical():
    ref, brain = phantom()
    recipe = {n: "moderate" for n in A.ORDER}
    o1, l1 = corrupt_retry(ref, brain, recipe, 5)
    o2, l2 = corrupt_retry(ref, brain, recipe, 5)
    o3, _ = corrupt_retry(ref, brain, recipe, 6)
    np.testing.assert_array_equal(o1, o2)
    assert json.dumps(l1) == json.dumps(l2)
    assert not np.array_equal(o1, o3)


def test_motion_rms_in_band():
    ref, brain = phantom()
    cfg = copy.deepcopy(CFG2)
    cfg["motion"]["rms_band"] = [[0.02, 0.06], [0.06, 0.10], [0.10, 0.20]]
    for seed in range(3):
        _, lab = A.corrupt(ref, brain, {"motion": "mild"}, seed=seed, cfg2=cfg)
        assert 0.02 <= lab["params"]["motion"]["motion_rms"] < 0.06


def test_motion_band_unreachable_raises():
    ref, brain = phantom()
    cfg = copy.deepcopy(CFG2)
    cfg["motion"]["rms_band"] = [[5.0, 6.0], [6.0, 7.0], [7.0, 8.0]]
    cfg["motion"]["max_tries"] = 3
    with pytest.raises(RuntimeError, match="mild"):
        A.corrupt(ref, brain, {"motion": "mild"}, seed=0, cfg2=cfg)


def test_magnitude_nonnegative_all_combinations():
    ref, brain = phantom()
    for r in range(len(A.ORDER) + 1):
        for combo in itertools.combinations(A.ORDER, r):
            out, lab = corrupt_retry(ref, brain, {n: "moderate" for n in reversed(combo)}, r)
            assert out.shape == (AP, LR) and out.dtype == np.float32
            assert np.isfinite(out).all() and out.min() >= 0
            assert lab["artifacts"] == list(combo)


def test_replay_bit_for_bit():
    ref, brain = phantom()
    out, lab = corrupt_retry(ref, brain, {n: "moderate" for n in A.ORDER}, 11)
    assert "budget" in lab
    np.testing.assert_array_equal(A.replay(ref, json.loads(json.dumps(lab))), out)


def test_noise_label_sigmas():
    """D45: the band holds the total σ; the added σ brings native ⊕ added to it."""
    ref, brain = phantom()
    _, lab = A.corrupt(ref, brain, {"noise": "borderline"}, seed=2, cfg2=CFG2, native_sigma=0.02)
    p = lab["params"]["noise"]
    lo, hi = CFG2["noise"]["sigma_total"][2]
    assert lo <= p["sigma_total"] <= hi and p["sigma"] == p["sigma_added"] and p["sigma_native"] == 0.02
    assert np.hypot(0.02, p["sigma_added"]) == pytest.approx(p["sigma_total"])
    assert lab["canvas"] == [192, 240] and lab["pe_axis"] == "LR" and lab["severity"] == {"noise": "borderline"}


@pytest.mark.parametrize("native", [0.013, 0.025, 0.0299])
def test_noise_total_band_every_site(native):
    """Native σ below 3.0 % (D51): every mild draw lands in the total band and adds ≥ native / 2 (D49)."""
    lo, hi = CFG2["noise"]["sigma_total"][0]
    for seed in range(20):
        p = A.sample_noise(np.random.default_rng(seed), (lo, hi), native)
        assert lo <= p["sigma_total"] < hi
        assert p["sigma_added"] >= 0.5 * native - 1e-12


def test_measured_total_sigma():
    """On a flat bright image with native noise, the output's std equals the drawn total σ."""
    rng = np.random.default_rng(0)
    x = (1 + 0.025 * rng.standard_normal((AP, LR))).astype(np.float32)
    p = A.sample_noise(np.random.default_rng(1), CFG2["noise"]["sigma_total"][1], 0.025)
    assert run(A.build("noise", p), x).std() == pytest.approx(p["sigma_total"], rel=0.03)


def test_noise_needs_native_sigma():
    ref, brain = phantom()
    with pytest.raises(ValueError, match="native"):
        A.corrupt(ref, brain, {"noise": "mild"}, seed=0, cfg2=CFG2)
    with pytest.raises(ValueError, match="too high"):
        A.sample_noise(np.random.default_rng(0), (0.030, 0.045), 0.041)
    with pytest.raises(ValueError, match="too high"):  # IXI's noisiest split scans (3.5 %) cannot take mild noise (D51)
        A.sample_noise(np.random.default_rng(0), CFG2["noise"]["sigma_total"][0], 0.035)


# ---------------------------------------------------------------- D65: one budget for the motion-like artifacts
POOL = ["motion", "ghosting", "spike"]


def test_share_level():
    b = CFG2["damage_band"]
    assert [A.share_level(d, b) for d in (0.021, 0.026, 0.037, 0.038, 0.047, 0.05, 0.061)] == \
        ["faint", "mild", "mild", "moderate", "moderate", "borderline", "borderline"]


@pytest.mark.parametrize("k", [2, 3])
def test_sample_budget_floor_and_sum(k):
    rng = np.random.default_rng(0)
    for band in CFG2["damage_band"]:
        for _ in range(100):
            b = A.sample_budget(rng, band, POOL[:k], 0.02)
            d = np.array(list(b["shares"].values()))
            assert d.min() >= 0.02 - 1e-12
            assert np.isclose(np.sqrt((d**2).sum()), b["target"])
            assert max(band[0], 0.02 * np.sqrt(k)) <= b["target"] <= band[1]


def test_budget_infeasible_raises():
    with pytest.raises(ValueError, match="D65"):
        A.sample_budget(np.random.default_rng(0), CFG2["damage_band"][0], POOL, 0.03)  # 0.03·√3 > 0.038


@pytest.mark.parametrize("level", A.LEVELS)
def test_budget_case_in_band(level):
    ref, brain = phantom()
    out, lab = corrupt_retry(ref, brain, {a: level for a in POOL}, 3, native_sigma=0.015)
    b = lab["budget"]
    lo, hi = CFG2["damage_band"][A.LEVELS.index(level)]
    assert b["level"] == level and b["pool"] == POOL and b["floor"] == 0.02
    assert lo <= b["damage"] <= hi <= 0.061
    own = {"motion": "motion_rms", "ghosting": "ghosting_rms", "spike": "spike_rms"}
    for a in POOL:
        e = lab["params"][a][own[a]]
        assert e >= 0.02 * (1 - CFG2["budget"]["motion_tol"]) - 1e-9
        assert lab["severity"][a] == A.share_level(e, CFG2["damage_band"])
    np.testing.assert_array_equal(A.replay(ref, json.loads(json.dumps(lab))), out)


def test_budget_floor_follows_own_noise():
    ref, brain = phantom()
    _, lab = corrupt_retry(ref, brain, {"motion": "borderline", "spike": "borderline"}, 4, native_sigma=0.027)
    assert lab["budget"]["floor"] == 0.027


def test_budget_needs_one_level():
    ref, brain = phantom()
    with pytest.raises(ValueError, match="one level"):
        A.corrupt(ref, brain, {"motion": "mild", "spike": "moderate"}, seed=1, cfg2=CFG2, native_sigma=0.02)


def test_single_motion_like_artifact_unchanged():
    """With at most one motion-like artifact the budget is not used: output and label are as before D65."""
    ref, brain = phantom()
    no_budget = {k: v for k, v in CFG2.items() if k != "budget"}
    for recipe in ({"motion": "moderate", "noise": "mild"}, {"bias": "mild", "spike": "borderline"},
                   {"ghosting": "mild", "noise": "borderline", "bias": "moderate"}):
        a, la = A.corrupt(ref, brain, recipe, seed=7, cfg2=CFG2, native_sigma=0.02)
        b, lb = A.corrupt(ref, brain, recipe, seed=7, cfg2=no_budget, native_sigma=0.02)
        np.testing.assert_array_equal(a, b)
        assert "budget" not in la and json.dumps(la) == json.dumps(lb)

