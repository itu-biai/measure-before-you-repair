"""blind-mri-v2 (3-D) tests, as listed in the v2 before-doc §11 (04_dataset_v2_3d/1_before.html)."""
import copy
import gzip
import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

from bamr.config import REPO, load_config
from bamr.data import artifacts3d as B
from bamr.data.artifacts import LEVELS
from bamr.data.assemble import Member
from bamr.data.blindio import AFFINE3, BOX3, nifti_bytes3
from bamr.data.plan_v2 import plan_cases_v2, split_v2

CFG = load_config(REPO / "configs" / "dataset_v2.yaml")
C2, C3 = CFG["stage2"], CFG["stage3"]
SHAPE = (48, 64, 60)  # small (LR, AP, SI) phantom box for speed
SMALL2 = copy.deepcopy(C2)
SMALL2["spike"]["k_offset"] = [4, 12]


def phantom(shape=SHAPE):
    """Head-like volume: scalp 0.6, GM 0.8, WM 1, a ventricle; brain = GM + WM; air = 0."""
    g = np.meshgrid(*[np.arange(n) - (n - 1) / 2 for n in shape], indexing="ij")
    r = np.sqrt(sum((x / (0.42 * n)) ** 2 for x, n in zip(g, shape)))
    v = np.zeros(shape, np.float32)
    v[r < 1.0] = 0.6
    brain = r < 0.85
    v[brain] = 0.8
    v[r < 0.7] = 1.0
    v[np.sqrt((g[0] / 4) ** 2 + ((g[1] - 5) / 9) ** 2 + (g[2] / 6) ** 2) < 1] = 0.3
    return v, brain


def kspace(x):
    return torch.fft.fftshift(torch.fft.fftn(torch.from_numpy(np.ascontiguousarray(B.to_t(x))))).numpy()


# ---------------------------------------------------------------- box (D75)
def test_box3d_round_trip_and_size():
    vol = np.random.default_rng(0).random((180, 250, 230)).astype(np.float32)
    mask = np.zeros(vol.shape, bool)
    mask[20:170, 30:229, 40:196] = True  # 150 × 199 × 156 mm, the largest brain of the 581 (D75)
    o = B.box_origin(mask)
    box = B.cut_box(vol, o)
    assert box.shape == B.BOX == BOX3
    assert B.cut_box(mask, o).sum() == mask.sum()  # the whole brain is inside the box
    lo = [max(-x, 0) for x in o]
    hi = [min(n, s - x) for n, s, x in zip(B.BOX, vol.shape, o)]
    src = tuple(slice(a + x, b + x) for a, b, x in zip(lo, hi, o))
    np.testing.assert_array_equal(box[tuple(slice(a, b) for a, b in zip(lo, hi))], vol[src])
    assert box[:lo[0]].sum() == 0  # zero padding where the box runs past the volume


def test_to_t_puts_ap_last():
    x = np.zeros(SHAPE)
    assert B.to_t(x).shape == (SHAPE[0], SHAPE[2], SHAPE[1])
    y = np.random.default_rng(1).random(SHAPE)
    np.testing.assert_array_equal(B.from_t(B.to_t(y)), y)


# ---------------------------------------------------------------- motion (D76, D88)
MC = {**C2["motion"], "rtol": 0.01}


def test_motion3d_kspace_error_along_ap_only():
    ref, _ = phantom()
    p = {"degrees": [[1.5, -1.0, 2.0]], "translation": [[1.0, -2.0, 1.5]], "times": [0.4],
         "image_interpolation": "bspline"}
    d = np.abs(kspace(B.apply3(ref, "motion", p)) - kspace(ref))
    per_ap = d.sum(axis=(0, 1))  # one value per A–P plane
    per_lr = d.sum(axis=(1, 2))
    centre = SHAPE[1] // 2
    assert per_ap[centre] < 1e-3 * per_ap.max()  # the k-space centre segment is the original scan's
    untouched = per_ap < 1e-3 * per_ap.max()
    edges = np.flatnonzero(np.diff(untouched.astype(int)))
    assert len(edges) <= 2  # one contiguous untouched block of A–P planes, the rest from the moved head
    assert (per_lr > 1e-3 * per_lr.max()).all()  # every L–R partition is affected: the split is not along L–R


def test_motion3d_six_dof_and_replay():
    ref, brain = phantom()
    p = B.sample_motion3(np.random.default_rng(3), [0.02, 0.03], MC, ref, brain)
    deg, tr = np.asarray(p["degrees"]), np.asarray(p["translation"])
    assert deg.shape[1] == tr.shape[1] == 3 and np.count_nonzero(deg) + np.count_nonzero(tr) == 6 * len(deg)
    assert np.abs(deg).max() <= 2 and np.abs(tr).max() <= 2 and 0 < p["amplitude_scale"] <= 1
    assert abs(p["motion_rms"] - p["target_rms"]) <= 0.01 * p["target_rms"]
    assert 0.02 <= p["target_rms"] <= 0.03
    np.testing.assert_array_equal(np.abs(B.apply3(ref, "motion", p)), np.abs(B.apply3(ref, "motion", p)))
    assert B.damage3(ref, brain, "motion", p) == p["motion_rms"]


# ---------------------------------------------------------------- ghosting and spike (D77)
def test_ghosting3d_copies_along_ap_only():
    v = np.zeros(SHAPE, np.float32)
    c = [n // 2 for n in SHAPE]
    v[c[0] - 1:c[0] + 2, c[1] - 1:c[1] + 2, c[2] - 1:c[2] + 2] = 1
    p = {"num_ghosts": 2, "axis": 2, "intensity": 0.6, "restore": None}
    out = np.abs(B.apply3(v, "ghosting", p))
    ap_ghost = out[c[0], (c[1] + SHAPE[1] // 2) % SHAPE[1], c[2]]
    lr_ghost = out[(c[0] + SHAPE[0] // 2) % SHAPE[0], c[1], c[2]]
    si_ghost = out[c[0], c[1], (c[2] + SHAPE[2] // 2) % SHAPE[2]]
    assert ap_ghost > 0.1 and lr_ghost < 1e-4 and si_ghost < 1e-4


def test_ghosting3d_hits_target():
    ref, brain = phantom()
    p = B.sample_ghosting3(np.random.default_rng(4), [0.02, 0.02], C2["ghosting"], ref, brain)
    assert abs(p["ghosting_rms"] - 0.02) <= 1e-3 * 0.02 and p["axis"] == 2


def test_spike3d_reaches_every_axial_slice_and_hits_target():
    ref, brain = phantom()
    p = B.sample_spike3(np.random.default_rng(5), [0.02, 0.02], SMALL2["spike"], ref, ref, brain)
    assert abs(p["spike_rms"] - 0.02) <= 1e-3 * 0.02
    d = np.abs(np.abs(B.apply3(ref, "spike", p)) - ref)
    assert (d.max(axis=(0, 1)) > 1e-3).all()  # every S–I (axial) slice carries the stripes
    pos = np.asarray(p["spikes_positions"])
    assert pos.shape[1] == 3 and ((pos > 0) & (pos < 1)).all()


# ---------------------------------------------------------------- bias (D78)
def test_bias3d_whole_brain_maxmin_and_all_terms():
    ref, brain = phantom()
    p = B.sample_bias3(np.random.default_rng(6), [0.35, 0.45], brain)
    assert np.count_nonzero(p["coefficients"]) == 20
    field = np.abs(B.apply3(np.ones(SHAPE, np.float32), "bias", p))[brain]
    assert field.max() / field.min() - 1 == pytest.approx(p["target_maxmin"], rel=1e-4)
    assert np.exp(np.log(field).mean()) == pytest.approx(1.0, rel=1e-4)  # WM = 1 scale kept


# ---------------------------------------------------------------- chain (D08, D65 in grade-matched units, D61)
def cfg_equiv():
    c = copy.deepcopy(SMALL2)
    c["budget"]["equiv"] = {"motion": 1.0, "ghosting": 2.0, "spike": 1.5}
    for a, f in c["budget"]["equiv"].items():
        c[a]["rms_band"] = [[x / f for x in b] for b in c["damage_band"]]
    return c


def corrupt_retry(ref, brain, recipe, seed, cfg2, sigma=0.02):
    for k in range(6):
        try:
            return B.corrupt3(ref, brain, recipe, 1000 * seed + k, cfg2, sigma)
        except RuntimeError:
            continue
    raise RuntimeError("no valid draw")


def test_chain3d_order_budget_noise_and_replay():
    ref, brain = phantom()
    c2 = cfg_equiv()
    recipe = {"noise": "mild", "spike": "moderate", "motion": "moderate", "bias": "mild", "ghosting": "moderate"}
    out, lab = corrupt_retry(ref, brain, recipe, 7, c2)
    assert lab["artifacts"] == ["bias", "motion", "ghosting", "spike", "noise"]
    b = lab["budget"]
    band = c2["damage_band"][LEVELS.index("moderate")]
    assert band[0] <= b["total"] <= band[1]
    assert b["total"] == pytest.approx(np.sqrt(sum(v ** 2 for v in b["equivalent"].values())))
    for a in ("ghosting", "spike"):  # own damage × equiv = the motion-equivalent share
        own = lab["params"][a][f"{a}_rms"]
        assert own * c2["budget"]["equiv"][a] == pytest.approx(b["shares"][a], rel=2e-3)
    assert lab["params"]["motion"]["motion_rms"] == pytest.approx(b["shares"]["motion"], rel=0.011)
    assert all(v >= c2["budget"]["min_share"] * 0.98 for v in b["equivalent"].values())
    n = lab["params"]["noise"]
    assert n["sigma_total"] ** 2 == pytest.approx(n["sigma_native"] ** 2 + n["sigma_added"] ** 2)
    np.testing.assert_array_equal(B.replay3(ref, lab), out)
    assert out.dtype == np.float32 and out.shape == SHAPE and (out >= 0).all()


def test_chain3d_single_levels_land_in_own_bands():
    ref, brain = phantom()
    c2 = cfg_equiv()
    for a in ("ghosting", "spike"):
        for i, lv in enumerate(LEVELS):
            _, lab = B.corrupt3(ref, brain, {a: lv}, 11 + i, c2, 0.02)
            lo, hi = c2[a]["rms_band"][i]
            assert lo <= lab["params"][a][f"{a}_rms"] * (1 - 1e-3) <= hi


def test_clean_case_is_reference():
    ref, brain = phantom()
    out, lab = B.corrupt3(ref, brain, {}, 1, SMALL2, 0.02)
    np.testing.assert_array_equal(out, ref)
    assert lab["artifacts"] == []


# ---------------------------------------------------------------- split and plan (D80)
def candidates():
    """580 members, sites interleaved; a few noisy ones; 10 calibration subjects beyond rank 60."""
    sites = (["Guys"] * 11 + ["HH"] * 6 + ["IOP"] * 3) * 29
    return [Member(f"IXI{r:03d}", s, r, 0.035 if r % 37 == 5 else 0.02, True) for r, s in enumerate(sites)]


def v1_doc(c):
    test = [m for m in c[:60] if m.native_sigma < 0.03][:37]
    dev = [m for m in c[:60] if m.native_sigma < 0.03][37:47]
    row = lambda m: {"id": m.id, "site": m.site}  # noqa: E731
    return {"test": [row(m) for m in test], "dev": [row(m) for m in dev]}


CALIB = {f"IXI{r:03d}" for r in range(60, 70)}


def test_split_v2_rules():
    c = candidates()
    v1 = v1_doc(c)
    s = split_v2(c, v1, CALIB, C3)
    for name in ("test", "dev", "reserve"):
        assert dict(Counter(m.site for m in s[name])) == C3["split"][name]
    ids = {n: {m.id for m in s[n]} for n in ("test", "dev", "reserve")}
    assert {r["id"] for r in v1["test"]} <= ids["test"] and {r["id"] for r in v1["dev"]} <= ids["dev"]
    assert not ids["test"] & ids["dev"] and not ids["test"] & ids["reserve"] and not ids["dev"] & ids["reserve"]
    assert not (ids["test"] | ids["dev"]) & CALIB
    assert all(m.native_sigma < 0.03 for n in ("test", "dev", "reserve") for m in s[n])
    assert {m.id for m in s["sweep"]} == {r["id"] for r in v1["dev"]}


def test_split_v2_flagged_replaced_by_reserve_of_site():
    c = candidates()
    s0 = split_v2(c, v1_doc(c), CALIB, C3)
    victim = next(m for m in s0["test"] if m.site == "HH")
    c3 = copy.deepcopy(C3)
    c3["flagged"] = {victim.id: "native motion"}
    s = split_v2(c, v1_doc(c), CALIB, c3)
    first_hh_reserve = next(m for m in s0["reserve"] if m.site == "HH")
    assert victim not in s["test"] and first_hh_reserve in s["test"]
    assert any(e["id"] == victim.id and e["replaced_by"] == first_hh_reserve.id for e in s["excluded"])


def test_plan_v2_one_case_per_subject():
    c = candidates()
    s = split_v2(c, v1_doc(c), CALIB, C3)
    cases = plan_cases_v2(s, C3, CFG["plan_seed"], C2)
    by = Counter(x["set"] for x in cases)
    assert by == {"test": 148, "dev": 60, "sweep": 150}
    for name in ("test", "dev"):
        subs = [x["subject"] for x in cases if x["set"] == name]
        assert len(subs) == len(set(subs))
    ids = [x["case_id"] for x in cases]
    assert len(set(ids)) == len(ids) and all(i[0] == {"test": "t", "dev": "d", "sweep": "s"}[x["set"]]
                                             for i, x in zip(ids, cases))
    pool = C2["budget"]["pool"]
    for x in cases:  # pooled motion-like artifacts share one level (D65)
        lv = {x["recipe"][a] for a in pool if a in x["recipe"]}
        if sum(a in pool for a in x["recipe"]) >= 2:
            assert len(lv) == 1 and None not in lv


# ---------------------------------------------------------------- blind files (D81)
def test_blind_v2_header_identical_and_clean_gzip():
    a = nifti_bytes3(np.zeros(BOX3, np.float32), np.float32)
    b = nifti_bytes3(np.random.default_rng(0).random(BOX3).astype(np.float32), np.float32)
    assert gzip.decompress(a)[:352] == gzip.decompress(b)[:352]
    assert a[4:8] == b"\0\0\0\0" and not a[3] & 0x08
    assert np.allclose(AFFINE3[:3, 3], -(np.array(BOX3) - 1) / 2)
    with pytest.raises(ValueError):
        nifti_bytes3(np.zeros(SHAPE, np.float32), np.float32)


# ---------------------------------------------------------------- v1 untouched (D66)
V1 = REPO.parent / "dataset"


@pytest.mark.skipif(not (V1 / "blind-mri-v1.sha256").exists(), reason="dataset not on this machine")
def test_v1_untouched():
    lines = (V1 / "blind-mri-v1.sha256").read_text().split("\n")
    bad = []
    for line in filter(None, lines):
        digest, name = line.split(maxsplit=1)
        p = V1 / name.lstrip("*")
        if hashlib.sha256(Path(p).read_bytes()).hexdigest() != digest:
            bad.append(name)
    assert not bad and len([x for x in lines if x]) == 1434
