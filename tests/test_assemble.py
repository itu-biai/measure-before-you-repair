import copy
import itertools
from collections import Counter

import pytest

from bamr.config import load_config
from bamr.data import artifacts as A
from bamr.data.assemble import (Member, budget_feasible, case_seed, deal, level_counts, make_deck, plan_cases,
                                split_subjects)

CFG = load_config()
C3 = CFG["stage3"]
C2 = CFG["stage2"]
SEED = CFG["seed"]
SITES = ["Guys"] * 33 + ["HH"] * 19 + ["IOP"] * 8
NOISY = {3: 0.031, 9: 0.035}  # two Guys members already inside the mild noise band (D51)


def members(order=7):
    """60 priority members, sites interleaved as a shuffled priority order would be."""
    sites = [SITES[(i * order) % 60] for i in range(60)]
    assert Counter(sites) == Counter(SITES)
    out = []
    for r, site in enumerate(sites):
        sig = NOISY.get(r, 0.02) if site == "Guys" else 0.02
        out.append(Member(f"IXI{r:03d}", site, r, sig, True))
    noisy = [m for m in out if m.native_sigma >= 0.030]
    assert len(noisy) == 2, "fixture: both noisy members must be Guys"
    return out


def c3_with(flagged=None):
    c = copy.deepcopy(C3)
    c["flagged"] = flagged or {}
    return c


# ---------------------------------------------------------------- split (D51, D52, D57)
def test_split_counts_and_sites():
    sp = split_subjects(members(), c3_with())
    assert [len(sp[s]) for s in ("test", "dev", "reserve")] == [37, 10, 11]
    for s, want in (("test", (20, 12, 5)), ("dev", (6, 3, 1)), ("reserve", (5, 4, 2))):
        c = Counter(m.site for m in sp[s])
        assert (c["Guys"], c["HH"], c["IOP"]) == want
    ids = [m.id for s in ("test", "dev", "reserve") for m in sp[s]]
    assert len(ids) == len(set(ids))
    for site in ("Guys", "HH", "IOP"):
        ranks = {s: [m.rank for m in sp[s] if m.site == site] for s in ("test", "dev", "reserve")}
        for s in ranks:
            assert ranks[s] == sorted(ranks[s])
        assert max(ranks["test"]) < min(ranks["dev"]) < min(ranks["reserve"])
    ex = {e["id"]: e for e in sp["excluded"]}
    assert set(ex) == {"IXI003", "IXI009"} and all("D51" in e["reason"] for e in ex.values())


def _with_spare_guys():
    """The 58 eligible members fill every place exactly; one extra Guys member makes room for one more exclusion."""
    ms = members() + [Member("IXI060", "Guys", 60, 0.02, True)]
    return ms, next(i for i, m in enumerate(ms) if m.site == "Guys" and m.native_sigma < 0.03)


def test_split_sigma_limit_is_exclusive():
    ms, i = _with_spare_guys()
    ms[i] = Member(ms[i].id, "Guys", ms[i].rank, 0.030, True)
    sp = split_subjects(ms, c3_with())
    assert ms[i].id in {e["id"] for e in sp["excluded"]}


def test_split_qc_fail_excluded():
    ms, i = _with_spare_guys()
    ms[i] = Member(ms[i].id, "Guys", ms[i].rank, 0.02, False)
    sp = split_subjects(ms, c3_with())
    assert {e["id"]: e["reason"] for e in sp["excluded"]}[ms[i].id].startswith("QC")


def test_split_too_few_raises():
    ms = members()
    i = next(i for i, m in enumerate(ms) if m.site == "HH")
    ms[i] = Member(ms[i].id, "HH", ms[i].rank, 0.02, False)
    with pytest.raises(ValueError, match="HH"):
        split_subjects(ms, c3_with())


@pytest.mark.parametrize("where", ["test", "dev", "reserve"])
def test_split_flagged_replaced_in_place(where):
    base = split_subjects(members(), c3_with())
    victim = next(m for m in base[where] if m.site == "HH")
    first_hh_reserve = next(m for m in base["reserve"] if m.site == "HH")
    sp = split_subjects(members(), c3_with({victim.id: "visible head motion"}))
    ex = {e["id"]: e for e in sp["excluded"]}
    assert ex[victim.id]["reason"].startswith("D57")
    if where == "reserve":
        assert sp["test"] == base["test"] and sp["dev"] == base["dev"]
        assert len(sp["reserve"]) == 10 and victim not in sp["reserve"]
        assert ex[victim.id]["replaced_by"] is None
    else:
        i = base[where].index(victim)
        assert sp[where][i] == first_hh_reserve
        assert [m for j, m in enumerate(sp[where]) if j != i] == [m for j, m in enumerate(base[where]) if j != i]
        other = "dev" if where == "test" else "test"
        assert sp[other] == base[other]
        assert first_hh_reserve not in sp["reserve"] and len(sp["reserve"]) == 10
        assert ex[victim.id]["replaced_by"] == first_hh_reserve.id


def test_split_runs_out_of_reserves_raises():
    base = split_subjects(members(), c3_with())
    iop = [m.id for m in base["test"] if m.site == "IOP"][:3]
    with pytest.raises(ValueError, match="IOP"):
        split_subjects(members(), c3_with({i: "artifact" for i in iop}))


# ---------------------------------------------------------------- deck and levels (D53, D55)
def test_level_counts():
    z = dict.fromkeys(A.LEVELS, 0)
    assert level_counts(5, A.LEVELS, z, A.LEVELS) == {"mild": 2, "moderate": 2, "borderline": 1}
    run = {"mild": 2, "moderate": 2, "borderline": 1}
    assert level_counts(20, A.LEVELS, run, A.LEVELS) == {"mild": 7, "moderate": 6, "borderline": 7}
    st = ("mild", "moderate")
    assert level_counts(15, st, dict.fromkeys(st, 0), st) == {"mild": 8, "moderate": 7}


def _size(r):
    return len(r)


@pytest.mark.parametrize("set_name, sizes, regular, s4, s5", [
    ("test", [5, 25, 50, 50, 15, 3], 5, 3, 3),
    ("dev", [2, 10, 20, 20, 5, 3], 2, 1, 3)])
def test_deck_counts(set_name, sizes, regular, s4, s5):
    deck = make_deck(set_name, C3, SEED)
    assert [sum(len(r) == k for r in deck) for k in range(6)] == sizes
    assert [len(r) for r in deck] == sorted(len(r) for r in deck)
    per = Counter(tuple(r) for r in deck)
    for combo, n in per.items():
        assert n == (regular if len(combo) <= 3 else s4 if len(combo) == 4 else s5)
    assert len(per) == 32
    for r in deck:
        assert list(r) == [a for a in A.ORDER if a in r]
        if len(r) >= 4:
            assert set(r.values()) <= {"mild", "moderate"}
    # copies of one combination sit next to each other
    runs = [k for k, _ in itertools.groupby(tuple(r) for r in deck)]
    assert len(runs) == 32


def _balance(deck):
    """artifact → size group → Counter(level); group 4 = stress (4–5 artifacts)."""
    out = {a: {g: Counter() for g in (1, 2, 3, 4)} for a in A.ORDER}
    for r in deck:
        for a, lev in r.items():
            out[a][min(len(r), 4)][lev] += 1
    return out


def test_severity_balance_test():
    b = _balance(make_deck("test", C3, SEED))
    totals = Counter()
    for i, a in enumerate(A.ORDER):
        assert sorted(b[a][1][lv] for lv in A.LEVELS) == [1, 2, 2]
        assert sorted(b[a][2][lv] for lv in A.LEVELS) == [6, 7, 7]
        assert [b[a][3][lv] for lv in A.LEVELS] == [10, 10, 10]
        tot = {lv: sum(b[a][g][lv] for g in (1, 2, 3)) for lv in A.LEVELS}
        assert sorted(tot.values()) == [18, 18, 19] and tot[A.LEVELS[i % 3]] == 19
        totals.update(tot)
        assert dict(b[a][4]) == {"mild": 8, "moderate": 7}
    assert max(totals.values()) - min(totals.values()) <= 1


def test_severity_balance_dev():
    b = _balance(make_deck("dev", C3, SEED))
    for a in A.ORDER:
        tot = [sum(b[a][g][lv] for g in (1, 2, 3)) for lv in A.LEVELS]
        assert sorted(tot) == [7, 7, 8]
        assert dict(b[a][4]) == {"mild": 4, "moderate": 3}


@pytest.mark.parametrize("set_name, n_subjects, rows", [("test", 37, 4), ("dev", 10, 6)])
def test_distinct_per_subject_and_one_card_per_row(set_name, n_subjects, rows):
    deck = make_deck(set_name, C3, SEED)
    hands = deal(deck, n_subjects, set_name, SEED)
    assert len(hands) == n_subjects
    assert sorted(c for h in hands for c, _ in h) == list(range(len(deck)))
    for h in hands:
        combos = [frozenset(r) for _, r in h]
        assert len(set(combos)) == len(combos) == rows
        assert sorted(c // n_subjects for c, _ in h) == list(range(rows))
        for c, r in h:
            assert r == deck[c]


# ---------------------------------------------------------------- the plan (D54, D56, D58)
def test_plan_counts_offsets_ids():
    sp = split_subjects(members(), c3_with())
    cases = plan_cases(sp, C3, SEED, C2)
    by = {s: [c for c in cases if c["set"] == s] for s in ("test", "dev", "sweep")}
    assert [len(by[s]) for s in by] == [148, 60, 150]
    for s, subs in (("test", sp["test"]), ("dev", sp["dev"])):
        for m in subs:
            offs = sorted(c["offset_mm"] for c in by[s] if c["subject"] == m.id)
            assert offs == C3["offsets_mm"][s]
    for m in sp["dev"]:
        sw = [c for c in by["sweep"] if c["subject"] == m.id]
        assert len(sw) == 15 and {c["offset_mm"] for c in sw} == {0}
        assert sorted((a, lv) for c in sw for a, lv in c["recipe"].items()) == \
            sorted(itertools.product(A.ORDER, A.LEVELS))
    for s, p, n in (("test", "t", 148), ("dev", "d", 60), ("sweep", "s", 150)):
        ids = [c["case_id"] for c in by[s]]
        assert sorted(ids) == [f"{p}{i:04d}" for i in range(1, n + 1)]
        assert ids != sorted(ids)
    assert len({c["case_id"] for c in cases}) == 358
    assert all(c["k"] == i for s in by for i, c in enumerate(by[s]))
    assert {c["native_sigma"] for c in cases} <= {m.native_sigma for m in sp["test"] + sp["dev"]}


def test_plan_deterministic():
    sp = split_subjects(members(), c3_with())
    assert plan_cases(sp, C3, SEED, C2) == plan_cases(sp, C3, SEED, C2)
    assert make_deck("test", C3, SEED + 1) != make_deck("test", C3, SEED)


def test_case_seed_attempts_differ():
    assert case_seed(SEED, "test", 3, 0) != case_seed(SEED, "test", 3, 1)
    assert case_seed(SEED, "test", 3) != case_seed(SEED, "dev", 3)
    assert case_seed(SEED, "test", 3) == case_seed(SEED, "test", 3, 0)


# ---------------------------------------------------------------- D65: motion-like budget in the plan
POOL = set(C2["budget"]["pool"])


def _pooled(c):
    return sum(a in POOL for a in c["recipe"]) >= 2


def _levels_by_artifact(cases, artifacts, solo_only=False):
    out = {a: {g: Counter() for g in (1, 2, 3, 4)} for a in artifacts}
    for c in cases:
        for a, lv in c["recipe"].items():
            if a in artifacts and not (solo_only and _pooled(c)):
                out[a][min(len(c["recipe"]), 4)][lv] += 1
    return out


def test_budget_plan_test():
    sp = split_subjects(members(), c3_with())
    t = [c for c in plan_cases(sp, C3, SEED, C2) if c["set"] == "test"]
    pooled = [c for c in t if _pooled(c)]
    assert len(pooled) == 68
    for c in pooled:  # one shared level for the motion-like artifacts
        assert len({lv for a, lv in c["recipe"].items() if a in POOL}) == 1
    assert all(lv is not None for c in t for lv in c["recipe"].values())
    shared = {g: Counter(next(lv for a, lv in c["recipe"].items() if a in POOL) for c in pooled
                         if min(len(c["recipe"]), 4) == g) for g in (2, 3, 4)}
    assert sorted(shared[2].values()) == [5, 5, 5]
    assert sorted(shared[3].values()) == [11, 12, 12]
    assert dict(shared[4]) == {"mild": 9, "moderate": 9}
    b = _levels_by_artifact(t, ["bias", "noise"])  # unchanged per-artifact balance for the other artifacts
    for a in ("bias", "noise"):
        assert sorted(sum(b[a][g][lv] for g in (1, 2, 3)) for lv in A.LEVELS) == [18, 18, 19]
        assert dict(b[a][4]) == {"mild": 8, "moderate": 7}
    solo = _levels_by_artifact(t, sorted(POOL), solo_only=True)
    for a in POOL:
        assert sorted(solo[a][1][lv] for lv in A.LEVELS) == [1, 2, 2]
        tot = [sum(solo[a][g][lv] for g in (1, 2, 3)) for lv in A.LEVELS]
        assert max(tot) - min(tot) <= 1


def test_budget_plan_respects_own_noise():
    noisy = [Member(m.id, m.site, m.rank, 0.029 if m.native_sigma < 0.03 else m.native_sigma, True) for m in members()]
    sp = split_subjects(noisy, c3_with())
    for c in plan_cases(sp, C3, SEED, C2):
        if not _pooled(c):
            continue
        m = next(x for x in sp[c["set"]] if x.id == c["subject"])
        lv = next(lv for a, lv in c["recipe"].items() if a in POOL)
        assert budget_feasible(c["recipe"], m, lv, C2)
        if min(len(c["recipe"]), 4) < 4:
            continue
        k = sum(a in POOL for a in c["recipe"])  # σ 2.9 %: 3 shares need the borderline band
        assert lv == ("borderline" if k == 3 else lv)


def test_make_deck_leaves_pooled_open():
    deck = make_deck("test", C3, SEED, tuple(POOL))
    for r in deck:
        for a, lv in r.items():
            assert (lv is None) == (a in POOL and sum(x in POOL for x in r) >= 2)

