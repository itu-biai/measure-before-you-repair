"""Stage 3 planning: which subjects, which slices, which recipes. Pure: reads no images (D51–D58).

Split (D51, D52, D57): the 60 priority subjects (D04) minus those that fail QC or whose own noise is already at the
mild noise level; filled per site in priority order: test, then dev, then reserves. A flagged subject is replaced in
place by the next reserve of its site.
Recipes (D47, D53, D55): a deck sorted by number of artifacts (combination order shuffled within a size, copies
adjacent), with levels balanced per artifact within each size; subject p takes cards p, p + S, p + 2S, … and its
recipes are shuffled over its slices. Seeds and opaque IDs follow from the config seed alone.
Budget (D65): in a case with two or more motion-like artifacts they share one level, the level of their joint damage.
It is set after dealing, balanced per size, among the levels the subject's own noise allows (every share ≥ max(own σ,
min_share)); the other artifacts keep their per-artifact balance.
"""
import itertools
from dataclasses import dataclass

import numpy as np

from bamr.data.artifacts import LEVELS, ORDER

SET_INDEX = {"test": 0, "dev": 1, "sweep": 2}
PREFIX = {"test": "t", "dev": "d", "sweep": "s"}
SPLIT_SETS = ("test", "dev", "reserve")
_DECK, _LEVELS, _DEAL, _IDS, _BUDGET = range(5)  # planning RNG purposes


@dataclass(frozen=True)
class Member:
    id: str
    site: str
    rank: int
    native_sigma: float
    qc_ok: bool


def _rng(seed: int, set_name: str, purpose: int) -> np.random.Generator:
    return np.random.default_rng([seed, 3, SET_INDEX[set_name], purpose])


def case_seed(seed: int, set_name: str, k: int, attempt: int = 0) -> int:
    """Seed of the k-th case of a set in plan order; a failed draw retries with attempt + 1 (D58)."""
    return int(np.random.SeedSequence([seed, 3, SET_INDEX[set_name], k, attempt]).generate_state(1)[0])


# ---------------------------------------------------------------- split
def split_subjects(candidates: list[Member], c3: dict) -> dict:
    excluded, eligible = [], []
    for m in sorted(candidates, key=lambda m: m.rank):
        if not m.qc_ok:
            excluded.append({"id": m.id, "site": m.site, "rank": m.rank, "reason": "QC failed (Stage 1)",
                             "replaced_by": None})
        elif m.native_sigma >= c3["max_native_sigma"]:
            excluded.append({"id": m.id, "site": m.site, "rank": m.rank, "replaced_by": None,
                             "reason": f"D51: own σ {100 * m.native_sigma:.2f} % ≥ "
                                       f"{100 * c3['max_native_sigma']:.1f} % (already at the mild noise level)"})
        else:
            eligible.append(m)
    out = {s: [] for s in SPLIT_SETS}
    for site in sorted({m.site for m in eligible}):
        pool = [m for m in eligible if m.site == site]
        for s in SPLIT_SETS:
            n = c3["split"][s].get(site, 0)
            if len(pool) < n:
                raise ValueError(f"{site}: {len(pool)} eligible subjects left for {n} {s} places")
            out[s] += pool[:n]
            pool = pool[n:]
    for s in SPLIT_SETS:
        out[s].sort(key=lambda m: m.rank)
    flagged = c3.get("flagged") or {}
    for m in sorted((m for s in SPLIT_SETS for m in out[s] if m.id in flagged), key=lambda m: m.rank):
        where = next(s for s in SPLIT_SETS if m in out[s])
        new = None
        if where == "reserve":
            out["reserve"].remove(m)
        else:
            spare = [r for r in out["reserve"] if r.site == m.site and r.id not in flagged]
            if not spare:
                raise ValueError(f"{m.site}: no reserve left to replace flagged {m.id}")
            new = spare[0]
            out["reserve"].remove(new)
            out[where][out[where].index(m)] = new
        excluded.append({"id": m.id, "site": m.site, "rank": m.rank, "reason": f"D57: {flagged[m.id]}",
                         "replaced_by": new.id if new else None})
    unknown = set(flagged) - {e["id"] for e in excluded}
    if unknown:
        raise ValueError(f"flagged subjects not in the split: {sorted(unknown)}")
    out["excluded"] = excluded
    return out


# ---------------------------------------------------------------- deck and levels
def level_counts(n: int, levels: tuple, running: dict[str, int], order: tuple) -> dict[str, int]:
    """n slots over the levels: n // k each; the n % k extra go to the levels with the lowest running count,
    ties in `order` (D53)."""
    base, rem = divmod(n, len(levels))
    extra = sorted(levels, key=lambda lv: (running[lv], order.index(lv)))[:rem]
    return {lv: base + (lv in extra) for lv in levels}


def _pooled(card, a: str, pool) -> bool:
    return a in pool and sum(x in pool for x in card) >= 2


def make_deck(set_name: str, c3: dict, seed: int, pool=()) -> list[dict[str, str]]:
    """Recipes sorted by size; within a size the combination order is shuffled and copies sit together. Artifacts that
    share a budget (D65) are left None here; budget_levels sets them after dealing."""
    cp = c3["copies"][set_name]
    rng = _rng(seed, set_name, _DECK)
    cards = []
    for size in range(len(ORDER) + 1):
        combos = list(itertools.combinations(ORDER, size))
        n = cp["regular"] if size <= 3 else cp["stress4"] if size == 4 else cp["stress5"]
        for i in rng.permutation(len(combos)):
            cards += [combos[i]] * n
    stress = tuple(c3["stress_levels"])
    groups = [(1, LEVELS), (2, LEVELS), (3, LEVELS), (4, stress)]  # 4 = stress group, sizes 4–5
    deck = [dict.fromkeys(c) for c in cards]
    rng = _rng(seed, set_name, _LEVELS)
    for ai, a in enumerate(ORDER):
        running = dict.fromkeys(LEVELS, 0)
        for g, levels in groups:
            slots = [i for i, c in enumerate(cards) if a in c and min(len(c), 4) == g and not _pooled(c, a, pool)]
            if levels is LEVELS:
                order = LEVELS[ai % 3:] + LEVELS[:ai % 3]
                counts = level_counts(len(slots), levels, running, order)
                for lv in levels:
                    running[lv] += counts[lv]
            else:
                counts = level_counts(len(slots), levels, dict.fromkeys(levels, 0), levels)
            bag = [lv for lv in levels for _ in range(counts[lv])]
            for i, j in zip(slots, rng.permutation(len(bag))):
                deck[i][a] = bag[j]
    return deck


def deal(deck: list, n_subjects: int, set_name: str, seed: int) -> list[list[tuple[int, dict]]]:
    """Hands per subject position: subject p takes cards p, p + S, …, then its hand is shuffled over its slices."""
    if len(deck) % n_subjects:
        raise ValueError(f"{len(deck)} cards do not deal evenly to {n_subjects} subjects")
    rng = _rng(seed, set_name, _DEAL)
    seat = rng.permutation(n_subjects)  # subject position → column of the deck
    hands = []
    for p in range(n_subjects):
        cards = list(range(int(seat[p]), len(deck), n_subjects))
        hands.append([(c, deck[c]) for c in (cards[i] for i in rng.permutation(len(cards)))])
    return hands


def budget_feasible(recipe: dict, m: Member, level: str, c2: dict) -> bool:
    """Can the level's band hold k shares of at least max(own σ, min_share) each? (D65) v2 drops the own-σ part: its
    shares are judged on the clinical scale, not against the scan's noise (own_sigma_floor: false, D89)."""
    b = c2["budget"]
    k = sum(a in b["pool"] for a in recipe)
    own = m.native_sigma if b.get("own_sigma_floor", True) else 0.0
    return c2["damage_band"][LEVELS.index(level)][1] > max(own, b["min_share"]) * np.sqrt(k)


def budget_levels(cases: list[tuple[dict, Member]], c3: dict, c2: dict, set_name: str, seed: int) -> None:
    """Set the shared level of every pooled recipe in place: balanced per size group (2, 3, and stress 4–5 with the
    stress levels; sizes 2 and 3 carry a running count), the most constrained cases choosing first. A stress case that
    cannot take any stress level gets its lowest feasible level."""
    pool, rng = c2["budget"]["pool"], _rng(seed, set_name, _BUDGET)
    running = dict.fromkeys(LEVELS, 0)
    for g, levels in ((2, LEVELS), (3, LEVELS), (4, tuple(c3["stress_levels"]))):
        items = [(r, m) for r, m in cases if min(len(r), 4) == g]
        if not items:
            continue
        left = level_counts(len(items), levels, running if levels is LEVELS else dict.fromkeys(levels, 0), LEVELS)
        tie = rng.permutation(len(items))
        ok = [[lv for lv in levels if budget_feasible(r, m, lv, c2)] for r, m in items]
        for j in sorted(range(len(items)), key=lambda j: (len(ok[j]), tie[j])):
            r, m = items[j]
            choice = ok[j] or [lv for lv in LEVELS if budget_feasible(r, m, lv, c2)][:1]
            if not choice:
                raise ValueError(f"{m.id}: no level can hold its motion-like budget (D65)")
            pref = list(rng.permutation(levels))
            lv = max(choice, key=lambda v: (left.get(v, 0), -pref.index(v) if v in pref else -len(pref)))
            left[lv] = left.get(lv, 0) - 1
            if levels is LEVELS:
                running[lv] += 1
            for a in r:
                if a in pool:
                    r[a] = lv


# ---------------------------------------------------------------- the plan
def plan_cases(split: dict, c3: dict, seed: int, c2: dict) -> list[dict]:
    """Every case of every set, in plan order (subjects by rank, offsets ascending), with opaque shuffled IDs."""
    pool = tuple(c2["budget"]["pool"]) if c2.get("budget") else ()
    out = []
    for set_name in ("test", "dev"):
        subs, offs = split[set_name], sorted(c3["offsets_mm"][set_name])
        hands = deal(make_deck(set_name, c3, seed, pool), len(subs), set_name, seed)
        if any(len(h) != len(offs) for h in hands):
            raise ValueError(f"{set_name}: hands of {len(hands[0])} cards for {len(offs)} slices")
        pending = []
        for m, hand in zip(subs, hands):
            for off, (card, recipe) in zip(offs, hand):
                out.append(_case(set_name, m, off, recipe, card))
                if sum(a in pool for a in recipe) >= 2:
                    pending.append((out[-1]["recipe"], m))
        if pending:
            budget_levels(pending, c3, c2, set_name, seed)
    for m in split["dev"]:
        for off in c3["offsets_mm"]["sweep"]:
            for a in ORDER:
                for lv in LEVELS:
                    out.append(_case("sweep", m, off, {a: lv}, None))
    for set_name in SET_INDEX:
        cases = [c for c in out if c["set"] == set_name]
        ids = _rng(seed, set_name, _IDS).permutation(len(cases))
        for k, (c, i) in enumerate(zip(cases, ids)):
            c["k"] = k
            c["case_id"] = f"{PREFIX[set_name]}{int(i) + 1:04d}"
    return out


def _case(set_name: str, m: Member, off: int, recipe: dict, card) -> dict:
    return {"case_id": None, "set": set_name, "subject": m.id, "site": m.site, "native_sigma": float(m.native_sigma),
            "offset_mm": int(off), "recipe": dict(recipe), "card": None if card is None else int(card), "k": None}
