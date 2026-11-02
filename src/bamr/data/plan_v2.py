"""blind-mri-v2 planning: which subjects, which recipes (D80). Pure: reads no images.

Split (D80): one subject per test and dev-mixed case. v1's 37 test subjects stay in test and v1's 10 dev subjects stay in
dev (so v1 dev, which Task 2 pilots have looked at, never reaches v2 test); the rest is filled per site in the seeded
priority order (D04) from subjects that pass Stage 1 QC, have own noise below the mild band (D51) and are not
calibration subjects (D28): test first, then dev, then reserves. A flagged subject (D57) is replaced by the next reserve
of its site. The single-artifact sweep keeps v1's design: v1's 10 dev subjects × 5 artifacts × 3 levels.
Recipes: v1's deck, dealing and level balance (D47, D53, D55) and the motion-like budget levels (D65); with one case
per subject, each subject gets one card.
"""
from bamr.data.artifacts import LEVELS, ORDER
from bamr.data.assemble import PREFIX, SET_INDEX, Member, _rng, budget_levels, deal, make_deck

SPLIT_SETS = ("test", "dev", "reserve")
_IDS = 3


def split_v2(cands: list[Member], v1: dict, calib_ids: set[str], c3: dict) -> dict:
    """cands: every subject in priority order (rank = position). v1: the v1 split document (test / dev id lists)."""
    v1_test = [r["id"] for r in v1["test"]]
    v1_dev = [r["id"] for r in v1["dev"]]
    by_id = {m.id: m for m in cands}
    excluded, eligible = [], []
    for m in sorted(cands, key=lambda m: m.rank):
        reason = ("QC failed (Stage 1)" if not m.qc_ok else
                  "calibration subject (D28)" if m.id in calib_ids else
                  f"D51: own σ {100 * m.native_sigma:.2f} % ≥ {100 * c3['max_native_sigma']:.1f} %"
                  if m.native_sigma >= c3["max_native_sigma"] else None)
        if reason:
            excluded.append({"id": m.id, "site": m.site, "rank": m.rank, "reason": reason, "replaced_by": None})
        else:
            eligible.append(m)
    ok = {m.id for m in eligible}
    for sid in v1_test + v1_dev:
        if sid not in ok:
            raise ValueError(f"v1 subject {sid} is not eligible for v2")
    out = {"test": [by_id[s] for s in v1_test], "dev": [by_id[s] for s in v1_dev], "reserve": []}
    fresh = [m for m in eligible if m.id not in set(v1_test) | set(v1_dev)]
    for site in sorted({m.site for m in cands}):
        pool = [m for m in fresh if m.site == site]
        for s in SPLIT_SETS:
            n = c3["split"][s].get(site, 0) - sum(m.site == site for m in out[s])
            if n < 0 or len(pool) < n:
                raise ValueError(f"{site}: cannot fill {s} ({len(pool)} left, {n} needed)")
            out[s] += pool[:n]
            pool = pool[n:]
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
    for s in SPLIT_SETS:
        out[s].sort(key=lambda m: m.rank)
    out["sweep"] = [m for m in out["dev"] if m.id in set(v1_dev)]
    out["excluded"] = excluded
    return out


def plan_cases_v2(split: dict, c3: dict, seed: int, c2: dict) -> list[dict]:
    """Every case of every set in plan order, with opaque shuffled IDs (t/d/s + 4 digits)."""
    pool = tuple(c2["budget"]["pool"])
    out = []
    for set_name in ("test", "dev"):
        subs = split[set_name]
        hands = deal(make_deck(set_name, c3, seed, pool), len(subs), set_name, seed)
        pending = []
        for m, hand in zip(subs, hands):
            if len(hand) != 1:
                raise ValueError(f"{set_name}: {len(hand)} cards for one subject; v2 has one case per subject (D80)")
            card, recipe = hand[0]
            out.append(_case(set_name, m, recipe, card))
            if sum(a in pool for a in recipe) >= 2:
                pending.append((out[-1]["recipe"], m))
        if pending:
            budget_levels(pending, c3, c2, set_name, seed)
    for m in split["sweep"]:
        for a in ORDER:
            for lv in LEVELS:
                out.append(_case("sweep", m, {a: lv}, None))
    for set_name in SET_INDEX:
        cases = [c for c in out if c["set"] == set_name]
        ids = _rng(seed, set_name, _IDS).permutation(len(cases))
        for k, (c, i) in enumerate(zip(cases, ids)):
            c["k"] = k
            c["case_id"] = f"{PREFIX[set_name]}{int(i) + 1:04d}"
    return out


def _case(set_name: str, m: Member, recipe: dict, card) -> dict:
    return {"case_id": None, "set": set_name, "subject": m.id, "site": m.site, "native_sigma": float(m.native_sigma),
            "recipe": dict(recipe), "card": None if card is None else int(card), "k": None}
