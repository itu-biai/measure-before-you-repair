"""Task 4 (D159): the report's paper view: a short card per run, readable without our labels. Versions named by what
was done, the model's words kept (labels replaced), the repair and the check on the final scan taken from the log."""
import json
import re

from bamr.agent.paper_view import card, plain, rows, version_names

ARTS = ("noise", "spike", "ghosting", "bias", "motion")


def _lineage(*steps):
    out, parent = [{"handle": "v0", "parent": None, "repair": None}], "v0"
    for h, fix in steps:
        out.append({"handle": h, "parent": parent, "repair": fix})
        parent = h
    return out


def _call(n, fn, on, out=None, reading=None):
    return {"call": n, "fn": fn, "on": on, "out": out, "reading": reading, "note": None}


CALLS = [_call(1, "noise_tool", "v0", reading="yes 0.044"), _call(2, "spike_tool", "v0", reading="yes 0.014"),
         _call(3, "bias_tool", "v0", reading="no 0.12"), _call(4, "fix_spike", "v0", "v1"),
         _call(5, "fix_bias", "v1", "v4"), _call(6, "fix_ghosting", "v1"), _call(7, "fix_noise", "v1", "v2"),
         _call(8, "spike_tool", "v2", reading="no 0.0"), _call(9, "motion_tool", "v2", reading="cannot tell None"),
         _call(10, "finish", None)]
LINEAGE = _lineage(("v1", "fix_spike"), ("v2", "fix_noise"))


def _res(**kw):
    find = {a: "Absent. Nothing seen." for a in ARTS} | {
        "noise": "Present. Sigma 4.4% of WM on v0 (line 3.0%). After fix_noise it reads 1.7% on v2.",
        "spike": "Present, marginal. One bright point on v0. After denoising none is left on v2.",
        "motion": "Uncertain. The motion tool could not tell on v2."}
    rep = {"examination": "Measured v0 with all tools.", "findings": find,
           "what_was_done": "v0 to v1: fix_spike. v1 to v2: fix_noise.",
           "impression": "v2 is clean; v4 was not kept.", "recommendation": "Review v2. Compare with v0 first.",
           "limits": "none"}
    return {"system": "agent", "model": "anthropic/claude-haiku-5.5", "version": "v2", "lineage": LINEAGE,
            "report": {"report": rep, "repairs": ["fix_spike", "fix_noise"], "rescan": "no",
                       "rescan_why": "motion_tool could not tell on v2",
                       "artifacts": {a: "no" for a in ARTS} | {"noise": "yes", "spike": "yes"}},
            "provenance": {"calls": CALLS, "returned": ["v0", "v1", "v2"]},
            "check": {"pass": {"a": True, "b": True, "c": False}}, **kw}


def test_version_names_follow_what_was_done():
    n = version_names(CALLS, LINEAGE)
    assert n == {"v0": "the original scan", "v1": "the scan after spike removal",
                 "v2": "the final scan", "v4": "the scan after spike removal and bias correction"}
    assert version_names([], _lineage())["v0"] == "the original scan"  # nothing kept: original = final


def test_plain_has_no_labels_or_function_names():
    n = version_names(CALLS, LINEAGE)
    assert plain("v2 is clean; v4 was not kept.", n) == \
        "The final scan is clean; the scan after spike removal and bias correction was not kept."
    assert plain("After fix_noise it reads 1.7% of WM on v2's brain. motion_tool: none.", n) == \
        "After denoising it reads 1.7% of white matter on the final scan's brain. The motion tool: none."
    assert plain("Compared v9 with v0.", n) == "Compared an earlier version with the original scan."
    # the model's own noun before a label, a label alone in brackets, a function name in brackets: no doubling
    assert plain("Review the restored volume v2, then check it against the original v0.", n) == \
        "Review the final scan, then check it against the original scan."
    assert plain("Repairs in order: spike (v1), noise (v2). Version v4 was dropped.", n) == \
        "Repairs in order: spike, noise. The scan after spike removal and bias correction was dropped."
    assert plain("After denoising (fix_noise, DRUNet) and despiking (fix_spike).", n) == \
        "After denoising (DRUNet) and despiking."
    assert plain("Check against the raw v0 and the unfiltered v1. fix_spike v0 unchanged.", n) == \
        "Check against the original scan and the scan after spike removal. Spike removal on the original scan " \
        "unchanged."
    assert plain("A-P leak, A–P line energy, along A-P.", n) == "front–back leak, front–back line energy, along " \
                                                              "front–back."


def test_rows_answer_repair_and_final_check():
    r = {x["artifact"]: x for x in rows(_res())}
    assert [x["artifact"] for x in rows(_res())] == list(ARTS)
    assert (r["noise"]["answer"], r["noise"]["repair"], r["noise"]["final"]) == ("found", "step 2: denoising",
                                                                                "measured by its repair")
    assert (r["spike"]["answer"], r["spike"]["repair"], r["spike"]["final"]) == ("found", "step 1: spike removal",
                                                                                "clear")
    assert (r["bias"]["answer"], r["bias"]["repair"], r["bias"]["final"]) == ("not found", None,
                                                                              "not measured on the final scan")
    assert (r["motion"]["answer"], r["motion"]["final"]) == ("uncertain", "cannot tell")
    assert r["motion"]["repair"] == "no repair exists: the rescan decision"
    assert r["noise"]["note"] == "Sigma 4.4% of white matter on the original scan (line 3.0%). After denoising it " \
                                 "reads 1.7% on the final scan."
    assert r["spike"]["note"].startswith("Marginal. One bright point on the original scan.")  # the lead's qualifier kept
    long = _res()
    long["report"]["report"]["findings"]["bias"] = "Absent. " + " ".join(f"Sentence {i} on v0 is here." for i in range(9))
    assert rows(long)[3]["note"] == "Sentence 0 on the original scan is here. Sentence 1 on the original scan is here. …"


def test_card_reads_without_labels(tmp_path):
    d = tmp_path / "d0042"
    d.mkdir()
    (d / "result.json").write_text(json.dumps(_res()))
    html = card(d).read_text()
    text = re.sub(r"<[^>]+>", " ", html)
    assert not re.search(r"\bv\d+\b|\bfix_\w+|_tool\b|\bWM\b", text), re.findall(r"\bv\d+\b|fix_\w+|\w+_tool", text)
    for s in ("the original scan", "Rescan: no", "Review the final scan.", "spike removal", "denoising",
              "not measured on the final scan", "✗"):
        assert s in text, s
    assert (d / "card.html").exists()


def test_card_without_report(tmp_path):
    d = tmp_path / "d0043"
    d.mkdir()
    (d / "result.json").write_text(json.dumps(_res(report=None, check=None, fused=True)))
    text = re.sub(r"<[^>]+>", " ", card(d).read_text())
    assert "no report" in text and "fuse" in text
