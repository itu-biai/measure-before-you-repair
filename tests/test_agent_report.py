"""Task 4 Stage 2: the agent's final report, its provenance and the faithfulness check (D147, plan R2, R3)."""
import json

from agent_fakes import TABLE, ScriptedClient, marker_volume

from bamr.agent import functions as F
from bamr.agent.compute import Compute
from bamr.agent.loop import CaseLog, run_agent
from bamr.agent.report import check_report, provenance

ARTS = ("noise", "spike", "ghosting", "bias", "motion")


def _fin(version="v1", repairs=("fix_spike",), findings=None, done="v0 → spike repair → v1, stripes gone",
         artifacts=None, **text):
    f = {a: "absent: nothing seen" for a in ARTS}
    f["spike"] = "present: one bright point in k-space, marker 0.200"
    f |= findings or {}
    rep = {"examination": "3-D T1 volume as received (v0); measured, repaired, measured again", "findings": f,
           "what_was_done": done, "impression": "spike repaired", "recommendation": "no rescan",
           "limits": "no clean scan to compare"} | text
    art = {a: "no" for a in ARTS} | {"spike": "yes"} | (artifacts or {})
    return {"report": rep, "repairs": list(repairs), "artifacts": art, "rescan": "no",
            "rescan_why": "motion marker 0.000, below 0.033", "version": version}


def _run(tmp_path, script, x0=None):
    client = ScriptedClient(script)
    comp = Compute(tmp_path / "cache", workers=0, blocked=(), table=TABLE)
    log = CaseLog(tmp_path / "case")
    x0 = marker_volume(spike=0.2) if x0 is None else x0
    return run_agent(x0, client, comp, tmp_path / "work", log, ["d0042"]), client


def test_finish_schema():
    p = next(t for t in F.TOOLS if t["function"]["name"] == "finish")["function"]["parameters"]
    assert list(p["properties"]) == ["report", "repairs", "artifacts", "rescan", "rescan_why", "version"]
    r = p["properties"]["report"]
    assert list(r["properties"]) == ["examination", "findings", "what_was_done", "impression", "recommendation",
                                     "limits"]
    assert list(r["properties"]["findings"]["properties"]) == list(ARTS)
    assert p["properties"]["repairs"]["items"]["enum"] == ["fix_spike", "fix_ghosting", "fix_noise", "fix_bias"]
    assert F.check("finish", _fin(), {"v0", "v1"}) is None
    assert "repairs" in F.check("finish", _fin(repairs=("despike",)), {"v0", "v1"})
    assert "findings" in F.check("finish", dict(_fin(), report=dict(_fin()["report"], findings={})), {"v0", "v1"})


def test_faithful_run_passes(tmp_path):
    res, _ = _run(tmp_path, [[("spike_tool", {"version": "v0"}), ("fix_spike", {"version": "v0"})],
                             [("finish", _fin())]])
    c = res["check"]
    assert c["pass"] == {"a": True, "b": True, "c": True}, c
    assert res["report"]["repairs"] == ["fix_spike"] and "version" not in res["report"]
    assert [p["fn"] for p in res["provenance"]["calls"]] == ["spike_tool", "fix_spike", "finish"]
    assert res["provenance"]["calls"][1]["out"] == "v1"
    assert res["provenance"]["returned"] == ["v0", "v1"]
    saved = json.loads((tmp_path / "case" / "result.json").read_text())
    assert saved["check"] == c


def test_claimed_repairs_differ(tmp_path):
    res, _ = _run(tmp_path, [[("fix_spike", {"version": "v0"})],
                             [("finish", _fin(repairs=("fix_spike", "fix_bias")))]])
    assert res["check"]["pass"]["a"] is False and res["check"]["repairs"]["history"] == ["fix_spike"]


def test_repair_not_named_in_text(tmp_path):
    res, _ = _run(tmp_path, [[("fix_spike", {"version": "v0"})], [("finish", _fin(done="v0 → v1"))]])
    assert res["check"]["pass"]["a"] is False and res["check"]["repairs"]["unnamed_in_text"] == ["fix_spike"]


def test_number_not_in_log(tmp_path):
    res, _ = _run(tmp_path, [[("fix_spike", {"version": "v0"})], [("finish", _fin(impression="spike 87.5 % gone"))]])
    assert res["check"]["pass"]["b"] is False and res["check"]["numbers"]["unsupported"] == ["87.5 %"]


def test_findings_contradict_answers(tmp_path):
    fin = _fin(findings={"bias": "present: brighter at the front", "noise": "uncertain: grainy"})
    res, _ = _run(tmp_path, [[("fix_spike", {"version": "v0"})], [("finish", fin)]])
    c = res["check"]["findings"]
    assert res["check"]["pass"]["c"] is False and c["contradict"] == ["bias"] and c["uncertain"] == ["noise"]


def test_provenance_rows():
    trace = [{"fn": "noise_tool", "call": 1, "on": "v0", "finding": {"present": True, "value": 0.04, "why": "σ"}},
             {"fn": "fix_noise", "call": 2, "on": "v0", "result": "v1", "report": {"changed": True, "what": "denoised"}},
             {"fn": "look", "call": 3, "error": "unknown version 'v9'"}]
    rows = provenance(trace, [{"handle": "v0"}, {"handle": "v1"}])["calls"]
    assert rows[0]["reading"] == "yes 0.04" and rows[1]["out"] == "v1" and rows[1]["note"] == "denoised"
    assert rows[2]["note"].startswith("error")


def test_no_report_no_check():
    assert check_report(None, [], [], []) is None


def test_page(tmp_path):
    from bamr.agent.report import page
    res, client = _run(tmp_path, [[("spike_tool", {"version": "v0"}), ("fix_spike", {"version": "v0"})],
                                  [("finish", _fin())]])
    d = tmp_path / "case"
    assert (d / "pictures" / "returned_look.png").exists() and (d / "pictures" / "returned_vs_v0.png").exists()
    n_sent = sum(1 for m in client.seen[-1] if m["role"] == "user" for p in (m["content"] if isinstance(
        m["content"], list) else []) if p.get("type") == "image_url")
    assert n_sent == 2  # v0 and the spike tool's picture: the page's pictures are never sent to the model
    html = page(d).read_text()
    for h in ("Examination", "Findings", "What was done", "Impression", "Recommendation", "Limits", "Provenance",
              "Faithfulness check", "pictures/v0_look.png", "pictures/returned_look.png", "one bright point"):
        assert h in html, h
    assert html.count("✓") >= 3 and "<script" not in html


def test_numbers_tightened():
    """Review finding 1: a whole number must be given as a whole number; × 100 only next to a %."""
    from bamr.agent.report import quoted_with_unit, supported_quote
    received = ['{"value": 14.7, "z": 15.43, "share": 0.372, "stripes_mm": 4, "call": 13, "sigma": 0.0244}',
                "line 0.083; rescan at 0.033"]
    ok = lambda text: all(supported_quote(q, pct, received) for q, pct in quoted_with_unit(text))  # noqa: E731
    assert not ok("about 15 mm")  # 14.7 rounds to 15, but nothing whole was given as 15
    assert ok("13 calls") and ok("z 15.4") and ok("σ 2.44 %") and ok("line 0.083")
    assert not ok("a 37 improvement")  # 0.372 × 100 only counts next to a %
    assert ok("a 37 % share")
    assert not ok("0.5 of it")
