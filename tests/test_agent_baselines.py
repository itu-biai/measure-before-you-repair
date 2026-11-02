"""Task 4 Stage 1: the fixed chain ② and the one-step router ③ (D138), on fake tools; no API call."""
import pytest
from agent_fakes import TABLE, ScriptedClient, marker_volume

from bamr.agent.baselines import CHAIN, run_chain, run_router
from bamr.agent.compute import Compute
from bamr.agent.loop import CaseLog
from bamr.agent.prompts import ROUTER


@pytest.fixture
def fixlog(tmp_path, monkeypatch):
    p = tmp_path / "fixes.log"
    monkeypatch.setenv("BAMR_FAKE_FIXLOG", str(p))
    return lambda: p.read_text().split() if p.exists() else []


def _comp(tmp_path):
    return Compute(tmp_path / "cache", workers=0, blocked=(), table=TABLE)


def _chain(tmp_path, x0):
    return run_chain(x0, _comp(tmp_path), tmp_path / "work", CaseLog(tmp_path / "case"))


def test_chain_order_and_gates(tmp_path, fixlog):
    assert CHAIN == ("fix_spike", "fix_ghosting", "fix_noise", "fix_bias")
    res = _chain(tmp_path, marker_volume(noise=0.04, spike=0.2, bias=0.1))
    assert fixlog() == ["spike", "noise", "bias"]
    assert [(r["handle"], r["repair"]) for r in res["lineage"]] == [("v0", None), ("v1", "fix_spike"),
                                                                    ("v2", "fix_noise"), ("v3", "fix_bias")]
    assert res["version"] == "v3" and res["system"] == "chain" and res["model"] is None
    assert res["report"]["artifacts"] == {"noise": "yes", "spike": "yes", "ghosting": "no", "bias": "yes",
                                          "motion": "no"}
    assert (tmp_path / "case" / "output.nii.gz").exists() and (tmp_path / "case" / "result.json").exists()


def test_chain_full_order(tmp_path, fixlog):
    _chain(tmp_path, marker_volume(noise=0.04, spike=0.2, ghosting=0.1, bias=0.1))
    assert fixlog() == ["spike", "ghosting", "noise", "bias"]


def test_chain_report_cannot_tell_is_no(tmp_path, monkeypatch):
    monkeypatch.setenv("BAMR_FAKE_MOTION_NONE", "1")
    res = _chain(tmp_path, marker_volume(motion=0.06))
    assert res["report"]["artifacts"]["motion"] == "no" and res["report"]["rescan"] == "no"
    assert "cannot tell" in res["report"]["seen"]


def test_chain_rescan_from_output(tmp_path):
    res = _chain(tmp_path, marker_volume(spike=0.2))  # the spike repair raises the motion marker to 0.05 (as d0007)
    assert res["report"]["artifacts"]["motion"] == "no"  # on v0
    assert res["report"]["rescan"] == "yes" and "0.05" in res["report"]["rescan_why"]


def _ans(**yes):
    return {"seen": "I see stripes and shading.",
            "artifacts": {a: ("yes" if a in yes else "no") for a in ("noise", "spike", "ghosting", "bias", "motion")},
            "rescan": "no", "rescan_why": "no motion seen"}


def _router(tmp_path, script, x0):
    client = ScriptedClient(script)
    res = run_router(x0, client, _comp(tmp_path), tmp_path / "work", CaseLog(tmp_path / "case"), ["d0042"])
    return res, client


def test_router_runs_named_fixes_in_chain_order(tmp_path, fixlog):
    res, client = _router(tmp_path, [[("answer", _ans(bias=1, spike=1))]],
                          marker_volume(noise=0.04, spike=0.2, bias=0.1))
    assert fixlog() == ["spike", "bias"]
    assert res["report"]["artifacts"]["bias"] == "yes" and res["report"]["seen"].startswith("I see")
    assert res["named"] == ["fix_spike", "fix_bias"] and res["unparsed"] is False
    assert (res["turns"], res["calls"], res["system"]) == (1, 1, "router")
    m = client.seen[0]
    assert len(m) == 1 and m[0]["role"] == "user" and m[0]["content"][0] == {"type": "text", "text": ROUTER}
    assert m[0]["content"][1]["type"] == "image_url"


def test_router_named_fix_still_gated(tmp_path, fixlog):
    res, _ = _router(tmp_path, [[("answer", _ans(ghosting=1, motion=1))]], marker_volume(spike=0.2))
    assert fixlog() == [] and res["version"] == "v0" and res["report"]["artifacts"]["motion"] == "yes"


def test_router_parses_text(tmp_path, fixlog):
    text = "Stripes across the slices.\nNoise: no\nSpikes: yes\nGhosting: no\nBias: no\nMotion: no\nRescan: no"
    res, _ = _router(tmp_path, [text], marker_volume(spike=0.2))
    assert res["unparsed"] is False and res["report"]["artifacts"]["spike"] == "yes" and fixlog() == ["spike"]
    assert res["report"]["seen"] == text


def test_router_unparsed(tmp_path, fixlog):
    res, _ = _router(tmp_path, ["Looks fine."], marker_volume(spike=0.2))
    assert res["unparsed"] is True and res["version"] == "v0" and fixlog() == []
    assert set(res["report"]["artifacts"].values()) == {"no"}


def test_router_with_readings(tmp_path, fixlog):
    """③b (D148): one call that sees v0 and the five tools' findings, cards and pictures; named fixes gated, in order."""
    from bamr.agent.prompts import CARDS, ROUTER_READINGS
    client = ScriptedClient([[("answer", _ans(bias=1, noise=1))]])
    res = run_router(marker_volume(noise=0.04, bias=0.1), client, _comp(tmp_path), tmp_path / "work",
                     CaseLog(tmp_path / "case"), ["d0042"], readings=True)
    assert fixlog() == ["noise", "bias"] and res["system"] == "router_readings" and res["calls"] == 1
    m = client.seen[0]
    assert len(m) == 1
    c = m[0]["content"]
    assert c[0] == {"type": "text", "text": ROUTER_READINGS}
    assert sum(p["type"] == "image_url" for p in c) == 6
    texts = [p["text"] for p in c if p["type"] == "text"][1:]
    assert len(texts) == 5
    for a, t in zip(("noise", "spike", "ghosting", "bias", "motion"), texts):
        assert t.startswith(f"{a}_tool: {CARDS[a + '_tool']}") and '"present": ' in t and '"tool_version"' in t
    assert "0.033" in ROUTER_READINGS and "rescan" in ROUTER_READINGS
