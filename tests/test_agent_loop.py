"""Task 4 Stage 1: the agent loop (D127, D135, D137, D139) with a scripted model and fake tools; no API call."""
import json
import re

import numpy as np
import pytest
from agent_fakes import TABLE, ScriptedClient, marker_volume

from bamr.agent.compute import Compute
from bamr.agent.loop import CaseLog, LeakError, run_agent
from bamr.agent.openrouter import ApiError
from bamr.agent.prompts import NUDGE, SYSTEM

FIN = {"report": {"examination": "v0 and v1", "findings": {"noise": "absent", "spike": "present: stripes",
                                                             "ghosting": "absent", "bias": "absent", "motion": "absent"},
                  "what_was_done": "spike repair → v1", "impression": "stripes gone", "recommendation": "none",
                  "limits": "none"},
       "repairs": ["fix_spike"],
       "artifacts": {"noise": "no", "spike": "yes", "ghosting": "no", "bias": "no", "motion": "no"},
       "rescan": "no", "rescan_why": "motion tool below 0.033", "version": "v1"}


def _run(tmp_path, script, x0=None, table=TABLE, forbidden=("d0042",), **kw):
    client = ScriptedClient(script, reasoning=kw.pop("reasoning", None))
    comp = Compute(tmp_path / "cache", workers=0, blocked=(), table=table)
    log = CaseLog(tmp_path / "case")
    x0 = marker_volume(spike=0.2) if x0 is None else x0
    res = run_agent(x0, client, comp, tmp_path / "work", log, list(forbidden), **kw)
    return res, client, log


def _texts(messages):
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            yield c
        elif isinstance(c, list):
            yield from (p["text"] for p in c if p.get("type") == "text")


def _tool_msgs(msgs):
    return [json.loads(m["content"]) for m in msgs if m["role"] == "tool"]


def test_happy_path_fix_and_finish(tmp_path):
    res, client, log = _run(tmp_path, [[("spike_tool", {"version": "v0"})], [("fix_spike", {"version": "v0"})],
                                       [("compare", {"a": "v1", "b": "v0"})], [("finish", FIN)]])
    assert res["version"] == "v1" and res["fused"] is False and res["error"] is None
    assert [(r["handle"], r["repair"]) for r in res["lineage"]] == [("v0", None), ("v1", "fix_spike")]
    assert res["report"] == {k: FIN[k] for k in ("report", "repairs", "artifacts", "rescan", "rescan_why")}
    assert (res["turns"], res["calls"], res["bad_calls"]) == (4, 4, 0)
    assert res["usage"]["cost"] == pytest.approx(0.004) and res["gen_ids"] == ["gen-1", "gen-2", "gen-3", "gen-4"]
    tm = _tool_msgs(client.seen[3])
    assert tm[0]["call"] == 1 and tm[0]["on"] == "v0" and tm[0]["finding"]["present"] is True
    assert "tool_version" in tm[0]["finding"] and "version" not in tm[0]["finding"]
    assert tm[0]["finding"]["picture"] == "sent in the next message"
    assert tm[1]["result"] == "v1" and tm[1]["report"]["changed"] is True
    assert "rms_change" in tm[2]
    d = tmp_path / "case"
    assert json.loads((d / "result.json").read_text())["version"] == "v1"
    assert (d / "output.nii.gz").exists()
    logged = (d / "messages.jsonl").read_text()
    assert "base64" not in logged and "pictures/" in logged
    sent = [q for q in (d / "pictures").glob("*.png") if not q.name.startswith("returned")]
    assert len(sent) == 3  # v0, the spike tool's, the compare (returned_* are for the page, never sent)
    assert len((d / "api.jsonl").read_text().splitlines()) == 4


def test_first_message_and_picture_follows_tool_message(tmp_path):
    _, client, _ = _run(tmp_path, [[("spike_tool", {"version": "v0"})], [("finish", dict(FIN, version="v0"))]])
    first = client.seen[0]
    assert first[0] == {"role": "system", "content": SYSTEM}
    assert first[1]["role"] == "user" and first[1]["content"][1]["type"] == "image_url"
    m = client.seen[1]
    i = next(k for k, x in enumerate(m) if x["role"] == "tool")
    assert isinstance(m[i]["content"], str) and "image" not in m[i]["content"]
    nxt = m[i + 1]
    assert nxt["role"] == "user" and nxt["content"][0] == {"type": "text", "text": "picture for call 1"}
    assert nxt["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert len(nxt["content"]) == 2 and i + 2 == len(m)


def test_no_leak(tmp_path):
    _, client, _ = _run(tmp_path, [[("spike_tool", {"version": "v0"})], [("fix_spike", {"version": "v0"})],
                                   [("look", {"version": "v1", "plane": None, "slice": None, "window": "air"})], [("finish", FIN)]],
                        forbidden=("d0042", str(tmp_path)))
    text = "\n".join(t for msgs in client.seen for t in _texts(msgs))
    assert "d0042" not in text and str(tmp_path) not in text and ".nii" not in text
    assert not re.search(r"\b[dst]\d{4}\b|/Users/|\b(dev|test)\b", text)


def test_leak_raises(tmp_path):
    table = dict(TABLE, noise_tool=("tool", "agent_fakes:leaky_tool"))
    with pytest.raises(LeakError):
        _run(tmp_path, [[("noise_tool", {"version": "v0"})], [("finish", dict(FIN, version="v0"))]], table=table)


def test_bad_calls_return_errors(tmp_path):
    res, client, _ = _run(tmp_path, [[("despike", {"version": "v0"})], [("spike_tool", "{not json")],
                                     [("spike_tool", {"version": "v5"})],
                                     [("look", {"version": "v0", "plane": "axial", "slice": None, "window": None, "zoom": 2})],
                                     [("finish", dict(FIN, version="v0"))]])
    errs = [m["error"] for m in _tool_msgs(client.seen[-1])]
    assert "unknown function" in errs[0] and "JSON" in errs[1] and "v5" in errs[2] and "unexpected" in errs[3]
    assert res["bad_calls"] == 4 and res["calls"] == 5 and res["version"] == "v0"


def test_two_calls_one_turn(tmp_path):
    """D142: every call of a turn runs, in order; each tool_call id is answered."""
    res, client, _ = _run(tmp_path, [[("spike_tool", {"version": "v0"}), ("noise_tool", {"version": "v0"})],
                                     [("finish", dict(FIN, version="v0"))]])
    m = client.seen[1]
    ids = [c["id"] for c in next(x for x in m if x["role"] == "assistant")["tool_calls"]]
    tools = [json.loads(x["content"]) for x in m if x["role"] == "tool"]
    assert [x["tool_call_id"] for x in m if x["role"] == "tool"] == ids
    assert [t["call"] for t in tools] == [1, 2] and all("finding" in t for t in tools)
    assert res["bad_calls"] == 0 and res["calls"] == 3  # the two, then finish


def test_chained_handles_in_one_turn(tmp_path):
    """A call may name a version an earlier call of the same turn made."""
    res, client, _ = _run(tmp_path, [[("fix_spike", {"version": "v0"}), ("fix_noise", {"version": "v1"})],
                                     [("finish", dict(FIN, version="v2"))]],
                          x0=marker_volume(spike=0.2, noise=0.05))
    tm = _tool_msgs(client.seen[1])
    assert tm[0]["result"] == "v1" and tm[1]["on"] == "v1" and tm[1]["result"] == "v2"
    assert [r["repair"] for r in res["lineage"]] == [None, "fix_spike", "fix_noise"] and res["bad_calls"] == 0


def test_finish_with_others_refused(tmp_path):
    """finish ends the run only on its own; the other calls of that turn still run."""
    res, client, _ = _run(tmp_path, [[("fix_spike", {"version": "v0"}), ("finish", FIN)],
                                     [("finish", FIN)]])
    tm = _tool_msgs(client.seen[1])
    assert tm[0]["result"] == "v1" and "on its own" in tm[1]["error"]
    assert res["version"] == "v1" and res["turns"] == 2 and res["bad_calls"] == 1


def test_pictures_one_message_after_tools(tmp_path):
    """All pictures of a turn: one user message after the tool messages, each labelled with its call."""
    _, client, _ = _run(tmp_path, [[("spike_tool", {"version": "v0"}), ("noise_tool", {"version": "v0"})],
                                   [("finish", dict(FIN, version="v0"))]])
    m = client.seen[1]
    last = m[-1]
    assert m[-2]["role"] == "tool" and m[-3]["role"] == "tool" and last["role"] == "user"
    kinds = [(p["type"], p.get("text")) for p in last["content"]]
    assert kinds == [("text", "picture for call 1"), ("image_url", None), ("text", "picture for call 2"),
                     ("image_url", None)]


def test_text_only_nudged(tmp_path):
    res, client, _ = _run(tmp_path, ["It looks fine to me.", "", [("finish", dict(FIN, version="v0"))]])
    texts = list(_texts(client.seen[2]))
    assert texts.count(NUDGE) == 2 and "It looks fine to me." in texts
    assert not any(m["role"] == "assistant" and not m.get("content") and not m.get("tool_calls")
                   for m in client.seen[2])  # an empty reply is not sent back
    assert res["bad_calls"] == 2 and res["version"] == "v0" and res["fused"] is False


def test_fuse_40(tmp_path):
    res, client, _ = _run(tmp_path, [[("fix_spike", {"version": "v0"})]] + [[("spike_tool", {"version": "v1"})]] * 99)
    assert res["calls"] == 40 and res["fused"] is True and res["version"] == "v0" and res["report"] is None
    assert len(client.seen) == 40
    assert (tmp_path / "case" / "output.nii.gz").exists()


def test_fuse_time(tmp_path):
    t = iter([0.0, 10.0, 11.0, 2000.0, 2001.0, 2002.0])  # start, top of turn 1, before its call, top of turn 2
    res, client, _ = _run(tmp_path, [[("spike_tool", {"version": "v0"})], [("finish", dict(FIN, version="v0"))]],
                          clock=lambda: next(t))
    assert res["fused"] is True and res["calls"] == 1 and len(client.seen) == 1


def test_gated_fix_same_handle(tmp_path):
    res, client, _ = _run(tmp_path, [[("fix_bias", {"version": "v0"})], [("finish", dict(FIN, version="v0"))]])
    tm = _tool_msgs(client.seen[1])[0]
    assert tm["result"] == "v0" and tm["report"]["changed"] is False
    assert [r["handle"] for r in res["lineage"]] == ["v0"]


def test_api_error_returns_v0(tmp_path):
    res, _, log = _run(tmp_path, [[("fix_spike", {"version": "v0"})], ApiError(400, "bad request")])
    assert res["version"] == "v0" and "400" in res["error"] and res["fused"] is False
    assert (tmp_path / "case" / "output.nii.gz").exists()


def test_reasoning_details_passed_back(tmp_path):
    _, client, _ = _run(tmp_path, [[("spike_tool", {"version": "v0"})], [("finish", dict(FIN, version="v0"))]],
                        reasoning="thought")
    sent = [m for m in client.seen[1] if m["role"] == "assistant"][0]
    assert sent["reasoning_details"] == [{"type": "reasoning.text", "text": "thought 1"}]
    assert sent["tool_calls"][0]["function"]["name"] == "spike_tool"


def test_output_is_returned_version(tmp_path):
    import nibabel as nib
    res, _, _ = _run(tmp_path, [[("fix_spike", {"version": "v0"})], [("finish", FIN)]])
    y = np.asarray(nib.load(tmp_path / "case" / "output.nii.gz").dataobj)
    assert float(y[1, 0, 0]) == 0.0 and y.dtype == np.float32


def test_fuse_inside_one_turn(tmp_path):
    """D127 with D142: one turn of 45 calls runs 40; the rest are answered with an error, not run; the run is fused."""
    res, client, log = _run(tmp_path, [[("spike_tool", {"version": "v0"})] * 45, [("finish", FIN)]])
    assert res["fused"] is True and res["calls"] == 40 and res["version"] == "v0" and len(client.seen) == 1
    msgs = [json.loads(line) for line in (tmp_path / "case" / "messages.jsonl").read_text().splitlines()]
    tools = [json.loads(m["content"]) for m in msgs if m["role"] == "tool"]
    assert len(tools) == 45 and sum("not run" in t.get("error", "") for t in tools) == 5


def test_eyes_only(tmp_path):
    """④a (D150): look, compare, the four repairs and finish; a repair's reply carries no reading."""
    from bamr.agent.prompts import SYSTEM_EYES
    res, client, _ = _run(tmp_path, [[("noise_tool", {"version": "v0"}), ("fix_spike", {"version": "v0"}),
                                      ("fix_bias", {"version": "v1"})], [("finish", FIN)]], eyes=True)
    names = [t["function"]["name"] for t in client.tools]
    assert names == ["look", "compare", "fix_spike", "fix_ghosting", "fix_noise", "fix_bias", "finish"]
    assert client.seen[0][0] == {"role": "system", "content": SYSTEM_EYES}
    tm = _tool_msgs(client.seen[1])
    assert "unknown function" in tm[0]["error"]
    assert tm[1] == {"call": 2, "on": "v0", "result": "v1"}
    assert tm[2] == {"call": 3, "on": "v1", "result": "v1", "unchanged": True}
    assert res["system"] == "agent_eyes" and res["version"] == "v1" and res["check"]["pass"]["a"] is True
    for m in client.seen[1]:
        if m["role"] == "tool":
            assert not any(w in m["content"] for w in ('"present"', '"value"', '"why"', '"report"'))
