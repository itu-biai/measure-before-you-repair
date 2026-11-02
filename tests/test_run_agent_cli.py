"""Task 4 Stage 2: the runner's case lists (D144) and refusals; the check columns in the score rows (D147)."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from bamr.eval.score_agent import case_row, summary

REPO = Path(__file__).resolve().parents[1]
S2 = REPO.parent / "visual-guide" / "task4" / "02_develop" / "before_numbers.json"


def _script():
    spec = importlib.util.spec_from_file_location("run_agent_cli", REPO / "scripts" / "run_agent.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.mark.skipif(not S2.exists(), reason="the visual guide is not next to the repo")
def test_case_lists():
    m, b = _script(), json.loads(S2.read_text())
    assert m.DEVELOP == b["develop"] and m.HELD_OUT == b["held_out"]
    assert m.cases("develop") == b["develop"] and m.cases("heldout") == b["held_out"] and len(m.cases("check")) == 10
    assert not set(m.DEVELOP) & set(m.HELD_OUT)
    assert m.cases("d0001,t0004") == ["d0001", "t0004"]  # Task 5 opened the test set (30 Oct)
    for bad in ("x0001", "../refs", "t04"):
        with pytest.raises(SystemExit):
            m.cases(bad)


def test_test_cases_from_the_inputs(tmp_path, monkeypatch):
    m = _script()
    (tmp_path / "inputs").mkdir()
    for c in ("t0002", "d0001", "t0001", "s0001"):
        (tmp_path / "inputs" / f"{c}.nii.gz").write_bytes(b"")
    monkeypatch.setattr(m, "DS", tmp_path)
    assert m.cases("test") == ["t0001", "t0002"]  # names only: no file is opened


def test_systems():
    m = _script()
    assert m.SYSTEMS == ("chain", "chain_u", "router", "router_readings", "agent", "agent_eyes")


def test_check_columns():
    x = np.ones((4, 4, 4), np.float32)
    lab = {"artifacts": ["spike"], "params": {}}
    res = {"report": {"artifacts": {"spike": "yes"}, "rescan": "no"},
           "check": {"pass": {"a": True, "b": False, "c": True}}, "lineage": [{"handle": "v0"}], "seconds": 1.0}
    r = case_row(x, x, x * 0.9, x > 0, lab, res)
    assert (r["check_a"], r["check_b"], r["check_c"]) == (True, False, True)
    r2 = case_row(x, x, x * 0.9, x > 0, lab, {"report": None, "lineage": [{"handle": "v0"}], "seconds": 1.0})
    assert r2["check_a"] is None
    s = summary([r, r2])
    assert s["checked"] == 1 and s["check_pass"] == {"a": 1, "b": 0, "c": 1, "all": 0}


def test_network_failure_is_not_a_result(tmp_path):
    """D165: a case that ended on our own network (OpenRouter 0: transport error after the retries) is set aside, not
    kept as the model's answer, so the next run does it again; any reply from the server stays a result."""
    m = _script()
    net = {"error": "OpenRouter 0: ConnectError: [Errno 8] nodename nor servname provided, or not known"}
    assert m.network_failure(net)
    assert not m.network_failure({"error": "OpenRouter 400: {'message': 'bad tool'}"})
    assert not m.network_failure({"error": None})
    for case, res, kept in (("t0001", net, False), ("t0002", {"error": None}, True)):
        d = tmp_path / "agent-qwen" / case
        d.mkdir(parents=True)
        (d / "result.json").write_text("{}")
        assert m.set_aside(d, res) is (not kept)
        assert (d / "result.json").exists() is kept
    moved = tmp_path / "_network" / "agent-qwen" / "t0001-1"  # kept with its log and cost, out of the scored folders
    assert (moved / "result.network.json").exists() and not (moved / "result.json").exists()
    d = tmp_path / "agent-qwen" / "t0001"
    d.mkdir()
    (d / "result.json").write_text("{}")
    m.set_aside(d, net)  # a second failure of the same case is kept too
    assert (tmp_path / "_network" / "agent-qwen" / "t0001-2" / "result.network.json").exists()


def test_rate_limit_is_not_a_result(tmp_path):
    """D167: a case that ended on OpenRouter's rate limit for our account (429 after the retries) is set aside like a
    network failure, to <out>/_ratelimit/, so it runs again; other server replies stay results."""
    m = _script()
    rl = {"error": 'OpenRouter 429: {"error":{"message":"Rate limit exceeded: new-account-rpm/openai/gpt-6.1-terra"}}'}
    assert m.rate_limited(rl) and not m.network_failure(rl)
    assert not m.rate_limited({"error": "OpenRouter 0: ConnectError"})
    assert not m.rate_limited({"error": "OpenRouter 400: bad tool"}) and not m.rate_limited({"error": None})
    d = tmp_path / "agent-terra" / "t0018"
    d.mkdir(parents=True)
    (d / "result.json").write_text("{}")
    assert m.set_aside(d, rl)
    moved = tmp_path / "_ratelimit" / "agent-terra" / "t0018-1"
    assert (moved / "result.network.json").exists() and not d.exists()
    assert not (tmp_path / "_network").exists()


def test_score_cli_workers():
    """D166: the oracle's compute workers are set on the command line (1 keeps Task 5 inside the 8 GB rule)."""
    spec = importlib.util.spec_from_file_location("score_agent_cli", REPO / "scripts" / "score_agent.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    a = m.parser().parse_args(["--out", "stage5-test", "--oracle", "--workers", "1"])
    assert (a.out, a.oracle, a.workers) == ("stage5-test", True, 1)
    assert m.parser().parse_args([]).workers == 2
