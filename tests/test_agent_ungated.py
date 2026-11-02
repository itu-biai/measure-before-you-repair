"""Task 4 Stage 2: ②u, every repair ungated (D152), and ⑤, the oracle (D151), on the Task 3 repairs' own steps."""
import numpy as np
import pytest
from agent_fakes import TABLE, marker_volume
from phantoms import head_phantom, magnitude

from bamr.agent import ungated as U
from bamr.agent.baselines import run_chain
from bamr.agent.compute import Compute
from bamr.agent.loop import CaseLog
from bamr.data.artifacts3d import to_t
from bamr.tools.finding import PROTOCOL_V2
from bamr.tools.noise import LINE, noise_tool


@pytest.fixture
def fixlog(tmp_path, monkeypatch):
    p = tmp_path / "fixes.log"
    monkeypatch.setenv("BAMR_FAKE_FIXLOG", str(p))
    return lambda: p.read_text().split() if p.exists() else []


def _comp(tmp_path):
    return Compute(tmp_path / "cache", workers=0, blocked=(), table=TABLE)


def test_noise_u_denoises_below_the_line(monkeypatch):
    x = magnitude(head_phantom((48, 60, 60)), sigma=0.01)
    f = noise_tool(x, PROTOCOL_V2)
    assert f.present is False and f.value < LINE  # the gated fix would return it unchanged
    seen = {}

    def fake(a, s, device=None):
        seen["sigma"] = s
        return a * 0.5
    monkeypatch.setattr("bamr.fixes.noise.denoise3", fake)
    y, rep = U.noise_u(x, PROTOCOL_V2)
    assert rep.changed and seen["sigma"] == f.value and rep.evidence["source"] == "tool"
    y2, rep2 = U.noise_u(x, PROTOCOL_V2, sigma=0.05)  # the oracle's fallback is used only when the tool has no σ
    assert seen["sigma"] == f.value


def test_bias_u_divides_below_the_line(monkeypatch):
    x = head_phantom().astype(np.float32)  # ≈ 320 ml: above the fix's 100 ml floor
    brain = x > 0
    field = np.zeros(x.shape) + 0.01 * np.linspace(-1, 1, x.shape[0])[:, None, None]  # max/min − 1 ≈ 0.02 < 0.083
    monkeypatch.setattr(U, "brain_mask", lambda a, p: brain)
    monkeypatch.setattr(U, "n4_logfield", lambda a, m, v: field)
    y, rep = U.bias_u(x, PROTOCOL_V2)
    assert rep.changed and rep.evidence["reading_before"] < 0.083
    m = brain & (x > 0)
    assert np.allclose(y, x / np.exp(field - field[m].mean()), atol=1e-6)


def test_label_spike_positions_to_k():
    shape = (192, 240, 240)  # array (LR, AP, SI)
    st = np.array(to_t(np.zeros(shape, np.float32)).shape)  # TorchIO (LR, SI, AP)
    offsets = [(12, -30, 7), (-40, 5, 22)]  # in TorchIO axes
    positions = ((st // 2 + np.array(offsets) + 0.5) / st).tolist()
    ks = U.label_ks({"spikes_positions": positions}, shape)
    assert ks == [[12, 7, -30], [-40, 22, 5]]  # array axes: LR, AP, SI


def test_label_fallbacks():
    lab = {"artifacts": ["noise", "ghosting", "motion"],
           "params": {"noise": {"sigma_total": 0.05}, "ghosting": {"num_ghosts": 3}, "motion": {}}}
    fb = U.label_fallbacks(lab, (8, 8, 8))
    assert fb == {"noise": {"sigma": 0.05}, "ghosting": {"n": 3}}


def test_chain_ungated_runs_every_repair(tmp_path, fixlog):
    res = run_chain(marker_volume(), _comp(tmp_path), tmp_path / "work", CaseLog(tmp_path / "case"), mode="ungated")
    assert fixlog() == ["spike_u", "ghosting_u", "noise_u", "bias_u"] and res["system"] == "chain_u"


def test_oracle_runs_labelled_repairs_only(tmp_path, fixlog):
    lab = {"artifacts": ["noise", "bias", "motion"], "params": {"noise": {"sigma_total": 0.05}, "motion": {}}}
    res = run_chain(marker_volume(), _comp(tmp_path), tmp_path / "work", CaseLog(tmp_path / "case"), mode="oracle",
                    label=lab)
    assert fixlog() == ["noise_u", "bias_u"] and res["system"] == "oracle"
    assert res["report"]["artifacts"] == {"noise": "yes", "spike": "no", "ghosting": "no", "bias": "yes",
                                          "motion": "yes"}


def test_oracle_nothing_labelled_returns_v0(tmp_path, fixlog):
    res = run_chain(marker_volume(noise=0.05), _comp(tmp_path), tmp_path / "work", CaseLog(tmp_path / "case"),
                    mode="oracle", label={"artifacts": [], "params": {}})
    assert fixlog() == [] and res["version"] == "v0" and res["steps"] == []


def test_oracle_prefers_label_period_when_tool_is_off_whole(monkeypatch):
    """Review finding 6: a period off a whole number is not a measurement the fix can use; ⑤ takes the label's."""
    from bamr.tools.finding import Finding
    seen = {}
    monkeypatch.setattr(U.tg, "ghosting_tool", lambda x, p: Finding("ghosting", 1, "ghosting", True, 0.01, "u", "high",
                                                                  "rhythm", {"period_lines": 3.4}))
    monkeypatch.setattr(U.fg, "_fix", lambda x, k, why, ev, p: (seen.update(k=k, source=ev["source"]), (x, None))[1])
    U.ghosting_u(np.zeros((4, 4, 4), np.float32), PROTOCOL_V2, n=4)
    assert seen == {"k": 4, "source": "label"}
    U.ghosting_u(np.zeros((4, 4, 4), np.float32), PROTOCOL_V2)  # ②u (no label): the tool's period, rounded
    assert seen == {"k": 3, "source": "tool"}
