"""Task 3 Stage 1: the D115 scoring rules on small synthetic tables (no dataset file is read)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from bamr.eval.score_fixes import TARGETS, score_fix


def table(fix="spike", removed=0.98, no_after=30, n_alone=32, n_mixed=4, n_none=10, seconds=13.0):
    rows = []
    for i in range(n_alone):
        rows.append(dict(case_id=f"s{i:04d}", artifacts=fix, has=True, changed=True, identical=False, own_in=0.03,
                         own_out=0.03 * (1 - removed), after_present=i >= no_after, seconds=seconds, zeros_kept=True,
                         own_air_in=0.04, own_air_out=0.002, n_true=1, n_repaired=1))
    for i in range(n_mixed):
        rows.append(dict(case_id=f"d{i:04d}", artifacts=f"{fix}+noise", has=True, changed=True, identical=False,
                         own_in=0.02, own_out=0.002, after_present=False, seconds=seconds, zeros_kept=True,
                         own_air_in=0.02, own_air_out=0.015, n_true=2, n_repaired=2))
    for i in range(n_none):
        rows.append(dict(case_id=f"s{100 + i:04d}", artifacts="noise", has=False, changed=False, identical=True,
                         own_in=np.nan, own_out=np.nan, after_present=False, seconds=4.0, zeros_kept=True,
                         own_air_in=np.nan, own_air_out=np.nan, n_true=0, n_repaired=0))
    return pd.DataFrame(rows)


def test_spike_targets_pass():
    s = score_fix(table(), "spike")
    assert all(t["pass"] for t in s["targets"].values()), s["targets"]
    assert s["alone"]["n"] == 32 and abs(s["alone"]["removed_median"] - 0.98) < 1e-9
    assert s["mixed"]["n"] == 4 and s["no_harm"]["n"] == 10


def test_spike_target_fails_on_removed_and_still_yes():
    s = score_fix(table(removed=0.90, no_after=28), "spike")
    assert s["targets"]["removed_median"]["pass"] is False and s["targets"]["no_after"]["pass"] is False


def test_spike_target_fails_on_worse():
    d = table()
    d.loc[0, "own_out"] = 0.05
    assert score_fix(d, "spike")["targets"]["worse"]["pass"] is False


def test_spike_target_fails_on_seconds():
    assert score_fix(table(seconds=61.0), "spike")["targets"]["seconds_max"]["pass"] is False


def test_no_harm_requires_identical():
    d = table()
    d.loc[d.has == False, "identical"] = [False] + [True] * 9  # noqa: E712
    assert score_fix(d, "spike")["targets"]["no_harm"]["pass"] is False


def test_unchanged_must_be_identical():
    d = table()
    d.loc[0, ["changed", "identical"]] = [False, False]
    assert score_fix(d, "spike")["targets"]["unchanged_identical"]["pass"] is False


def test_zeros_kept_required():
    d = table()
    d.loc[3, "zeros_kept"] = False
    assert score_fix(d, "spike")["targets"]["zeros_kept"]["pass"] is False


def test_missed_lists_unrepaired():
    d = table()
    d.loc[d.case_id == "d0001", "n_repaired"] = 1
    d.loc[d.case_id == "d0002", ["changed", "identical"]] = [False, True]
    assert score_fix(d, "spike")["missed"] == ["d0001", "d0002"]


def test_bias_targets():
    d = table("bias", removed=0.71, no_after=32).drop(columns=["own_air_in", "own_air_out", "n_true", "n_repaired"])
    s = score_fix(d, "bias")
    assert all(t["pass"] for t in s["targets"].values()), s["targets"]
    assert TARGETS["bias"]["no_after"] == 32
    d.loc[0, "after_present"] = True
    assert score_fix(d, "bias")["targets"]["no_after"]["pass"] is False


def test_dev_only_ids():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import score_fixes
    man = pd.DataFrame({"case_id": ["s0001", "d0002", "t0003"], "set": ["sweep", "dev", "test"]})
    assert score_fixes.ids(man) == ["d0002", "s0001"]


def test_unmeasured_after_is_not_a_no():
    """Review I1: a volume the tool could not read afterwards (None) is not a "no"."""
    d = table("bias", removed=0.71, no_after=32).drop(columns=["own_air_in", "own_air_out", "n_true", "n_repaired"])
    d["after_present"] = d.after_present.astype(object)
    d.loc[0, "after_present"] = None
    assert score_fix(d, "bias")["targets"]["no_after"]["pass"] is False


def test_errors_target():
    d = table()
    d["error"] = False
    assert score_fix(d, "spike")["targets"]["errors"]["pass"] is True
    d.loc[5, "error"] = True
    s = score_fix(d, "spike")
    assert s["targets"]["errors"]["pass"] is False and s["targets"]["errors"]["got"] == 1


def test_seconds_on_own_volumes_only():
    """§10: "≤ 60 s each" is a target on the 32 spike-alone volumes; mixed ones are reported only."""
    d = table()
    d.loc[d.case_id == "d0001", "seconds"] = 75.0
    s = score_fix(d, "spike")
    assert s["targets"]["seconds_max"]["pass"] is True and s["mixed"]["seconds_max"] == 75.0


def test_spike_no_harm_is_the_volumes_the_tool_passes():
    """§10: the volumes the spike tool passes (the fix left them) come back byte-identical; want = their count."""
    d = table()
    d.loc[d.case_id == "d0003", ["changed", "identical"]] = [False, True]  # a missed spike: passed by the tool
    t = score_fix(d, "spike")["targets"]["no_harm"]
    assert t["want"] == 11 and t["got"] == 11 and t["pass"] is True


def noise_table(**k):
    return table("noise", **k).drop(columns=["own_air_in", "own_air_out", "n_true", "n_repaired"])


def test_noise_targets():
    """D119: removed median ≥ 40 %, the noise tool "no" on all 32, none worse, ≤ 90 s each."""
    assert TARGETS["noise"] == {"removed_median": 0.40, "no_after": 32, "worse": 0, "seconds_max": 90.0}
    s = score_fix(noise_table(removed=0.42, no_after=32, seconds=32.0), "noise")
    assert all(t["pass"] for t in s["targets"].values()), s["targets"]
    assert score_fix(noise_table(removed=0.39, no_after=32), "noise")["targets"]["removed_median"]["pass"] is False
    assert score_fix(noise_table(removed=0.42, no_after=31), "noise")["targets"]["no_after"]["pass"] is False
    assert score_fix(noise_table(removed=0.42, no_after=32, seconds=91.0), "noise")["targets"]["seconds_max"][
        "pass"] is False


def test_noise_no_harm_is_the_volumes_the_tool_passes():
    """§11: the volumes the noise tool passes come back byte-identical; a false alarm (changed, no noise in the
    label) is reported by name, not counted against no harm."""
    d = noise_table(removed=0.42, no_after=32)
    d.loc[d.case_id == "s0100", ["changed", "identical"]] = [True, False]
    s = score_fix(d, "noise")
    assert s["targets"]["no_harm"] == {"want": 9, "got": 9, "pass": True}
    assert s["false_alarms"] == ["s0100"]


def test_noise_after_and_errors():
    """The noise tool's yes/no on the output (or before, when the fix left the volume); an error in its second
    reading counts as an error."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import score_fixes as sf
    assert "noise" in sf.FIXES
    assert sf._after_present({"evidence": {"present_before": True, "present_after": False}}, "noise") is False
    assert sf._after_present({"evidence": {"present_before": False}}, "noise") is False
    assert sf._after_present({"evidence": {"present_before": None}}, "noise") is None
    rep = {"why": "σ 4.00 % of WM", "evidence": {"present_after": None, "why_after": "tool error: ValueError: x"}}
    assert sf._error(rep) is True


def test_gpu_turns_wait_is_not_the_fix_time():
    """Workers take turns on the GPU (one at a time); the time spent waiting for a turn is kept apart, so `seconds`
    stays the fix's own time (D119: ≤ 90 s each)."""
    import threading
    import time
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import score_fixes as sf
    lock = threading.Lock()
    f = sf.gpu_turns(lambda a: a + 1, lock)
    lock.acquire()
    threading.Timer(0.3, lock.release).start()
    assert f(1) == 2 and sf.GPU_WAIT[0] >= 0.25
    sf.GPU_WAIT[0] = 0.0
    assert f(2) == 3 and sf.GPU_WAIT[0] < 0.05


def ghost_table(**k):
    return table("ghosting", **k).drop(columns=["n_true", "n_repaired"])


def test_ghosting_targets():
    """D124: removed median ≥ 90 %, the ghosting tool "no" on all 32, none worse, ≤ 30 s each; the air reported."""
    assert TARGETS["ghosting"] == {"removed_median": 0.90, "no_after": 32, "worse": 0, "seconds_max": 30.0}
    s = score_fix(ghost_table(removed=0.97, no_after=32, seconds=12.0), "ghosting")
    assert all(t["pass"] for t in s["targets"].values()), s["targets"]
    assert s["alone"]["own_air_in_median"] == 0.04 and s["alone"]["own_air_out_median"] == 0.002
    assert score_fix(ghost_table(removed=0.89, no_after=32), "ghosting")["targets"]["removed_median"]["pass"] is False
    assert score_fix(ghost_table(removed=0.97, no_after=31), "ghosting")["targets"]["no_after"]["pass"] is False
    assert score_fix(ghost_table(removed=0.97, no_after=32, seconds=31.0), "ghosting")["targets"]["seconds_max"][
        "pass"] is False


def test_ghosting_no_harm_is_the_volumes_the_tool_passes():
    """§14: the volumes the ghosting tool passes come back byte-identical; a false alarm is reported by name."""
    d = ghost_table(removed=0.97, no_after=32)
    d.loc[d.case_id == "s0100", ["changed", "identical"]] = [True, False]
    s = score_fix(d, "ghosting")
    assert s["targets"]["no_harm"] == {"want": 9, "got": 9, "pass": True}
    assert s["false_alarms"] == ["s0100"]


def test_ghosting_after_and_errors():
    """The ghosting tool's yes/no on the output (or before, when the fix left the volume: a period that is not whole
    stays a "yes"); an error in either tool's second reading counts as an error."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import score_fixes as sf
    assert sf.FIXES["ghosting"] == "stage3-ghosting"
    ev = {"present_before": True, "ghost_after": {"present": False, "value": 0.002, "why": "no repeat"}}
    assert sf._after_present({"evidence": ev}, "ghosting") is False
    assert sf._after_present({"evidence": {"present_before": True}}, "ghosting") is True
    assert sf._after_present({"evidence": {"present_before": None}}, "ghosting") is None
    for k in ("ghost_after", "motion_after"):
        rep = {"why": "copies every 80 mm", "evidence": {k: {"present": None, "why": "tool error: ValueError: x"}}}
        assert sf._error(rep) is True
    assert sf._error({"why": "copies every 80 mm", "evidence": ev}) is False


def test_ghosting_motion_summary_leaves_false_alarms_out():
    """Review M1: the motion tool's answers after the fix, alone and mixed; a changed volume without ghosting (a false
    alarm) is in neither group."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import score_fixes as sf
    d = ghost_table(removed=0.97, no_after=32, n_mixed=3, n_none=2)
    d["motion_after"] = pd.Series(False, index=d.index, dtype=object)  # None = "cannot tell"
    d.loc[d.case_id == "d0000", "motion_after"] = None
    d.loc[d.case_id == "s0100", ["changed", "identical", "motion_after"]] = [True, False, True]
    assert sf.motion_summary(d) == {"alone": {"yes": 0, "no": 32, "cannot_tell": 0},
                                    "mixed": {"yes": 0, "no": 2, "cannot_tell": 1}}
