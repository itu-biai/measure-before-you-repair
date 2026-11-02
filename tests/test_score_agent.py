"""Task 4 Stage 1: the check scorer (D132, D140) — error removed against the reference with its motion replayed,
detection, the rescan call and the process counts. Synthetic arrays; no dataset file is read."""
import numpy as np
import pytest

from bamr.eval.score_agent import case_row, motion_target, summary

ARTS = ("noise", "spike", "ghosting", "bias", "motion")


def _res(version="v1", report=None, **kw):
    rep = report if report is not None else {
        "seen": "x", "artifacts": {a: "no" for a in ARTS}, "rescan": "no", "rescan_why": ""}
    return {"system": "agent", "model": "m", "version": version, "report": rep,
            "lineage": [{"handle": "v0", "repair": None}, {"handle": "v1", "repair": "fix_spike"}],
            "turns": 5, "calls": 6, "bad_calls": 1, "fused": False, "error": None, "seconds": 100.0,
            "usage": {"prompt_tokens": 1000, "completion_tokens": 100, "cost": 0.01, "cached_tokens": 500,
                      "reasoning_tokens": 50}, "generation": {"cost": 0.011}, **kw}


def _lab(arts, motion_rms=0.0):
    return {"artifacts": list(arts), "params": {"motion": {"motion_rms": motion_rms}} if "motion" in arts else {}}


def test_motion_target_keeps_only_motion():
    seen = []
    replay = lambda ref, lab: (seen.append(lab["artifacts"]), ref + len(lab["artifacts"]))[1]  # noqa: E731
    ref = np.zeros((2, 2, 2), np.float32)
    assert motion_target(ref, _lab(["bias", "motion", "spike"], 0.02), replay)[0, 0, 0] == 1
    assert motion_target(ref, _lab(["bias", "spike"]), replay) is ref
    assert seen == [["motion"]]


def test_removed_and_worse():
    t = np.zeros((4, 4, 4), np.float32)
    brain = np.ones(t.shape, bool)
    r = case_row(t + 0.2, t + 0.05, t, brain, _lab(["spike"]), _res())
    assert r["error_in"] == pytest.approx(0.2) and r["error_out"] == pytest.approx(0.05)
    assert r["removed"] == pytest.approx(0.75) and r["worse"] is False
    w = case_row(t + 0.1, t + 0.3, t, brain, _lab(["spike"]), _res())
    assert w["removed"] == pytest.approx(-2.0) and w["worse"] is True
    assert r["repairs"] == ["fix_spike"] and r["cost"] == 0.011 and r["cached_tokens"] == 500


def test_clean_volume_removed_none():
    t = np.ones((3, 3, 3), np.float32)
    brain = np.ones(t.shape, bool)
    same = case_row(t, t, t, brain, _lab([]), _res(version="v0"))
    assert same["removed"] is None and same["worse"] is False
    assert case_row(t, t + 0.01, t, brain, _lab([]), _res())["worse"] is True


def test_detection_and_rescan():
    t = np.zeros((3, 3, 3), np.float32)
    brain = np.ones(t.shape, bool)
    rep = {"seen": "x", "artifacts": {"noise": "yes", "spike": "yes", "ghosting": "no", "bias": "no",
                                      "motion": "yes"}, "rescan": "yes", "rescan_why": ""}
    r = case_row(t, t, t, brain, _lab(["spike", "motion"], 0.04), _res(report=rep))
    assert r["right"] == {"noise": False, "spike": True, "ghosting": True, "bias": True, "motion": True}
    assert r["n_right"] == 4 and r["rescan_true"] is True and r["rescan_right"] is True
    low = case_row(t, t, t, brain, _lab(["motion"], 0.02), _res(report=rep))
    assert low["rescan_true"] is False and low["rescan_right"] is False
    fused = case_row(t, t, t, brain, _lab(["spike"]), _res(report=None, version="v0", fused=True))
    fused_none = case_row(t, t, t, brain, _lab(["spike"]), {**_res(version="v0", fused=True), "report": None})
    assert fused_none["has_report"] is False and fused_none["right"]["spike"] is False
    assert fused["fused"] is True


def test_summary_counts():
    t = np.zeros((3, 3, 3), np.float32)
    brain = np.ones(t.shape, bool)
    rows = [case_row(t + 0.2, t + 0.1, t, brain, _lab(["spike"]), _res()),
            case_row(t + 0.2, t + 0.3, t, brain, _lab(["spike"]), _res(fused=True)),
            case_row(t, t, t, brain, _lab([]), _res(version="v0"))]
    s = summary(rows)
    assert s["n"] == 3 and s["removed_median"] == pytest.approx(0.0, abs=1e-6) and s["worse"] == 1
    assert s["artifacts_right"] == 13 and s["artifacts_total"] == 15 and s["rescan_right"] == 3
    assert s["fuses"] == 1 and s["calls"] == 18 and s["bad_calls"] == 3 and s["cost"] == pytest.approx(0.033)


def test_summary_detection_sensitivity_specificity():
    t = np.zeros((3, 3, 3), np.float32)
    brain = np.ones(t.shape, bool)
    yes = lambda *arts: {"seen": "x", "artifacts": {a: "yes" if a in arts else "no" for a in ARTS},  # noqa: E731
                         "rescan": "no", "rescan_why": ""}
    rows = [case_row(t, t, t, brain, _lab(["spike"]), _res(report=yes("spike", "noise"))),   # spike TP, noise FP
            case_row(t, t, t, brain, _lab(["spike", "noise"]), _res(report=yes())),         # spike FN, noise FN
            case_row(t, t, t, brain, _lab([]), _res(report=yes("noise")))]                  # noise FP
    d = summary(rows)["detection"]
    assert (d["spike"]["tp"], d["spike"]["fn"], d["spike"]["fp"], d["spike"]["tn"]) == (1, 1, 0, 1)
    assert d["spike"]["sensitivity"] == pytest.approx(0.5) and d["spike"]["specificity"] == pytest.approx(1.0)
    assert (d["noise"]["tp"], d["noise"]["fn"], d["noise"]["fp"], d["noise"]["tn"]) == (0, 1, 2, 0)
    assert d["noise"]["sensitivity"] == 0 and d["noise"]["specificity"] == 0  # yes on both volumes without it
    assert d["bias"]["sensitivity"] is None and d["bias"]["specificity"] == pytest.approx(1.0)


def test_summary_quality_over_volumes_with_an_artifact():
    t = np.zeros((3, 3, 3), np.float32)
    brain = np.ones(t.shape, bool)
    q = lambda p, s, l: {"psnr_in": p - 5, "ssim_in": s - 0.1, "lpips_in": l + 0.1,  # noqa: E731
                         "psnr_out": p, "ssim_out": s, "lpips_out": l}
    rows = [{**case_row(t, t, t, brain, _lab(["spike"]), _res()), **q(30.0, 0.9, 0.1)},
            {**case_row(t, t, t, brain, _lab(["noise"]), _res()), **q(34.0, 0.8, 0.3)},
            {**case_row(t, t, t, brain, _lab([]), _res(version="v0")), **q(float("inf"), 1.0, 0.0)}]  # clean: left out
    s = summary(rows)["quality"]
    assert s["n"] == 2
    assert s["psnr_out"]["mean"] == pytest.approx(32.0) and s["psnr_out"]["sd"] == pytest.approx(np.std([30, 34], ddof=1))
    assert s["psnr_in"]["median"] == pytest.approx(27.0) and s["ssim_out"]["mean"] == pytest.approx(0.85)
    assert s["lpips_out"]["mean"] == pytest.approx(0.2) and s["lpips_in"]["mean"] == pytest.approx(0.3)
    assert summary([case_row(t, t, t, brain, _lab(["spike"]), _res())])["quality"] is None  # no quality columns
