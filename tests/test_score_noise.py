"""Task 2 Stage 1: scoring the noise tool on dev volumes (D72); toy tables only."""
import pandas as pd

from bamr.eval.score_noise import NOISE_BANDS, case_ids, level_of, score_noise


def row(case_id, artifacts, noise, present, value, sigma_true, noise_level="none", site="Guys", set_="sweep"):
    """No 'level' column: the scorer derives it from the value (D95)."""
    return dict(case_id=case_id, set=set_, site=site, artifacts=artifacts, noise=noise, noise_level=noise_level,
                sigma_true=sigma_true, present=present, value=value, raw=value)


def test_case_ids_excludes_test():
    m = pd.DataFrame({"case_id": ["t0001", "d0001", "s0001", "t0002"], "set": ["test", "dev", "sweep", "test"]})
    assert case_ids(m) == ["d0001", "s0001"]


def test_score_toy():
    df = pd.DataFrame([
        row("s1", "noise", True, True, 0.033, 0.032, "mild"),
        row("s2", "noise", True, False, 0.028, 0.031, "mild"),
        row("s3", "spike", False, True, 0.031, 0.020),
        row("s4", "bias", False, False, 0.020, 0.020),
    ])
    s = score_noise(df)["sweep"]
    assert s["n_noise"] == 2 and s["missed"] == 1
    assert s["false_alarms"] == {"spike": 1, "bias": 0}
    assert s["auc"] == 0.75 and 0 <= s["auc_no_spike"] <= 1


def test_score_levels():
    df = pd.DataFrame([
        row("s1", "noise", True, True, 0.033, 0.033, "mild"),
        row("s2", "noise", True, True, 0.040, 0.033, "mild", site="HH"),
        row("s3", "noise", True, True, 0.065, 0.033, "mild", site="HH"),
    ])
    s = score_noise(df)["sweep"]
    assert abs(s["same_level"] - 1 / 3) < 1e-9 and abs(s["within_one"] - 2 / 3) < 1e-9
    assert s["by_site"]["Guys"]["same_level"] == 1.0


def test_dev_only_rejects_test_ids():
    import pytest
    from bamr.eval.score_noise import dev_only
    assert dev_only("s0001") == "s0001" and dev_only("d0001") == "d0001"
    for bad in ("t0001", "x1", ""):
        with pytest.raises(ValueError):
            dev_only(bad)
    with pytest.raises(ValueError):
        case_ids(pd.DataFrame({"case_id": ["s0001", "t0003"], "set": ["sweep", "dev"]}))


def test_level_of_from_config():
    import yaml
    from pathlib import Path
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "dataset_v2.yaml").read_text())
    assert [tuple(b) for b in cfg["stage2"]["noise"]["sigma_total"]] == [tuple(b) for b in NOISE_BANDS]
    assert [level_of(v) for v in (0.0299, 0.030, 0.038, 0.048, 0.061, None)] == \
        ["none", "mild", "moderate", "borderline", "above", None]


# ---------------------------------------------------------------- Stage 1 leftovers (D99)
def test_score_counts_unmeasured():
    """#4: a case the tool could not measure is counted, missed, and kept out of the error."""
    df = pd.DataFrame([
        row("s1", "noise", True, True, 0.033, 0.033, "mild"),
        row("s2", "noise", True, None, None, 0.033, "mild"),
        row("s3", "bias", False, False, 0.020, 0.020),
    ])
    s = score_noise(df)["sweep"]
    assert s["unmeasured"] == 1 and s["missed"] == 1
    assert s["median_rel_err"] == 0.0 and s["median_ratio"] == 1.0


def test_same_decision_both_unmeasured():
    """#5: 'cannot measure' in both zero-air runs is the same finding."""
    from bamr.eval.score_noise import same_decision
    got = same_decision(pd.Series([None, True, False]), pd.Series([None, True, True]))
    assert got.tolist() == [True, True, False]
