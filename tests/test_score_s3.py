"""Task 2 Stage 3: scoring the bias and motion tools on dev volumes (D104); toy tables only."""
from pathlib import Path

import pandas as pd
import yaml

from bamr.eval.score_kspace import DAMAGE_BANDS
from bamr.eval.score_noise import level_of
from bamr.eval.score_s3 import BIAS_BANDS, level_gapped, score_s3


def test_bias_bands_match_config():
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "dataset_v2.yaml").read_text())
    assert [tuple(b) for b in cfg["stage2"]["bias"]["target_maxmin"]] == [tuple(b) for b in BIAS_BANDS]


def test_level_gapped_bias():
    """R6: a value between two bands takes the nearer one (split at the gap's middle); Stage 1's level_of would say
    'above' for 0.30."""
    assert [level_gapped(v, BIAS_BANDS) for v in (0.10, 0.15, 0.29, 0.30, 0.44, 0.50, 0.51, 0.69, 0.70, 0.75)] == \
        ["none", "mild", "mild", "moderate", "moderate", "borderline", "borderline", "borderline", "above", "above"]
    assert level_gapped(None, BIAS_BANDS) is None and level_gapped(float("nan"), BIAS_BANDS) is None


def test_level_gapped_touching_bands_as_before():
    for v in (0.0, 0.019, 0.0194, 0.025, 0.0283, 0.03, 0.04, 0.0553, 0.07):
        assert level_gapped(v, DAMAGE_BANDS) == level_of(v, DAMAGE_BANDS)


def row(cid, artifacts, has, present, value, value_true, level_true="none", score=1.0, s="sweep"):
    return dict(case_id=cid, set=s, site="Guys", artifacts=artifacts, has=has, level_true=level_true,
                value_true=value_true, present=present, value=value, score=score)


def test_score_s3_cannot_tell_apart():
    """'Cannot tell' (present None) is neither a miss nor a false alarm: counted apart, with and without the artifact."""
    df = pd.DataFrame([
        row("s1", "motion", True, True, 0.030, 0.030, "moderate", 0.007),
        row("s2", "motion", True, False, 0.010, 0.025, "mild", 0.002),
        row("s3", "ghosting", False, None, None, 0.0, score=0.03),
        row("s4", "noise", False, True, 0.020, 0.0, score=0.005),
        row("s5", "clean", False, False, 0.0, 0.0, score=0.0),
        row("d1", "ghosting+motion", True, None, None, 0.02, "mild", 0.04, "dev"),
    ])
    r = score_s3(df, DAMAGE_BANDS)
    s = r["sweep"]
    assert s["n_with"] == 2 and s["missed"] == 1 and s["cannot_tell"] == {"with": 0, "without": 1}
    assert s["false_alarms"] == {"ghosting": 0, "noise": 1, "clean": 0}
    assert s["median_rel_err"] == 0.3  # s1: 0 %, s2: 60 %
    assert abs(s["same_level"] - 0.5) < 1e-9 and s["within_one"] == 1.0  # s2 reads 'none' for 'mild'
    assert s["auc_n"] == [2, 2] and s["auc"] == 0.75  # measured only: 0.007 beats both, 0.002 beats 0.0 only
    assert s["auc_all_n"] == [2, 3]  # with the cannot-tell volume's score (0.03 beats both motion volumes)
    d = r["dev"]
    assert d["missed"] == 0 and d["cannot_tell"] == {"with": 1, "without": 0} and d["median_rel_err"] is None


def test_score_s3_faint_label_is_below_mild():
    df = pd.DataFrame([row("d1", "ghosting+motion", True, True, 0.015, 0.016, "faint", s="dev")])
    s = score_s3(df, DAMAGE_BANDS)["dev"]
    assert s["same_level"] == 1.0 and s["within_one"] == 1.0
