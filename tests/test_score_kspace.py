"""Task 2 Stage 2: scoring the spike and ghosting tools on dev volumes (D98); toy tables and phantoms only."""
from pathlib import Path

import pandas as pd
import yaml

from bamr.eval.score_kspace import (DAMAGE_BANDS, despike, ghost_where_ok, label_spike_k, score_kspace,
                                    spike_where_ok)
from bamr.tools.spike import spike_tool
from phantoms import add_spikes, head_phantom, magnitude


def test_damage_bands_match_config():
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "dataset_v2.yaml").read_text())
    assert [tuple(b) for b in cfg["stage2"]["damage_band"]] == [tuple(b) for b in DAMAGE_BANDS]


def test_label_spike_k():
    """s0007's label position → the array's (LR, AP, SI) k; the spectrum's brightest sample is there."""
    assert label_spike_k([[0.5286458333333334, 0.5479166666666667, 0.4395833333333333]]) == [(5, -15, 11)]


def test_spike_where_ok_either_sign():
    assert spike_where_ok([{"k": [-5, 15, -12]}], [(5, -15, 11)])
    assert not spike_where_ok([{"k": [5, -15, 14]}], [(5, -15, 11)])
    assert not spike_where_ok([], [(5, -15, 11)])


def test_ghost_where_ok():
    assert ghost_where_ok(80.0, 3) and ghost_where_ok(61.5, 4)
    assert not ghost_where_ok(120.0, 4) and not ghost_where_ok(None, 3)


def row(cid, artifacts, has, present, value, value_true, level_true="none", where_ok=None, score=1.0):
    return dict(case_id=cid, set="sweep", site="Guys", artifacts=artifacts, has=has, level_true=level_true,
                value_true=value_true, present=present, value=value, score=score, where_ok=where_ok)


def test_score_kspace_toy():
    df = pd.DataFrame([
        row("s1", "spike", True, True, 0.030, 0.030, "moderate", True, 12.0),
        row("s2", "spike", True, False, 0.0, 0.025, "mild", False, 5.0),
        row("s3", "spike", True, None, None, 0.040, "borderline", None, None),
        row("s4", "ghosting", False, True, 0.020, 0.0, score=8.0),
        row("s5", "clean", False, False, 0.0, 0.0, score=3.0),
    ])
    s = score_kspace(df)["sweep"]
    assert s["n_with"] == 3 and s["missed"] == 2 and s["unmeasured"] == 1
    assert s["false_alarms"] == {"ghosting": 1, "clean": 0}
    assert s["where_ok"] == 1.0 and s["median_rel_err"] == 0.5  # s1: 0 %, s2: 100 %
    assert abs(s["same_level"] - 1 / 3) < 1e-9 and abs(s["within_one"] - 2 / 3) < 1e-9
    assert s["auc_n"] == [2, 2] and s["auc"] == 0.75  # 12 > 8, 12 > 3, 5 > 3; not 5 > 8


def test_despike_removes_spike():
    x = magnitude(add_spikes(head_phantom(), [(5, -15, 12)], [0.035]), 0.02, 1)
    f = spike_tool(x)
    y = despike(x, [s["k"] for s in f.evidence["spikes"] + f.evidence["echoes"]])
    assert f.present is True and spike_tool(y).present is False


def test_score_kspace_faint_label():
    """Dev mixed has 'faint' shares (D65 budget, below the mild band): on the value scale that is the band below mild."""
    df = pd.DataFrame([row("d1", "ghosting+spike", True, True, 0.015, 0.016, "faint", True, 9.0),
                       row("d2", "spike", True, True, 0.030, 0.030, "moderate", True, 12.0)]).assign(set="dev")
    s = score_kspace(df)["dev"]
    assert s["same_level"] == 1.0 and s["within_one"] == 1.0
