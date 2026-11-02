"""Task 3 Stage 1: the fix report every fix returns (D112), and the rules on the fixes' source (D95, D99 #10)."""
import json
import re
from pathlib import Path

import numpy as np

from bamr.fixes.report import UNIT, FixReport, head_change, unchanged


def test_report_keys_and_strict_json():
    r = FixReport("fix_spike", 1, "spike", True, "1 stripe removed", "a reason", 0.0123456789,
                  {"a": float("nan"), "b": np.float32(2.5)})
    d = json.loads(r.to_json())
    assert list(d) == ["tool", "version", "artifact", "changed", "what", "why", "change", "unit", "evidence"]
    assert d["unit"] == UNIT and d["change"] == 0.012346
    assert d["evidence"] == {"a": None, "b": 2.5}


def test_unchanged_returns_input_itself():
    x = np.ones((4, 5, 6), np.float32)
    y, rep = unchanged(x, "fix_bias", 1, "bias", "no shading", {"line": 0.083})
    assert y is x
    assert rep.changed is False and rep.change == 0.0 and rep.what == "nothing changed"
    assert rep.why == "no shading" and rep.evidence == {"line": 0.083}


def test_head_change():
    x = np.zeros((10, 10, 10), np.float32)
    head = np.zeros(x.shape, bool)
    head[2:8, 2:8, 2:8] = True
    y = x + np.where(head, 0.1, 5.0).astype(np.float32)
    assert abs(head_change(y, x, head) - 0.1) < 1e-6


def test_fixes_never_name_levels_or_hidden_files():
    """D95: no level names; D99 #10: no path into the hidden answers is ever written in a fix."""
    files = sorted((Path(__file__).resolve().parents[1] / "src" / "bamr" / "fixes").glob("*.py"))
    assert files
    for p in files:
        s = p.read_text()
        assert not re.search(r"\b(mild|moderate|borderline|level)\b", s), p.name
        assert not re.search(r"\b(refs|labels|manifest)\b", s), p.name
