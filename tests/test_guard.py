"""D99 #10: the runtime path guard keeps refs and labels closed while a tool runs."""
import numpy as np
import pytest

from bamr.eval.guard import blind_guard
from bamr.tools.noise import noise_tool


def test_guard_blocks_refs_and_labels(tmp_path):
    for d in ("labels", "refs", "inputs"):
        (tmp_path / d).mkdir()
        (tmp_path / d / "a.json").write_text("{}")
    with blind_guard(tmp_path / "labels", tmp_path / "refs"):
        for d in ("labels", "refs"):
            with pytest.raises(PermissionError):
                open(tmp_path / d / "a.json").close()
            with pytest.raises(PermissionError):
                list((tmp_path / d).iterdir())
        assert (tmp_path / "inputs" / "a.json").read_text() == "{}"
    assert (tmp_path / "labels" / "a.json").read_text() == "{}"


def test_guard_allows_tool_run(tmp_path):
    g = np.ogrid[:96, :120, :120]
    obj = np.sqrt(((g[0] - 48) / 40) ** 2 + ((g[1] - 60) / 50) ** 2 + ((g[2] - 60) / 50) ** 2) < 1
    x = (obj + np.random.default_rng(0).normal(0, 0.04, obj.shape)).astype(np.float32)
    with blind_guard(tmp_path / "labels", tmp_path / "refs"):
        f = noise_tool(x)
    assert f.present is True


# ---------------------------------------------------------------- final-review fix (M5)
def test_guard_blocks_other_case_spelling(tmp_path):
    """macOS file names ignore case: labels/ blocked means LABELS/ is blocked too."""
    import sys
    (tmp_path / "labels").mkdir()
    (tmp_path / "labels" / "a.json").write_text("{}")
    if sys.platform != "darwin":
        pytest.skip("case-insensitive file system only")
    with blind_guard(tmp_path / "labels"):
        with pytest.raises(PermissionError):
            open(tmp_path / "LABELS" / "a.json").close()


def test_guard_blocks_listing_from_inside(tmp_path, monkeypatch):
    """os.listdir() with no argument lists the current folder: blocked when that folder is."""
    import os
    (tmp_path / "labels").mkdir()
    monkeypatch.chdir(tmp_path / "labels")
    with blind_guard(tmp_path / "labels"):
        with pytest.raises(PermissionError):
            os.listdir()


def test_score_tools_blocks_manifest():
    """The manifest lists every case's artifacts and levels: closed to tools like refs and labels."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import score_tools
    assert score_tools.DS / "manifest.csv" in score_tools.BLOCKED
    assert {score_tools.DS / "refs", score_tools.DS / "labels"} <= set(score_tools.BLOCKED)


def test_score_fixes_blocks_manifest():
    """The fixes run under the same guard as the tools: refs, labels and the manifest closed (D99 #10)."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import score_fixes
    assert {score_fixes.DS / "refs", score_fixes.DS / "labels", score_fixes.DS / "manifest.csv"} <= set(score_fixes.BLOCKED)
