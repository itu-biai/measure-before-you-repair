"""Task 4 Stage 1: versions as handles with lineage; an old version never changes (D139, before-doc §6)."""
import numpy as np
import pytest

from bamr.agent.store import VersionStore


def test_handles_count_up_and_lineage(tmp_path):
    s = VersionStore(tmp_path)
    x = np.zeros((4, 5, 6), np.float32)
    assert s.add(x, None, None) == "v0"
    assert s.add(x + 1, "v0", "fix_spike") == "v1"
    assert s.add(x + 2, "v0", "fix_bias") == "v2"
    assert s.add(x + 3, "v2", "fix_noise") == "v3"
    assert s.handles() == ["v0", "v1", "v2", "v3"]
    lin = s.lineage("v3")
    assert [(r["handle"], r["parent"], r["repair"]) for r in lin] == [("v0", None, None), ("v2", "v0", "fix_bias"),
                                                                      ("v3", "v2", "fix_noise")]
    assert all(len(r["hash"]) == 64 for r in lin)
    assert s.get("v3").dtype == np.float32 and float(s.get("v3")[0, 0, 0]) == 3.0
    assert s.path("v1").exists()
    with pytest.raises(KeyError):
        s.get("v9")


def test_old_version_never_changes(tmp_path):
    s = VersionStore(tmp_path)
    s.add(np.ones((3, 3, 3), np.float64), None, None)
    h0 = s.hash("v0")
    v = s.get("v0")
    with pytest.raises(ValueError):
        v[0, 0, 0] = 5
    s.add(np.zeros((3, 3, 3)), "v0", "fix_bias")
    assert s.hash("v0") == h0 and float(s.get("v0").sum()) == 27.0
    with pytest.raises(PermissionError):
        s.path("v0").write_bytes(b"x")


def test_hash_is_content(tmp_path):
    s = VersionStore(tmp_path)
    x = np.random.default_rng(0).random((4, 4, 4)).astype(np.float32)
    s.add(x, None, None)
    s.add(x.copy(), "v0", "fix_spike")
    y = x.copy()
    y[1, 2, 3] += 1e-3
    s.add(y, "v0", "fix_noise")
    assert s.hash("v0") == s.hash("v1") != s.hash("v2")
    assert s.hash("v0") != VersionStore.content_hash(x.reshape(8, 8))  # the shape is part of the hash
