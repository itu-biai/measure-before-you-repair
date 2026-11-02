"""Task 4 Stage 1: the compute queue and its cache (D139) — one computation per (function, volume), the guard inside
every worker, pictures in the tools' grey scale."""
import threading

import numpy as np
import pytest
from agent_fakes import TABLE, marker_volume
from phantoms import head_phantom

from bamr.agent import pictures
from bamr.agent.compute import Compute
from bamr.agent.store import VersionStore


def _v0(tmp_path, **s):
    st = VersionStore(tmp_path / "versions")
    st.add(marker_volume(**s), None, None)
    return st, [(st.path("v0"), st.hash("v0"))]


def test_same_call_computes_once(tmp_path):
    st, inp = _v0(tmp_path, noise=0.04)
    c = Compute(tmp_path / "cache", workers=0, blocked=(), table=TABLE)
    out = []
    ts = [threading.Thread(target=lambda: out.append(c.call("slow_tool", inp))) for _ in range(2)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert c.computed["slow_tool"] == 1
    assert [o.data["present"] for o in out] == [True, True]
    third = c.call("slow_tool", inp)
    assert third.cached and third.data == out[0].data and third.picture.exists()
    assert c.computed["slow_tool"] == 1


def test_cache_survives_restart(tmp_path):
    st, inp = _v0(tmp_path, spike=0.2)
    a = Compute(tmp_path / "cache", workers=0, blocked=(), table=TABLE).call("spike_tool", inp)
    b = Compute(tmp_path / "cache", workers=0, blocked=(), table=TABLE)
    o = b.call("spike_tool", inp)
    assert not a.cached and o.cached and o.data == a.data and b.computed["spike_tool"] == 0
    assert b.call("spike_tool", inp, {"x": 1}).cached is False  # the arguments are part of the key


def test_fix_output_only_when_changed(tmp_path):
    st, inp = _v0(tmp_path, bias=0.1)
    c = Compute(tmp_path / "cache", workers=0, blocked=(), table=TABLE)
    o = c.call("fix_bias", inp)
    assert o.data["changed"] is True and o.output.exists()
    assert float(np.load(o.output)[3, 0, 0]) == 0.0
    n = c.call("fix_noise", inp)
    assert n.data["changed"] is False and n.output is None


def test_errors_are_reported_not_cached(tmp_path):
    st, inp = _v0(tmp_path)
    c = Compute(tmp_path / "cache", workers=0, blocked=(), table=TABLE)
    o = c.call("look", inp, {"plane": "axial", "bogus": 1})
    assert "error" in o.data and not o.cached
    assert "error" in c.call("look", inp, {"plane": "axial", "bogus": 1}).data
    assert c.computed["look"] == 2


def test_worker_guard_blocks(tmp_path, monkeypatch):
    secret = tmp_path / "labels"
    secret.mkdir()
    (secret / "d0001.json").write_text("{}")
    monkeypatch.setenv("BAMR_TEST_BLOCKED", str(secret / "d0001.json"))
    st, inp = _v0(tmp_path)
    c = Compute(tmp_path / "cache", workers=1, blocked=(secret,), table=TABLE)
    try:
        o = c.call("peek", inp)
        n = c.call("noise_tool", inp)
    finally:
        c.close()
    assert "blind guard" in o.data["error"]
    assert n.data["present"] is False  # the worker goes on


def test_look_and_compare_pictures(tmp_path):
    x = head_phantom((48, 60, 60)).astype(np.float32)
    d = pictures.look(x, tmp_path / "look.png")
    assert (tmp_path / "look.png").stat().st_size > 1000 and set(d["planes"]) == {"axial", "coronal", "sagittal"}
    one = pictures.look(x, tmp_path / "one.png", plane="coronal", slice=10)
    assert one["planes"] == {"coronal": 10}
    with pytest.raises(ValueError):
        pictures.look(x, tmp_path / "bad.png", plane="axial", slice=60)
    assert pictures.compare(x, x, tmp_path / "c.png")["rms_change"] == 0.0
    y = x + np.float32(0.1)
    assert abs(pictures.compare(y, x, tmp_path / "c2.png")["rms_change"] - 0.1) < 1e-5
    pictures.digit(7, tmp_path / "d.png")
    assert pictures.data_url(tmp_path / "d.png").startswith("data:image/png;base64,iVBOR")


def test_dead_worker_one_new_pool(tmp_path, monkeypatch):
    """Review I2: every job in a broken pool fails at once; only one new pool may replace it."""
    monkeypatch.setenv("BAMR_DIE_FLAG", str(tmp_path / "died"))
    st = VersionStore(tmp_path / "versions")
    st.add(marker_volume(noise=0.1), None, None)
    st.add(marker_volume(noise=0.2), "v0", "x")
    c = Compute(tmp_path / "cache", workers=1, blocked=(), table=TABLE)
    out = {}
    try:
        ts = [threading.Thread(target=lambda h=h: out.__setitem__(h, c.call("die_once", [(st.path(h), st.hash(h))])))
              for h in ("v0", "v1")]
        [t.start() for t in ts]
        [t.join() for t in ts]
    finally:
        c.close()
    assert c.pools == 2
    assert out["v0"].data["present"] is True and out["v1"].data["present"] is True


def test_worker_that_always_dies_is_an_error(tmp_path):
    st, inp = _v0(tmp_path)
    c = Compute(tmp_path / "cache", workers=1, blocked=(), table=TABLE)
    try:
        o = c.call("die_always", inp)
    finally:
        c.close()
    assert "worker died" in o.data["error"] and not o.cached and c.pools == 4


def test_look_air_window_picture(tmp_path):
    """r03: the real look draws the air window and says so; an unknown window is a reason, not a crash."""
    import numpy as np
    from phantoms import head_phantom

    from bamr.agent.pictures import look
    x = head_phantom((48, 60, 60)).astype(np.float32)
    assert look(x, tmp_path / "a.png", window="air")["window"] == "air" and (tmp_path / "a.png").exists()
    assert look(x, tmp_path / "h.png")["window"] == "head"
    import pytest
    with pytest.raises(ValueError):
        look(x, tmp_path / "z.png", window="bone")
