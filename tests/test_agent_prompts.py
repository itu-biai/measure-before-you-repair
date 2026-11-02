"""Task 4 Stage 1: what the model reads, word for word (D137), the twelve functions, and Task 2/3 left as they are."""
import json
import subprocess
from pathlib import Path

import pytest

from bamr.agent import functions as F
from bamr.agent import prompts as P

REPO = Path(__file__).resolve().parents[1]
BEFORE = REPO.parent / "visual-guide" / "task4" / "01_harness" / "before_numbers.json"
ORDER = ["look", "compare", "noise_tool", "spike_tool", "ghosting_tool", "bias_tool", "motion_tool", "fix_spike",
         "fix_ghosting", "fix_noise", "fix_bias", "finish"]


FROZEN = REPO.parent / "visual-guide" / "task4" / "02_develop" / "texts_frozen.json"


@pytest.mark.skipif(not BEFORE.exists(), reason="the visual guide is not next to the repo")
def test_texts_structure():
    """Stage 2 changes the texts one at a time (D145); what stays: the protocol paragraph, RESCAN, the router."""
    b = json.loads(BEFORE.read_text())
    protocol = P.SYSTEM.split("\n\n")[3]
    assert protocol.startswith("Scanner protocol:") and protocol == P.PROTOCOL
    assert P.ROUTER == b["router_prompt"].replace("[same protocol lines as the agent]", protocol)
    assert P.RESCAN == 0.033 and str(P.RESCAN) in P.SYSTEM and list(P.CARDS) == ORDER


@pytest.mark.skipif(not FROZEN.exists(), reason="the texts are frozen at the end of Stage 2's development")
def test_texts_frozen():
    """After the freeze (D144): what the model reads is word for word the frozen file."""
    f = json.loads(FROZEN.read_text())
    assert P.SYSTEM == f["system"] and P.FIRST == f["first"] and P.CARDS == f["cards"]
    assert P.SYSTEM_EYES == f["system_eyes"] and P.CARDS_EYES == f["cards_eyes"]
    assert P.ROUTER == f["router"] and P.ROUTER_READINGS == f["router_readings"]
    assert F.TOOLS == f["tools"] and F.TOOLS_EYES == f["tools_eyes"] and F.ANSWER == f["answer"]


def test_twelve_functions_in_order():
    names = [t["function"]["name"] for t in F.TOOLS]
    assert names == ORDER
    for t in F.TOOLS:
        f = t["function"]
        assert t["type"] == "function" and f["strict"] is True
        assert f["description"] == P.CARDS[f["name"]]
        p = f["parameters"]
        assert p["type"] == "object" and p["additionalProperties"] is False
        assert sorted(p["required"]) == sorted(p["properties"])  # strict: every field required, optional ones nullable


def test_finish_fields_in_order():
    p = next(t for t in F.TOOLS if t["function"]["name"] == "finish")["function"]["parameters"]
    assert list(p["properties"]) == ["report", "repairs", "artifacts", "rescan", "rescan_why", "version"]
    a = p["properties"]["artifacts"]
    assert list(a["properties"]) == ["noise", "spike", "ghosting", "bias", "motion"]
    assert all(v["enum"] == ["yes", "no"] for v in a["properties"].values())
    assert p["properties"]["rescan"]["enum"] == ["yes", "no"]
    q = F.ANSWER["function"]["parameters"]
    assert F.ANSWER["function"]["name"] == "answer"
    assert list(q["properties"]) == ["seen", "artifacts", "rescan", "rescan_why"]


def test_check_errors():
    h = {"v0"}
    assert F.check("spike_tool", {"version": "v0"}, h) is None
    assert F.check("look", {"version": "v0", "plane": None, "slice": None, "window": None}, h) is None
    assert F.check("look", {"version": "v0", "plane": "axial", "slice": 120, "window": "air"}, h) is None
    assert "unknown function" in F.check("despike", {"version": "v0"}, h)
    assert "version" in F.check("spike_tool", {}, h)
    e = F.check("spike_tool", {"version": "v7"}, h)
    assert "v7" in e and "v0" in e
    assert "plane" in F.check("look", {"version": "v0", "plane": "top", "slice": None}, h)
    assert "slice" in F.check("look", {"version": "v0", "plane": "axial", "slice": "mid", "window": None}, h)
    assert "unexpected" in F.check("fix_bias", {"version": "v0", "strength": 2}, h)
    arts = ("noise", "spike", "ghosting", "bias", "motion")
    fin = {"report": {"examination": "x", "findings": {k: "absent" for k in arts}, "what_was_done": "",
                      "impression": "", "recommendation": "", "limits": ""}, "repairs": [],
           "artifacts": {k: "no" for k in arts}, "rescan": "no", "rescan_why": "", "version": "v0"}
    assert F.check("finish", fin, h) is None
    bad = dict(fin, artifacts=dict(fin["artifacts"], noise="maybe"))
    assert "noise" in F.check("finish", bad, h)
    assert "artifacts" in F.check("finish", dict(fin, artifacts={"noise": "no"}), h)
    assert "compare" not in (F.check("compare", {"a": "v0", "b": "v0"}, h) or "")


def test_task2_task3_untouched():
    """D139: the harness calls the Task 2 tools and Task 3 fixes as they are (commit 928d088 is the last before T4)."""
    paths = ["src/bamr/tools", "src/bamr/fixes"]
    try:
        diff = subprocess.run(["git", "diff", "--name-only", "928d088", "--", *paths], cwd=REPO, capture_output=True,
                              text=True, check=True).stdout
        st = subprocess.run(["git", "status", "--porcelain", "--", *paths], cwd=REPO, capture_output=True, text=True,
                            check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git not available")
    assert diff == "" and st == ""


def test_eyes_texts_are_the_agents_without_the_measuring_tools():
    """D150: ④a gets ④'s texts with the measuring tools taken out, nothing else changed."""
    sp, ep = P.SYSTEM.split("\n\n"), P.SYSTEM_EYES.split("\n\n")
    assert len(sp) == len(ep)
    changed = [i for i, (a, b) in enumerate(zip(sp, ep)) if a != b]
    assert changed == P.EYES_EDITED
    assert "measure" not in " ".join(ep[i] for i in changed) and "motion-tool" not in P.SYSTEM_EYES
    assert list(P.CARDS_EYES) == ["look", "compare", "fix_spike", "fix_ghosting", "fix_noise", "fix_bias", "finish"]
    for k in ("look", "compare", "finish"):
        assert P.CARDS_EYES[k] == P.CARDS[k]
    for k in ("fix_spike", "fix_ghosting", "fix_noise", "fix_bias"):
        assert "tool" not in P.CARDS_EYES[k] and "unchanged" in P.CARDS_EYES[k]
        assert P.CARDS[k].split("On 32 dev volumes")[1] in P.CARDS_EYES[k].replace("whose own noise is high",
                                                                                   "where the noise tool says yes without added noise")
    names = [t["function"]["name"] for t in F.tools(eyes=True)]
    assert names == list(P.CARDS_EYES) and "unknown function" in F.check("noise_tool", {"version": "v0"}, {"v0"},
                                                                            eyes=True)


def test_argument_errors_name_the_whole_form():
    """r01 (D145): a refused call's error names every expected field, so one retry can get it right."""
    e = F.check("finish", {"version": "v0"}, {"v0"})
    assert e.startswith("missing argument: report")
    for k in ("report {examination, findings {noise, spike, ghosting, bias, motion}, what_was_done, impression, "
              "recommendation, limits}", "repairs [", "artifacts {noise, spike, ghosting, bias, motion}", "rescan",
              "rescan_why", "version"):
        assert k in e, k
    assert "look takes: version, plane, slice, window" in F.check("look", {"version": "v0"}, {"v0"})
    assert F.check("look", {"version": "v9", "plane": None, "slice": None, "window": None}, {"v0"}).startswith("unknown version")


def test_look_air_window():
    """r03: look's air window (0 to 0.1 × WM) for faint signal outside the head (mortamet2009)."""
    assert "window" in F.check("look", {"version": "v0", "plane": None, "slice": None, "window": "lung"}, {"v0"})
    assert "air window" in P.CARDS["look"] and P.CARDS_EYES["look"] == P.CARDS["look"]
