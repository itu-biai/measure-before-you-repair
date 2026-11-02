"""Fakes for the Task 4 harness tests: cheap tools and fixes on marker volumes, and a scripted model (no API call).

A marker volume is 8 × 8 × 8 with x[i, 0, 0] the strength of artifact i (noise, spike, ghosting, bias, motion): a
tool says yes when its marker is > 0, a fix sets its marker to 0 (and fix_spike adds `SPIKE_MOTION` to the motion
marker, as the real spike repair raised d0007's motion reading). Everything here is module level, so the spawn
workers can import it by name.
"""
import json
import os
import time
from pathlib import Path

import numpy as np

from bamr.fixes.report import FixReport, unchanged
from bamr.tools.finding import Finding

ARTS = ("noise", "spike", "ghosting", "bias", "motion")
SPIKE_MOTION = 0.05


def marker_volume(**strength) -> np.ndarray:
    x = np.full((8, 8, 8), 0.5, np.float32)
    for i, a in enumerate(ARTS):
        x[i, 0, 0] = strength.get(a, 0.0)
    return x


def _png(path):
    if path is not None:
        Path(path).write_bytes(b"\x89PNG\r\n\x1a\nfake")


def _tool(a, x, picture):
    v = float(x[ARTS.index(a), 0, 0])
    _png(picture)
    return Finding(f"{a}_tool", 1, a, v > 0, v, "marker", "high", f"marker {v:.3f}", {"marker": v},
                   Path(picture).name if picture else None)


def noise_tool(x, protocol, picture=None):
    return _tool("noise", x, picture)


def spike_tool(x, protocol, picture=None):
    return _tool("spike", x, picture)


def ghosting_tool(x, protocol, picture=None):
    return _tool("ghosting", x, picture)


def bias_tool(x, protocol, picture=None):
    return _tool("bias", x, picture)


def motion_tool(x, protocol, picture=None):
    f = _tool("motion", x, picture)
    if os.environ.get("BAMR_FAKE_MOTION_NONE"):
        f.present, f.value = None, None
    return f


def slow_tool(x, protocol, picture=None):
    time.sleep(0.4)
    return _tool("noise", x, picture)


def leaky_tool(x, protocol, picture=None):
    f = _tool("noise", x, picture)
    f.why = "read from " + os.environ.get("BAMR_FAKE_LEAK", "d0042")
    return f


def peek(x, protocol, picture=None):
    """Tries to open a file the guard blocks (path in BAMR_TEST_BLOCKED)."""
    open(os.environ["BAMR_TEST_BLOCKED"]).close()
    return _tool("noise", x, picture)


def _fix(a, x):
    i = ARTS.index(a)
    if x[i, 0, 0] <= 0:
        return unchanged(x, f"fix_{a}", 1, a, f"marker {float(x[i, 0, 0]):.3f}: nothing to repair")
    y = np.array(x, np.float32)
    y[i, 0, 0] = 0
    if a == "spike":
        y[4, 0, 0] += SPIKE_MOTION
    log = os.environ.get("BAMR_FAKE_FIXLOG")
    if log:
        with open(log, "a") as f:
            f.write(a + "\n")
    return y, FixReport(f"fix_{a}", 1, a, True, f"marker {float(x[i, 0, 0]):.3f} → 0", "marker", 0.01, {})


def fix_spike(x, protocol):
    return _fix("spike", x)


def fix_ghosting(x, protocol):
    return _fix("ghosting", x)


def fix_noise(x, protocol):
    return _fix("noise", x)


def fix_bias(x, protocol):
    return _fix("bias", x)


def look(x, path, plane=None, slice=None, window=None):
    _png(path)
    return {"planes": {plane: slice} if plane else {"axial": 4, "coronal": 4, "sagittal": 4}}


def compare(a, b, path):
    _png(path)
    return {"rms_change": float(np.sqrt(np.mean((np.asarray(a, np.float64) - b) ** 2)))}


TABLE = {**{f"{a}_tool": ("tool", f"agent_fakes:{a}_tool") for a in ARTS},
         **{f"fix_{a}": ("fix", f"agent_fakes:fix_{a}") for a in ARTS[:4]},
         "look": ("look", "agent_fakes:look"), "compare": ("compare", "agent_fakes:compare"),
         "slow_tool": ("tool", "agent_fakes:slow_tool"), "peek": ("tool", "agent_fakes:peek"),
         "leaky_tool": ("tool", "agent_fakes:leaky_tool")}


class ScriptedClient:
    """A model that replies from a list. A reply is a list of calls [(name, args), …] or a text string; `seen` keeps
    a copy of every request's messages."""

    def __init__(self, script, reasoning=None):
        self.script = list(script)
        self.seen = []
        self.reasoning = reasoning
        self.n = 0

    def chat(self, messages, tools, tool_choice="auto"):
        from bamr.agent.openrouter import Reply
        self.seen.append(json.loads(json.dumps(messages)))
        self.tools = tools
        step = self.script.pop(0) if self.script else []
        if isinstance(step, Exception):
            raise step
        self.n += 1
        msg = {"role": "assistant", "content": step if isinstance(step, str) else None}
        if not isinstance(step, str):
            msg["tool_calls"] = [{"id": f"call_{self.n}_{k}", "type": "function",
                                  "function": {"name": name,
                                               "arguments": args if isinstance(args, str) else json.dumps(args)}}
                                 for k, (name, args) in enumerate(step)]
        if self.reasoning is not None:
            msg["reasoning_details"] = [{"type": "reasoning.text", "text": f"{self.reasoning} {self.n}"}]
        return Reply(msg, f"gen-{self.n}", {"prompt_tokens": 100, "completion_tokens": 10, "cost": 0.001},
                     "tool_calls" if msg.get("tool_calls") else "stop", 0.01)


def die_once(x, protocol, picture=None):
    """Kills its worker the first time (flag file in BAMR_DIE_FLAG), after a pause so a second job can queue."""
    flag = os.environ["BAMR_DIE_FLAG"]
    if not os.path.exists(flag):
        open(flag, "w").close()
        time.sleep(0.5)
        os._exit(1)
    return _tool("noise", x, picture)


def die_always(x, protocol, picture=None):
    os._exit(1)


TABLE |= {"die_once": ("tool", "agent_fakes:die_once"), "die_always": ("tool", "agent_fakes:die_always")}


def _fix_u(a, x, kw):
    """An ungated repair: always runs (and changes the volume), logged as <artifact>_u."""
    y = np.array(x, np.float32)
    y[ARTS.index(a), 0, 0] = 0
    y[7, 7, 7] += 0.001
    log = os.environ.get("BAMR_FAKE_FIXLOG")
    if log:
        with open(log, "a") as f:
            f.write(f"{a}_u\n")
    return y, FixReport(f"fix_{a}", 1, a, True, f"ungated {kw}", "marker", 0.01, {"args": kw})


def spike_u(x, protocol, **kw):
    return _fix_u("spike", x, kw)


def ghosting_u(x, protocol, **kw):
    return _fix_u("ghosting", x, kw)


def noise_u(x, protocol, **kw):
    return _fix_u("noise", x, kw)


def bias_u(x, protocol, **kw):
    return _fix_u("bias", x, kw)


TABLE |= {f"{a}_u": ("fixp", f"agent_fakes:{a}_u") for a in ARTS[:4]}
