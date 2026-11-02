"""The two baselines (D138, before-doc §8), on the same compute and cache as the agent.

② The fixed chain, no model: fix_spike → fix_ghosting → fix_noise → fix_bias, each on the previous output and gated
by its own tool. Repairs of how the scan was acquired (k-space faults) come first, image-domain ones after. Its report
is the five tools on v0 (yes only when a tool says yes; "cannot tell" counts as no). Its rescan call is the motion tool
on the output ≥ RESCAN.
③ The one-step router (AgentMRI-style dispatch, saju2025agentmri): one model call on v0's picture with the router
prompt. The model answers through `answer` (free text first, then yes/no × 5, then rescan); a plain-text answer is
read with a pattern. The fixes it names run in the chain's order, still gated. With no readable answer, nothing runs
and the run is flagged `unparsed`.
②u Every repair on every image (D152): the same order, each repair's gate replaced by "always" (bamr.agent.ungated).
⑤ The oracle (D151): the repairs of the labelled artifacts only, in the same order, ungated, with the label's numbers
where a tool has none; its report is the label. It runs in the scorer's process, outside the guard.
③b The router with readings (D148): the same one call, which also sees the five tools' findings on v0 (as the agent
sees them), each after the tool's card and followed by its picture. It separates what the readings add (③ → ③b) from
what the loop adds (③b → ④).
"""
import json
import re
import time
from pathlib import Path

import numpy as np

from bamr.agent.functions import ANSWER, ARTIFACTS, check_answer
from bamr.agent.loop import CaseLog, _add_usage, _image, _plain_finding, leak_check
from bamr.agent.openrouter import ApiError
from bamr.agent.prompts import CARDS, RESCAN, ROUTER, ROUTER_READINGS
from bamr.agent.store import VersionStore
from bamr.agent.ungated import label_fallbacks

CHAIN = ("fix_spike", "fix_ghosting", "fix_noise", "fix_bias")
_SAY = re.compile(r"\b(noise|spikes?|ghosting|bias|motion)\b[^\n.;,]{0,25}?\b(yes|no)\b", re.I)
_RESCAN = re.compile(r"\brescan\w*\b[^\n.;,]{0,30}?\b(yes|no)\b", re.I)


def _inp(store: VersionStore, h: str):
    return (store.path(h), store.hash(h))


def _repairs(store: VersionStore, compute, todo: list[tuple[str, dict]]) -> tuple[str, list[dict]]:
    """The given (repair, arguments), in the given order, each on the latest version; a new handle only when the volume
    changed."""
    v, steps = "v0", []
    for fix, args in todo:
        out = compute.call(fix, [_inp(store, v)], args or None)
        if out.output:
            v = store.add(np.load(out.output), v, fix)
        steps.append({"fix": fix, "args": args, "changed": out.output is not None, "error": out.data.get("error"),
                      "what": out.data.get("what")})
    return v, steps


def _in_order(fixes) -> list[tuple[str, dict]]:
    return [(f, {}) for f in CHAIN if f in fixes]


def _result(system, model, store, version, report, t, **kw) -> dict:
    return {"system": system, "model": model, "version": version, "report": report,
            "lineage": store.lineage(version), "turns": 0, "calls": 0, "bad_calls": 0, "fused": False, "error": None,
            "seconds": round(time.perf_counter() - t, 1),
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0, "cached_tokens": 0,
                      "reasoning_tokens": 0}, "gen_ids": [], **kw}


def _say(p) -> str:
    return "yes" if p is True else "no" if p is False else "cannot tell"


def run_chain(x0: np.ndarray, compute, work: Path, log: CaseLog, mode: str = "gated", label: dict | None = None) -> dict:
    """② (gated), ②u (ungated) or ⑤ (oracle, with the label: only from the scorer, outside the guard)."""
    t = time.perf_counter()
    store = VersionStore(Path(work) / "versions")
    store.add(x0, None, None)
    order = [c.removeprefix("fix_") for c in CHAIN]
    if mode == "oracle":
        fb, arts = label_fallbacks(label, x0.shape), label["artifacts"]
        version, steps = _repairs(store, compute, [(f"{a}_u", fb.get(a, {})) for a in order if a in arts])
        m = (label.get("params") or {}).get("motion") or {}
        mrms = float(m.get("motion_rms", 0.0)) if "motion" in arts else 0.0
        report = {"seen": "the label", "artifacts": {a: "yes" if a in arts else "no" for a in ARTIFACTS},
                  "rescan": "yes" if mrms >= RESCAN else "no", "rescan_why": f"label motion_rms {mrms}"}
    else:
        found = {a: compute.call(f"{a}_tool", [_inp(store, "v0")]).data for a in ARTIFACTS}
        todo = _in_order(CHAIN) if mode == "gated" else [(f"{a}_u", {}) for a in order]
        version, steps = _repairs(store, compute, todo)
        mo = compute.call("motion_tool", [_inp(store, version)]).data.get("value")
        seen = "; ".join(f"{a} tool on v0: {_say(f.get('present'))} (value {f.get('value')})" for a, f in found.items())
        report = {"seen": seen, "artifacts": {a: "yes" if f.get("present") is True else "no" for a, f in found.items()},
                  "rescan": "yes" if mo is not None and mo >= RESCAN else "no",
                  "rescan_why": f"motion tool on the output: {mo} (rescan at ≥ {RESCAN})"}
    log.output(store.get(version))
    res = _result({"gated": "chain", "ungated": "chain_u", "oracle": "oracle"}[mode], None, store, version, report, t,
                  steps=steps)
    log.result(res)
    return res


def parse_text(text: str) -> dict | None:
    """yes/no per artifact (and rescan) from a plain-text answer; None unless all five are there."""
    said = {}
    for m in _SAY.finditer(text or ""):
        a = m.group(1).lower().rstrip("s") if m.group(1).lower() != "bias" else "bias"
        said.setdefault(a, m.group(2).lower())
    if set(said) != set(ARTIFACTS):
        return None
    r = _RESCAN.search(text)
    return {"seen": text, "artifacts": {a: said[a] for a in ARTIFACTS}, "rescan": r.group(1).lower() if r else "no",
            "rescan_why": ""}


def run_router(x0: np.ndarray, client, compute, work: Path, log: CaseLog, forbidden: list[str],
               readings: bool = False) -> dict:
    t = time.perf_counter()
    system = "router_readings" if readings else "router"
    store = VersionStore(Path(work) / "versions")
    store.add(x0, None, None)
    look = compute.call("look", [_inp(store, "v0")], {"plane": None, "slice": None})
    content = [{"type": "text", "text": ROUTER_READINGS if readings else ROUTER}, _image(look.picture)]
    pics = [log.picture(look.picture, "v0_look.png")]
    for a in ARTIFACTS if readings else ():
        out = compute.call(f"{a}_tool", [_inp(store, "v0")])
        finding = _plain_finding(out.data, out.picture is not None) if "error" not in out.data else out.data
        content.append({"type": "text", "text": f"{a}_tool: {CARDS[a + '_tool']}\nreading on v0: "
                                                f"{json.dumps(finding, ensure_ascii=False)}"})
        if out.picture is not None:
            content.append(_image(out.picture))
            pics.append(log.picture(out.picture, f"{a}_tool.png"))
    msg = {"role": "user", "content": content}
    leak_check([msg], forbidden)
    log.message(msg, pics)
    model = getattr(getattr(client, "cfg", None), "slug", None)
    try:
        r = client.chat([msg], [ANSWER])
    except ApiError as e:
        log.output(store.get("v0"))
        res = _result(system, model, store, "v0", None, t, error=str(e), named=[], unparsed=True)
        log.result(res)
        return res
    log.message(r.message)
    log.api({"turn": 1, "gen_id": r.gen_id, "seconds": r.seconds, "finish_reason": r.finish_reason, "usage": r.usage,
             "tool_calls": len(r.message.get("tool_calls") or [])})
    answer = None
    for tc in r.message.get("tool_calls") or []:
        fn = tc.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            continue
        if fn.get("name") == "answer" and check_answer(args) is None:
            answer = args
            break
    if answer is None:
        answer = parse_text(r.message.get("content") or "")
    unparsed = answer is None
    if unparsed:
        answer = {"seen": r.message.get("content") or "", "artifacts": {a: "no" for a in ARTIFACTS}, "rescan": "no",
                  "rescan_why": ""}
    named = [f"fix_{a}" for a in ARTIFACTS if answer["artifacts"][a] == "yes" and f"fix_{a}" in CHAIN]
    named = [f for f in CHAIN if f in named]
    version, steps = _repairs(store, compute, _in_order(named))
    log.output(store.get(version))
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0, "cached_tokens": 0, "reasoning_tokens": 0}
    _add_usage(usage, r.usage)
    res = _result(system, model, store, version, answer, t, turns=1, calls=1, bad_calls=int(unparsed), usage=usage,
                  gen_ids=[r.gen_id], named=named, unparsed=unparsed, steps=steps)
    log.result(res)
    return res
