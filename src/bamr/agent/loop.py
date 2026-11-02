"""The agent ④ (D127, D135, D137, D139): a free ReAct loop (yao2023react) over the twelve functions, on version handles.

The model gets the system prompt and v0's picture, then calls functions and reads the results:
- every call of a turn runs, in order, so a call may name a version an earlier one made (D142);
- a finding, a fix report or a look/compare result as JSON in each tool message;
- the turn's pictures, if any, in one user message after the tool messages, each after "picture for call N" (D135).
Every fix that changes the volume makes a new handle; old ones stay, so it may go on from or return any of them.
finish ends the run, called on its own (with other calls it is refused, they run). Nothing else is said to it but
NUDGE, when a reply calls no function.

The fuse (40 calls or 30 min, D127) returns v0, flagged; it is checked before every call, so one turn cannot pass it. An API error that outlives the retries does the same, with
the error kept. The model never sees a case id, path or array. Every text the harness sends is checked for the
case's forbidden strings, case-id patterns, NIfTI names and absolute paths; a hit raises LeakError and stops the run.
"""
import gzip
import json
import re
import shutil
import time
from pathlib import Path

import nibabel as nib
import numpy as np

from bamr.agent.functions import check, tools
from bamr.agent.openrouter import ApiError
from bamr.agent.pictures import data_url
from bamr.agent.prompts import FIRST, NUDGE, SYSTEM, SYSTEM_EYES
from bamr.agent.report import check_report, provenance
from bamr.agent.store import VersionStore
from bamr.data.blindio import BOX3, nifti_bytes3, write_file

FUSE_CALLS, FUSE_SECONDS = 40, 1800
FINISH_ALONE = "finish ends the run: call it on its own, after you have read the results"
OVER = "not run: the run reached its limit of calls or time"
LEAK = re.compile(r"\b[dst]\d{4}\b|\.nii|(?:^|[\s\"'(])(?:/Users/|/private/|/var/|/tmp/|/home/|~/)")
REPORT = ("report", "repairs", "artifacts", "rescan", "rescan_why")


class LeakError(RuntimeError):
    pass


class CaseLog:
    """messages.jsonl (pictures as files, not bytes), api.jsonl, result.json, output.nii.gz and pictures/."""

    def __init__(self, dir: Path):
        self.dir = Path(dir)
        (self.dir / "pictures").mkdir(parents=True, exist_ok=True)

    def picture(self, src: Path, name: str) -> str:
        shutil.copyfile(src, self.dir / "pictures" / name)
        return f"pictures/{name}"

    def _append(self, file: str, rec: dict) -> None:
        with open(self.dir / file, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def message(self, m: dict, pictures: list[str] = ()) -> None:
        c = m.get("content")
        if isinstance(c, list):
            it = iter(pictures)
            c = [{"type": "image_url", "picture": next(it)} if p.get("type") == "image_url" else p for p in c]
            m = dict(m, content=c)
        self._append("messages.jsonl", m)

    def api(self, rec: dict) -> None:
        self._append("api.jsonl", rec)

    def output(self, y: np.ndarray) -> None:
        p = self.dir / "output.nii.gz"
        if y.shape == BOX3:
            write_file(p, nifti_bytes3(y, np.float32))
        else:  # small volumes (tests): a plain NIfTI
            p.write_bytes(gzip.compress(nib.Nifti1Image(np.asarray(y, np.float32), np.eye(4)).to_bytes(), mtime=0))

    def result(self, d: dict) -> None:
        tmp = self.dir / "result.json.tmp"
        tmp.write_text(json.dumps(d, indent=1, ensure_ascii=False))
        tmp.rename(self.dir / "result.json")  # written last: a case with result.json is done


def _texts(m: dict):
    c = m.get("content")
    if isinstance(c, str):
        yield c
    elif isinstance(c, list):
        yield from (p.get("text", "") for p in c if p.get("type") == "text")


def leak_check(messages: list[dict], forbidden: list[str]) -> None:
    for m in messages:
        for t in _texts(m):
            hit = next((f for f in forbidden if f and f in t), None) or (LEAK.search(t) and LEAK.search(t).group())
            if hit:
                raise LeakError(f"a {m['role']} message would carry {hit!r}")


def keep_pictures(compute, store: VersionStore, version: str, log: CaseLog) -> None:
    """For the page only, never sent to the model: a look at the returned version and its difference from v0."""
    look = compute.call("look", [(store.path(version), store.hash(version))], {"plane": None, "slice": None})
    if look.picture is not None:
        log.picture(look.picture, "returned_look.png")
    if version != "v0":
        cmp = compute.call("compare", [(store.path(version), store.hash(version)), (store.path("v0"), store.hash("v0"))])
        if cmp.picture is not None:
            log.picture(cmp.picture, "returned_vs_v0.png")


def _image(path: Path) -> dict:
    return {"type": "image_url", "image_url": {"url": data_url(path)}}


def _plain_finding(d: dict, picture: bool) -> dict:
    """The tool's JSON for the model: its code version renamed, so it is never read as a handle."""
    d = {("tool_version" if k == "version" else k): v for k, v in d.items()}
    if "picture" in d:
        d["picture"] = "sent in the next message" if picture else None
    return d


def _add_usage(tot: dict, u: dict) -> None:
    tot["prompt_tokens"] += u.get("prompt_tokens") or 0
    tot["completion_tokens"] += u.get("completion_tokens") or 0
    tot["cost"] += u.get("cost") or 0.0
    tot["cached_tokens"] += (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    tot["reasoning_tokens"] += (u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0


class _Run:
    def __init__(self, x0, client, compute, work, log, forbidden, fuse_calls, fuse_seconds, clock, eyes=False):
        self.client, self.compute, self.log, self.forbidden = client, compute, log, forbidden
        self.eyes, self.tools = eyes, tools(eyes)
        self.fuse_calls, self.fuse_seconds, self.clock = fuse_calls, fuse_seconds, clock
        self.store = VersionStore(Path(work) / "versions")
        self.store.add(x0, None, None)
        self.messages: list[dict] = []
        self.unchecked = 0  # messages appended since the last leak check
        self.calls = self.bad = self.turns = 0
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0, "cached_tokens": 0,
                      "reasoning_tokens": 0}
        self.gen_ids: list[str] = []
        self.trace: list[dict] = []  # every call as the model saw its result, for the provenance and the check

    def inp(self, h: str):
        return (self.store.path(h), self.store.hash(h))

    def append(self, m: dict, pictures: list[str] = ()) -> None:
        self.messages.append(m)
        self.log.message(m, pictures)

    def send(self):
        leak_check([m for m in self.messages[len(self.messages) - self.unchecked:] if m["role"] != "assistant"],
                   self.forbidden)
        self.unchecked = 0
        r = self.client.chat(self.messages, self.tools)
        self.turns += 1
        self.gen_ids.append(r.gen_id)
        _add_usage(self.usage, r.usage)
        self.log.api({"turn": self.turns, "gen_id": r.gen_id, "seconds": r.seconds, "finish_reason": r.finish_reason,
                      "usage": r.usage, "tool_calls": len(r.message.get("tool_calls") or [])})
        return r

    def push(self, m: dict, pictures: list[str] = ()) -> None:
        self.append(m, pictures)
        self.unchecked += 1

    def execute(self, n: int, name: str, raw) -> tuple[dict, Path | None, dict | None]:
        """One call → (tool-message JSON, picture to send after, finish arguments or None)."""
        try:
            args = json.loads(raw) if isinstance(raw, str) and raw.strip() else (raw or {})
        except json.JSONDecodeError:
            return {"call": n, "error": "the arguments are not valid JSON"}, None, None
        err = check(name, args, set(self.store.handles()), self.eyes)
        if err:
            return {"call": n, "error": err}, None, None
        if name == "finish":
            return {"call": n, "finished": True}, None, args
        if name == "compare":
            out = self.compute.call("compare", [self.inp(args["a"]), self.inp(args["b"])])
            body = {"call": n, "a": args["a"], "b": args["b"], **out.data}
        else:
            h = args["version"]
            extra = {"plane": args["plane"], "slice": args["slice"], "window": args["window"]} if name == "look" \
                else None
            out = self.compute.call(name, [self.inp(h)], extra)
            if "error" in out.data:
                body = {"call": n, "on": h, **out.data}
            elif name == "look":
                body = {"call": n, "on": h, **out.data}
            elif name.startswith("fix_"):
                new = self.store.add(np.load(out.output), h, name) if out.output else h
                if self.eyes:  # ④a: the version only, never a reading (D150)
                    body = {"call": n, "on": h, "result": new} | ({} if out.output else {"unchanged": True})
                else:
                    body = {"call": n, "on": h, "result": new, "report": _plain_finding(out.data, False)}
            else:
                body = {"call": n, "on": h, "finding": _plain_finding(out.data, out.picture is not None)}
        if "error" in out.data:
            return body, None, None
        return body, out.picture, None


def run_agent(x0: np.ndarray, client, compute, work: Path, log: CaseLog, forbidden: list[str],
              fuse_calls: int = FUSE_CALLS, fuse_seconds: float = FUSE_SECONDS, clock=time.monotonic,
              eyes: bool = False) -> dict:
    """④, or ④a with eyes=True: the same loop without the five measuring tools (D150)."""
    t0, wall = clock(), time.perf_counter()
    r = _Run(x0, client, compute, work, log, forbidden, fuse_calls, fuse_seconds, clock, eyes)
    look = compute.call("look", [r.inp("v0")], {"plane": None, "slice": None})
    r.push({"role": "system", "content": SYSTEM_EYES if eyes else SYSTEM})
    r.push({"role": "user", "content": [{"type": "text", "text": FIRST}, _image(look.picture)]},
           [log.picture(look.picture, "v0_look.png")])
    version, report, fused, error, fin_args = "v0", None, False, None, None
    while True:
        if r.calls >= fuse_calls or clock() - t0 >= fuse_seconds:
            fused = True
            break
        try:
            reply = r.send()
        except ApiError as e:
            error = str(e)
            break
        msg = reply.message
        calls = msg.get("tool_calls") or []
        if not calls:
            if msg.get("content"):
                r.append(msg)  # the model's own words go back; an empty reply does not
            r.calls += 1
            r.bad += 1
            r.push({"role": "user", "content": NUDGE})
            continue
        r.append(msg)
        after, finish = [], None
        alone = len(calls) == 1
        for tc in calls:
            if r.calls >= fuse_calls or clock() - t0 >= fuse_seconds:  # the fuse holds inside a turn too
                fused = True
                r.push({"role": "tool", "tool_call_id": tc.get("id", ""), "content": json.dumps({"error": OVER})})
                continue
            r.calls += 1
            n = r.calls
            fn = tc.get("function") or {}
            if fn.get("name") == "finish" and not alone:
                body, pic, fin = {"call": n, "error": FINISH_ALONE}, None, None
            else:
                body, pic, fin = r.execute(n, fn.get("name", ""), fn.get("arguments"))
            r.trace.append({"fn": fn.get("name", ""), **body})
            if "error" in body:
                r.bad += 1
            r.push({"role": "tool", "tool_call_id": tc.get("id", ""), "content": json.dumps(body, ensure_ascii=False)})
            if pic is not None:
                after.append((n, pic))
            finish = finish or fin
        if fused:
            break
        if finish:
            version, report, fin_args = finish["version"], {k: finish[k] for k in REPORT}, finish
            break
        if after:
            content = [p for n, pic in after for p in ({"type": "text", "text": f"picture for call {n}"}, _image(pic))]
            r.push({"role": "user", "content": content}, [log.picture(pic, f"call_{n:03d}.png") for n, pic in after])
    log.output(r.store.get(version))
    keep_pictures(compute, r.store, version, log)
    lineage = r.store.lineage(version)
    received = [t for m in r.messages if m["role"] != "assistant" for t in _texts(m)]
    received += [t["function"]["description"] for t in r.tools]
    res = {"system": "agent_eyes" if eyes else "agent", "model": getattr(getattr(client, "cfg", None), "slug", None), "version": version,
           "report": report, "lineage": lineage, "turns": r.turns, "calls": r.calls,
           "bad_calls": r.bad, "fused": fused, "error": error, "seconds": round(time.perf_counter() - wall, 1),
           "usage": r.usage, "gen_ids": r.gen_ids, "provenance": provenance(r.trace, lineage),
           "check": check_report(fin_args, r.trace, lineage, received)}
    log.result(res)
    return res
