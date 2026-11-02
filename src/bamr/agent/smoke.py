"""The digit smoke test (D135, route page §7): before any run, per model and route, a picture with a large 7 goes
through the same path as a tool's picture (JSON in the tool message, the picture in the next user message). The run
starts only if the model reads the 7 back; failures on this path are often silent (the model answers without the
picture)."""
import dataclasses
import json
import re
import time
from pathlib import Path

from bamr.agent.openrouter import ApiError
from bamr.agent.pictures import digit
from bamr.agent.loop import _image

DIGIT = 7
SYSTEM = "This is a test of the picture path."
ASK = "Call show_picture, then say which number is drawn in the picture that follows."
TOOL = {"type": "function", "function": {"name": "show_picture", "description": "Shows a picture.",
                                         "parameters": {"type": "object", "properties": {}, "required": [],
                                                        "additionalProperties": False}, "strict": True}}
WORDS = "zero one two three four five six seven eight nine".split()


def passed(text: str) -> bool:
    """7 (or "seven") and no other numeral, leaving out the "call 1" the model may quote and a numeral counting the
    panels ("3 panels", "the 2 side panels"). Number words other than "seven" are not readings: in the replies they
    count the panels ("the three panels", "the other two"); a wrong reading still fails, as it gives no 7. The check
    is that the picture reaches the model (D135)."""
    t = re.sub(r"\bcall \d+\b", "", (text or "").lower())
    t = re.sub(r"\b(\d|[a-z]+)(?: [a-z]+){0,2} panels\b", "panels", t)
    said = {int(d) for d in re.findall(r"\b\d\b", t)} | ({DIGIT} if WORDS[DIGIT] in re.findall(r"\b[a-z]+\b", t)
                                                          else set())
    return said == {DIGIT}


def run_smoke(client, work: Path) -> dict:
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    pic = work / "digit.png"
    digit(DIGIT, pic)
    t = time.perf_counter()
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": ASK}]
    out = {"pass": False, "text": "", "reason": "", "usage": [], "gen_ids": [], "finish": []}
    try:
        r = client.chat(msgs, [TOOL])
        out["usage"].append(r.usage), out["gen_ids"].append(r.gen_id), out["finish"].append(r.finish_reason)
        calls = r.message.get("tool_calls") or []
        if not calls:
            out["reason"], out["text"] = "no function call", r.message.get("content") or ""
            return out
        msgs.append(r.message)
        msgs += [{"role": "tool", "tool_call_id": c.get("id", ""), "content": json.dumps({"shown": True})}
                 for c in calls]
        msgs.append({"role": "user", "content": [{"type": "text", "text": "picture for call 1"}, _image(pic)]})
        r = client.chat(msgs, [TOOL])
        out["usage"].append(r.usage), out["gen_ids"].append(r.gen_id), out["finish"].append(r.finish_reason)
        out["text"] = r.message.get("content") or ""
        out["pass"] = passed(out["text"])
        out["reason"] = "read the digit" if out["pass"] else "did not read the digit back"
    except ApiError as e:
        out["reason"] = f"API error: {e}"
    finally:
        out["seconds"] = round(time.perf_counter() - t, 1)
    return out


def same_settings(saved: dict, cfg) -> bool:
    """The model settings a smoke file was written with equal these (compared as JSON: tuples come back as lists)."""
    return saved == json.loads(json.dumps(dataclasses.asdict(cfg)))
