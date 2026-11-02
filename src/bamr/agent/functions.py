"""The twelve functions the model may call (before-doc §5, D137) and the router's one (§8), as OpenAI-format tools.

finish carries the final report (Stage 2, D147): free text under a radiologist's headings, the findings one per artifact
(each begins with present, absent or uncertain: structured reporting, esr2023structured), then the repairs in the
returned version's history, the yes/no answers, the rescan and the version.

Each description is the tool's card, word for word (prompts.CARDS). Strict schemas: every field is required and the
optional ones are nullable, so one shape works on every provider. A version is a handle (v0, v1, …), never a path.
`check` turns a bad call into a reason for the model instead of a crash (the D68 rule for the loop).
"""
from bamr.agent.prompts import CARDS, CARDS_EYES

ARTIFACTS = ("noise", "spike", "ghosting", "bias", "motion")
FIXES = ("fix_spike", "fix_ghosting", "fix_noise", "fix_bias")
PLANES = ("axial", "coronal", "sagittal")
_VERSION = {"type": "string", "description": "a version handle: v0, v1, …"}
_YES_NO = {"type": "string", "enum": ["yes", "no"]}


def _obj(props: dict) -> dict:
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def _text(d: str) -> dict:
    return {"type": "string", "description": d}


_ANSWERS = {
    "artifacts": dict(_obj({a: dict(_YES_NO) for a in ARTIFACTS}),
                      description="each artifact in the volume as received: yes or no"),
    "rescan": dict(_YES_NO, description="whether motion makes a rescan necessary"),
    "rescan_why": _text("one line: why"),
}
_REPORT = {"seen": _text("what you saw and measured, in your own words"), **_ANSWERS}  # the routers' answer
_WRITTEN = _obj({
    "examination": _text("what was examined and how: which versions you looked at and measured"),
    "findings": dict(_obj({a: _text(f"{a}: begin with present, absent or uncertain, then what you saw and measured")
                           for a in ARTIFACTS}), description="one finding per artifact, in the volume as received"),
    "what_was_done": _text("each repair in order, the version it made and what it changed; any repair not kept"),
    "impression": _text("the conclusion, in a sentence or two"),
    "recommendation": _text("what should happen next, and why"),
    "limits": _text("what you could not tell, and why"),
})
PARAMS = {
    "look": _obj({"version": _VERSION,
                  "plane": {"type": ["string", "null"], "enum": [*PLANES, None],
                            "description": "one plane only, or null for all three"},
                  "slice": {"type": ["integer", "null"], "description": "slice number in that plane, or null for the "
                                                                         "middle of the head"},
                  "window": {"type": ["string", "null"], "enum": ["head", "air", None],
                             "description": "head (null: 0 to 1.35 × white matter) or air (0 to 0.1)"}}),
    "compare": _obj({"a": _VERSION, "b": _VERSION}),
    **{name: _obj({"version": _VERSION}) for name in ("noise_tool", "spike_tool", "ghosting_tool", "bias_tool",
                                                       "motion_tool", "fix_spike", "fix_ghosting", "fix_noise",
                                                       "fix_bias")},
    "finish": _obj({"report": dict(_WRITTEN, description="your report, as a radiologist writes one"),
                    "repairs": {"type": "array", "items": {"type": "string", "enum": list(FIXES)},
                                "description": "the repairs in the history of the version you return, in order"},
                    **_ANSWERS, "version": dict(_VERSION, description="the version you return")}),
}
TOOLS = [{"type": "function", "function": {"name": name, "description": CARDS[name], "parameters": PARAMS[name],
                                           "strict": True}} for name in CARDS]
TOOLS_EYES = [{"type": "function", "function": {"name": name, "description": CARDS_EYES[name],
                                                "parameters": PARAMS[name], "strict": True}} for name in CARDS_EYES]


def tools(eyes: bool = False) -> list[dict]:
    """④'s twelve functions, or ④a's seven (D150: no measuring tool)."""
    return TOOLS_EYES if eyes else TOOLS
ANSWER = {"type": "function", "function": {"name": "answer", "description": "Your answer.",
                                           "parameters": _obj(dict(_REPORT)), "strict": True}}
HANDLE_FIELDS = ("version", "a", "b")


def _check_value(path: str, v, s: dict) -> str | None:
    types = s["type"] if isinstance(s["type"], list) else [s["type"]]
    if v is None:
        return None if "null" in types else f"{path} must not be null"
    if "object" in types:
        if not isinstance(v, dict):
            return f"{path} must be an object with {', '.join(s['properties'])}"
        return _check_object(path, v, s)
    if "array" in types:
        if not isinstance(v, list):
            return f"{path} must be a list"
        return next((e for i, x in enumerate(v) if (e := _check_value(f"{path}[{i}]", x, s["items"]))), None)
    if "integer" in types and not (isinstance(v, int) and not isinstance(v, bool)):
        return f"{path} must be a whole number"
    if "string" in types and not isinstance(v, str):
        return f"{path} must be a string"
    if "enum" in s and v not in s["enum"]:
        return f"{path} must be one of {', '.join(str(e) for e in s['enum'] if e is not None)}"
    return None


def _check_object(path: str, args: dict, s: dict) -> str | None:
    pre = f"{path}." if path else ""
    for k in s["required"]:
        if k not in args:
            return f"missing argument: {pre}{k}"
    for k in args:
        if k not in s["properties"]:
            return f"unexpected argument: {pre}{k}"
    for k, sub in s["properties"].items():
        e = _check_value(pre + k, args[k], sub)
        if e:
            return e
    return None


def form(s: dict) -> str:
    """The shape of a function's arguments in one line: report {examination, findings {noise, …}, …}, repairs [...]."""
    def one(k, v):
        types = v["type"] if isinstance(v["type"], list) else [v["type"]]
        if "object" in types:
            return f"{k} {{{form(v)}}}"
        if "array" in types:
            return f"{k} [{', '.join(v['items'].get('enum', []))}]"
        return k
    return ", ".join(one(k, v) for k, v in s["properties"].items())


def check(name: str, args, handles: set[str], eyes: bool = False) -> str | None:
    """Why this call cannot run, or None. An argument error names the whole form (Stage 2 r01): one retry can fix it."""
    names = list(CARDS_EYES) if eyes else list(PARAMS)
    if name not in names:
        return f"unknown function {name!r}; the functions are: {', '.join(names)}"
    if not isinstance(args, dict):
        return f"the arguments must be a JSON object; {name} takes: {form(PARAMS[name])}"
    e = _check_object("", args, PARAMS[name])
    if e:
        return f"{e}; {name} takes: {form(PARAMS[name])}"
    for k in HANDLE_FIELDS:
        if k in args and args[k] not in handles:
            return f"unknown version {args[k]!r}; the versions are: {', '.join(sorted(handles, key=_order))}"
    return None


def _order(h: str):
    return int(h[1:]) if h[1:].isdigit() else 10 ** 9


def check_answer(args) -> str | None:
    """Why the router's answer cannot be read, or None."""
    if not isinstance(args, dict):
        return "the arguments must be a JSON object"
    return _check_object("", args, ANSWER["function"]["parameters"])
