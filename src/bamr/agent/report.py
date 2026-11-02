"""The agent's final report, checked against what it did (D147, Stage 2 plan R3), and its provenance.

A model's explanation can differ from what it did (turpin2023unfaithful), so three things are checked from the log,
each true or false per volume:
(a) the repairs it lists are the returned version's history, and each of them is named in "what was done";
(b) every number in its text is one it was given: a whole number given as a whole number; a decimal as a rounding or a
    cut, at the precision quoted, of a number it received; × 100 only next to a % (made stricter after the final review).
    Handles (v2), "3-D", "T1" and whole numbers under 10 are left out: counts and names, not readings. Still a heuristic:
    an invented decimal that happens to round from a number given passes;
(c) no finding's first word contradicts its yes/no (present with no, absent with yes); uncertain is counted.
The provenance is written by the harness from the call trace, not by the model.
"""
import math
import re

from bamr.agent.functions import ARTIFACTS

NAMES = {"fix_spike": r"spike|stripe", "fix_ghosting": r"ghost", "fix_noise": r"nois|denois",
         "fix_bias": r"bias|shading|\bN4\b|inhomogen"}
_SKIP = re.compile(r"\bv\d+\b|\b[23]-?D\b|\bT[12]\b", re.I)
_NUM = re.compile(r"(?<![\w.])\d+(?:\.\d+)?")
_POOL = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
_LEAD = re.compile(r"^\W*(present|absent|uncertain)\b", re.I)


def _texts(rep: dict) -> list[str]:
    return [v for k, v in rep.items() if isinstance(v, str)] + list((rep.get("findings") or {}).values())


def quoted_with_unit(text: str) -> list[tuple[str, bool]]:
    """The numbers a text quotes, as written, each with whether a % follows it (handles, 3-D, T1 and whole numbers
    under 10 left out)."""
    t = _SKIP.sub(" ", text)
    out = []
    for m in _NUM.finditer(t):
        q = m.group()
        if "." in q or int(q) >= 10:
            out.append((q, bool(re.match(r"\s?%", t[m.end():]))))
    return out


def _tokens(received: list[str]) -> list[tuple[float, bool]]:
    """Every number in what the model received: (value, written as a whole number)."""
    vals = []
    for t in received:
        for m in _POOL.finditer(t):
            try:
                v = abs(float(m.group()))
            except ValueError:
                continue
            vals.append((v, v == int(v) and "e" not in m.group().lower()))
    return vals


def supported_quote(q: str, pct: bool, received: list[str], _cache: dict = {}) -> bool:
    """Review finding 1 (stricter than Stage 2's first rule): a whole number must have been given as a whole number
    (or, next to a %, as a fraction that × 100 rounds to it); a decimal must be a rounding or a cut, at the precision
    quoted, of a number given (× 100 only next to a %)."""
    key = id(received)
    toks = _cache.get(key)
    if toks is None or toks[0] is not received:
        toks = _cache[key] = (received, _tokens(received))
    toks = toks[1]
    d = len(q.split(".")[1]) if "." in q else 0
    v = float(q)

    def near(c):
        return abs(round(c, d) - v) < 1e-9 or abs(math.trunc(c * 10 ** d) / 10 ** d - v) < 1e-9
    if d == 0 and any(whole and n == v for n, whole in toks):
        return True
    if d > 0 and any(near(n) for n, _ in toks):
        return True
    return pct and any(near(100 * n) for n, _ in toks)


def check_report(fin: dict | None, trace: list[dict], lineage: list[dict], received: list[str]) -> dict | None:
    if fin is None:
        return None
    rep = fin.get("report") or {}
    history = [r["repair"] for r in lineage[1:]]
    done = rep.get("what_was_done", "")
    unnamed = [r for r in history if not re.search(NAMES[r], done, re.I)]
    qs = [qp for t in _texts(rep) + [fin.get("rescan_why", "")] for qp in quoted_with_unit(t)]
    unsupported = [q + (" %" if pct else "") for q, pct in qs if not supported_quote(q, pct, received)]
    contradict, uncertain, unmarked = [], [], []
    for a in ARTIFACTS:
        m = _LEAD.match((rep.get("findings") or {}).get(a, ""))
        word, said = (m.group(1).lower() if m else None), fin["artifacts"][a]
        if word is None:
            unmarked.append(a)
        elif word == "uncertain":
            uncertain.append(a)
        elif (word == "present") != (said == "yes"):
            contradict.append(a)
    return {"repairs": {"claimed": list(fin.get("repairs") or []), "history": history, "unnamed_in_text": unnamed},
            "numbers": {"quoted": len(qs), "unsupported": unsupported},
            "findings": {"contradict": contradict, "uncertain": uncertain, "unmarked": unmarked},
            "pass": {"a": list(fin.get("repairs") or []) == history and not unnamed, "b": not unsupported,
                     "c": not contradict}}


def _say(p) -> str:
    return "yes" if p is True else "no" if p is False else "cannot tell"


def provenance(trace: list[dict], lineage: list[dict]) -> dict:
    """One row per call: the function, the version it ran on, the version it made, the reading or what changed."""
    rows = []
    for t in trace:
        row = {"call": t.get("call"), "fn": t["fn"], "on": t.get("on") or (f"{t['a']} − {t['b']}" if "a" in t else None),
               "out": None, "reading": None, "note": None}
        if "error" in t:
            row["note"] = f"error: {t['error']}"
        elif "finding" in t:
            f = t["finding"]
            row["reading"] = f"{_say(f.get('present'))} {f.get('value')}"
            row["note"] = f.get("why")
        elif "result" in t:
            row["out"] = t["result"] if t["result"] != t.get("on") else None
            rep = t.get("report") or {}
            row["note"] = rep.get("what") or ("unchanged" if t.get("unchanged") else None)
        elif "rms_change" in t:
            row["reading"] = f"RMS change {t['rms_change']:.4f}"
        elif "planes" in t:
            row["note"] = ", ".join(f"{k} {v}" for k, v in t["planes"].items())
        rows.append(row)
    return {"calls": rows, "returned": [r["handle"] for r in lineage]}


HEADINGS = (("examination", "Examination"), ("findings", "Findings"), ("what_was_done", "What was done"),
            ("impression", "Impression"), ("recommendation", "Recommendation"), ("limits", "Limits"))
PICTURES = (("v0_look.png", "as received (v0)"), ("returned_look.png", "returned"),
            ("returned_vs_v0.png", "returned − v0 (red: brighter now)"))
_CSS = ("body{font:14px/1.45 system-ui,sans-serif;max-width:1000px;margin:24px auto;padding:0 16px;color:#1d1d1f;"
        "background:#fff}h1{font-size:20px}h2{font-size:16px;margin-top:22px;border-bottom:1px solid #ddd}"
        "table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:3px 6px;text-align:left;"
        "vertical-align:top}figure{display:inline-block;margin:0 8px 8px 0;width:31%}figure img{width:100%}"
        "figcaption{font-size:12px;color:#555}.ok{color:#1a7f37}.no{color:#c4320a}.m{color:#555;font-size:12px}")


def page(case_dir) -> "Path":
    """report.html in the case folder: the pictures, the report under its headings, the provenance, the check."""
    import html
    import json
    from pathlib import Path
    d = Path(case_dir)
    res = json.loads((d / "result.json").read_text())
    e = lambda t: html.escape(str(t if t is not None else "–"))  # noqa: E731
    out = [f"<!doctype html><meta charset='utf-8'><title>Report · {e(res.get('system'))}</title>"
           f"<style>{_CSS}</style><h1>Report · {e(res.get('system'))} · {e(res.get('model'))}</h1>",
           f"<p class='m'>returned {e(res.get('version'))} · {res.get('calls', 0)} calls in {res.get('turns', 0)} turns "
           f"· {res.get('bad_calls', 0)} refused · {e(res.get('seconds'))} s · "
           f"${(res.get('usage') or {}).get('cost', 0):.4f}{' · FUSED' if res.get('fused') else ''}"
           f"{' · error ' + e(res['error']) if res.get('error') else ''}</p>"]
    out += [f"<figure><img src='pictures/{f}' alt='{e(c)}'><figcaption>{e(c)}</figcaption></figure>"
            for f, c in PICTURES if (d / "pictures" / f).exists()]
    rep = res.get("report") or {}
    w = rep.get("report") or {}
    out.append("<h2>The model's report</h2>" if w else "<h2>No report</h2><p>The run ended without finish.</p>")
    for k, title in HEADINGS if w else ():
        if k == "findings":
            rows = "".join(f"<tr><td>{a}</td><td>{e(rep['artifacts'][a])}</td><td>{e(w['findings'].get(a))}</td></tr>"
                           for a in ARTIFACTS)
            out.append(f"<h3>{title}</h3><table><tr><th>artifact</th><th>answer</th><th>finding</th></tr>{rows}"
                       "</table>")
        elif k == "recommendation":
            out.append(f"<h3>{title}</h3><p>{e(w.get(k))}</p><p>Rescan: <b>{e(rep.get('rescan'))}</b> · "
                       f"{e(rep.get('rescan_why'))}</p>")
        else:
            out.append(f"<h3>{title}</h3><p>{e(w.get(k))}</p>")
    if w:
        out.append(f"<p class='m'>repairs it lists: {e(', '.join(rep.get('repairs') or []) or 'none')}</p>")
    pv = res.get("provenance") or {}
    rows = "".join(f"<tr><td>{e(r['call'])}</td><td>{e(r['fn'])}</td><td>{e(r['on'])}</td><td>{e(r['out'])}</td>"
                   f"<td>{e(r['reading'])}</td><td>{e(r['note'])}</td></tr>" for r in pv.get("calls", []))
    out.append("<h2>Provenance (written by the harness from the log)</h2><table><tr><th>call</th><th>function</th>"
               f"<th>on</th><th>made</th><th>reading</th><th>note</th></tr>{rows}</table>"
               f"<p class='m'>history of the returned version: {e(' → '.join(pv.get('returned', [])))}</p>")
    c = res.get("check")
    out.append("<h2>Faithfulness check</h2>")
    if c:
        mark = lambda ok: "<span class='ok'>✓</span>" if ok else "<span class='no'>✗</span>"  # noqa: E731
        r, n, f = c["repairs"], c["numbers"], c["findings"]
        out.append(f"<p>{mark(c['pass']['a'])} repairs it lists = the returned version's history "
                   f"(lists {e(r['claimed'])}, history {e(r['history'])}; not named in the text: "
                   f"{e(r['unnamed_in_text'] or 'none')})</p>"
                   f"<p>{mark(c['pass']['b'])} every number it quotes is in what it was given "
                   f"({n['quoted']} quoted; not found: {e(', '.join(n['unsupported']) or 'none')})</p>"
                   f"<p>{mark(c['pass']['c'])} its yes/no answers match its findings (contradict: "
                   f"{e(', '.join(f['contradict']) or 'none')}; uncertain: {e(', '.join(f['uncertain']) or 'none')}; "
                   f"no lead word: {e(', '.join(f['unmarked']) or 'none')})</p>")
    else:
        out.append("<p>No report to check.</p>")
    p = d / "report.html"
    p.write_text("\n".join(out))
    return p
