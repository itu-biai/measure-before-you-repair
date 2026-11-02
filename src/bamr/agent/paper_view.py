"""The report's paper view (D159): one short card per finished run, readable without our labels.

Nothing new is asked of the model. The card keeps the model's own words (its answer per artifact, a short note from each
finding, its impression and recommendation) with version labels and function names replaced by what had been done
("the original scan", "the scan after spike removal", "the final scan"). What was repaired and what was measured on the
final scan come from the run log, not from the model. The full report (report.page) stays as written.
"""
import html
import json
import re
from pathlib import Path

from bamr.agent.functions import ARTIFACTS

FIX = {"spike": "fix_spike", "ghosting": "fix_ghosting", "noise": "fix_noise", "bias": "fix_bias"}
NOUN = {"fix_spike": "spike removal", "fix_ghosting": "ghost removal", "fix_noise": "denoising",
        "fix_bias": "bias correction"}
LABEL = {"noise": "Noise (grain)", "spike": "Spike (stripes)", "ghosting": "Ghosting (faint copies of the head)",
         "bias": "Bias (uneven brightness)", "motion": "Motion (head movement)"}
NOTE_SENTENCES, NOTE_CHARS = 2, 260
# a label with the model's own words for it in front ("the restored volume v4", "version v2"), which the name replaces
_HANDLE = re.compile(r"(?:\b(?:the|this)\s+)?(?:\b(?:original|restored|repaired|final|returned|received|denoised|"
                     r"corrected|current|last|latest|new|previous|raw|unfiltered|unrepaired|uncorrected|input|initial|"
                     r"earlier|intermediate|resulting)\s+)?(?:\b(?:volume|version|scan|image)\s+)?\b[vV](\d+)\b", re.I)
_FIX_ON = re.compile(r"\b(fix_(?:spike|ghosting|noise|bias))\s+(?=[vV]\d+\b)")  # "fix_spike v0 unchanged"
_BRACKETED = re.compile(r"\s*\(\s*[vV]\d+\s*\)|(?<=\()\s*fix_\w+\s*,\s*|\s*\(\s*fix_\w+\s*\)")  # said already
_FIXNAME = re.compile(r"\bfix_(spike|ghosting|noise|bias)\b")
_TOOLNAME = re.compile(r"\b(noise|spike|ghosting|bias|motion)_tool\b")
_LEAD = re.compile(r"^\W*(present|absent|uncertain)\b", re.I)
_SENT = re.compile(r"(?<=[.;!?])\s+(?=[A-Z])")


def _join(words: list[str]) -> str:
    return words[0] if len(words) == 1 else ", ".join(words[:-1]) + " and " + words[-1]


def version_names(calls: list[dict], lineage: list[dict]) -> dict[str, str]:
    """Every version made in the run, named by the repairs that led to it from the original."""
    parent = {r["handle"]: (r["parent"], r["repair"]) for r in lineage[1:]}
    parent |= {c["out"]: (c["on"], c["fn"]) for c in calls if c["fn"] in NOUN and c.get("out")}

    def steps(h):
        return [] if h not in parent else steps(parent[h][0]) + [NOUN[parent[h][1]]]
    names = {h: f"the scan after {_join(steps(h))}" for h in parent}
    names["v0"] = "the original scan"
    if len(lineage) > 1:
        names[lineage[-1]["handle"]] = "the final scan"
    return names


def plain(text: str | None, names: dict[str, str]) -> str:
    """The model's text with version labels, function names, "A–P" and "WM" written out."""
    def cap(m, word):
        start = m.start() == 0 or re.search(r"[.!?:]\s*$", m.string[:m.start()])
        return word[0].upper() + word[1:] if start else word
    t = _BRACKETED.sub("", text or "")
    t = _FIX_ON.sub(r"\1 on ", t)
    t = _HANDLE.sub(lambda m: cap(m, names.get(f"v{m.group(1)}", "an earlier version")), t)
    t = _FIXNAME.sub(lambda m: cap(m, NOUN[m.group()]), t)
    t = _TOOLNAME.sub(lambda m: cap(m, f"the {m.group(1)} tool"), t)
    t = re.sub(r"\bA\s?[-–]\s?P\b", "front–back", t)  # the phase-encoding direction, in plain words
    return re.sub(r"\bWM\b", "white matter", t)


def _note(finding: str, names: dict[str, str]) -> str:
    """The finding without its lead word (a short qualifier kept), at most two sentences or about 260 characters."""
    s = _SENT.split(plain(finding, names).strip())
    first = _LEAD.sub("", s[0]).strip(" ,;:.") if s and _LEAD.match(s[0]) else None
    if first is not None:
        s = s[1:]
        if first and len(first.split()) <= 4:
            s = [first[0].upper() + first[1:] + "."] + s
    keep = s[:NOTE_SENTENCES + (1 if first else 0)]
    while len(keep) > 1 and len(" ".join(keep)) > NOTE_CHARS:
        keep = keep[:-1]
    return " ".join(keep) + (" …" if len(keep) < len(s) else "")


def _final_check(a: str, calls: list[dict], lineage: list[dict]) -> str:
    final = lineage[-1]["handle"]
    seen = [c["reading"] or "" for c in calls if c["fn"] == f"{a}_tool" and c["on"] == final]
    if seen:
        r = seen[-1]
        return "still there" if r.startswith("yes") else "clear" if r.startswith("no") else "cannot tell"
    if len(lineage) > 1 and lineage[-1]["repair"] == FIX.get(a):
        return "measured by its repair"
    return "not measured on the final scan"


def rows(res: dict) -> list[dict]:
    """Per artifact: the model's answer and note; the repair kept and the check on the final scan, from the log."""
    rep = res.get("report") or {}
    w = rep.get("report") or {}
    calls = (res.get("provenance") or {}).get("calls", [])
    lineage = res["lineage"]
    names = version_names(calls, lineage)
    kept = [r["repair"] for r in lineage[1:]]
    out = []
    for a in ARTIFACTS:
        f = (w.get("findings") or {}).get(a, "")
        m = _LEAD.match(f)
        word = m.group(1).lower() if m else ("present" if (rep.get("artifacts") or {}).get(a) == "yes" else "absent")
        fix = FIX.get(a)
        out.append({"artifact": a, "label": LABEL[a],
                    "answer": {"present": "found", "absent": "not found", "uncertain": "uncertain"}[word],
                    "repair": ("no repair exists: the rescan decision" if a == "motion" else
                               f"step {kept.index(fix) + 1}: {NOUN[fix]}" if fix in kept else None),
                    "final": _final_check(a, calls, lineage), "note": _note(f, names)})
    return out


_CSS = ("body{font:14px/1.45 system-ui,sans-serif;max-width:980px;margin:20px auto;padding:0 16px;color:#1d1d1f;"
        "background:#fff}h1{font-size:19px;margin:0 0 2px}h2{font-size:14px;margin:16px 0 4px;text-transform:uppercase;"
        "letter-spacing:.04em;color:#555}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #ddd;"
        "padding:5px 6px;text-align:left;vertical-align:top}th{font-size:12px;color:#555}figure{display:inline-block;"
        "margin:0 8px 6px 0;width:31.5%}figure img{width:100%}figcaption,.m{font-size:12px;color:#555}"
        ".found{color:#b3261e;font-weight:600}.clear{color:#1a7f37}.warn{color:#9a6700}.ok{color:#1a7f37}"
        ".no{color:#c4320a}.rescan{font-size:15px;font-weight:600}")
PICTURES = (("v0_look.png", "the original scan"), ("returned_look.png", "the final scan"),
            ("returned_vs_v0.png", "final − original (red: brighter now)"))
CHECKS = (("a", "the repairs it lists are the ones applied"), ("b", "every number it quotes was measured"),
          ("c", "its yes/no answers match its findings"))


def card(case_dir) -> Path:
    """card.html in the case folder: the paper view of the run."""
    d = Path(case_dir)
    res = json.loads((d / "result.json").read_text())
    e = lambda t: html.escape(str(t if t is not None else "–"))  # noqa: E731
    out = [f"<!doctype html><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
           f"<title>Scan Report Card</title><style>{_CSS}</style><h1>Automated MRI quality report</h1>"
           f"<p class='m'>volume {e(d.name)} · T1-weighted brain, 3-D · agent model {e(res.get('model'))}</p>"]
    out += [f"<figure><img src='pictures/{f}' alt='{e(c)}'><figcaption>{e(c)}</figcaption></figure>"
            for f, c in PICTURES if (d / "pictures" / f).exists()]
    rep = res.get("report") or {}
    w = rep.get("report") or {}
    if not w:
        why = "it stopped at the fuse (call or time limit)" if res.get("fused") else "it ended without finishing"
        out.append(f"<p><b>The agent wrote no report</b>: {why}. The scan is returned as it was.</p>")
        p = d / "card.html"
        p.write_text("\n".join(out))
        return p
    calls = (res.get("provenance") or {}).get("calls", [])
    names = version_names(calls, res["lineage"])
    cls = {"found": "found", "clear": "clear", "still there": "found", "uncertain": "warn", "cannot tell": "warn",
           "not measured on the final scan": "warn"}
    body = "".join(f"<tr><td>{e(r['label'])}</td><td class='{cls.get(r['answer'], '')}'>{e(r['answer'])}</td>"
                   f"<td>{e(r['repair'])}</td><td class='{cls.get(r['final'], '')}'>{e(r['final'])}</td>"
                   f"<td>{e(r['note'])}</td></tr>" for r in rows(res))
    out.append("<h2>Findings</h2><table><tr><th>artifact</th><th>agent's answer</th><th>repair kept</th>"
               f"<th>checked on the final scan</th><th>the agent's note</th></tr>{body}</table>")
    steps = [f"<li>{e(NOUN[r['repair']][0].upper() + NOUN[r['repair']][1:])}</li>" for r in res["lineage"][1:]]
    n = lambda f: sum(1 for c in calls if f(c["fn"]))  # noqa: E731
    tried = n(lambda f: f in NOUN)
    out.append("<h2>What the agent did</h2>" + (f"<ol>{''.join(steps)}</ol>" if steps else "<p>No repair kept: the "
               "scan is returned as it was.</p>") +
               f"<p class='m'>{n(lambda f: f.endswith('_tool'))} measurements, {n(lambda f: f == 'look')} looks at the "
               f"images, {tried} repairs tried, {len(steps)} kept.</p>")
    out.append(f"<h2>Impression</h2><p>{e(plain(w.get('impression'), names))}</p>")
    out.append(f"<h2>Recommendation</h2><p class='rescan'>Rescan: {e(rep.get('rescan'))}</p>"
               f"<p>{e(plain(rep.get('rescan_why'), names))}</p><p>{e(plain(w.get('recommendation'), names))}</p>")
    c = (res.get("check") or {}).get("pass") or {}
    mark = lambda ok: "<span class='ok'>✓</span>" if ok else "<span class='no'>✗</span>"  # noqa: E731
    out.append("<h2>Checked against the run log</h2><p>" +
               " · ".join(f"{mark(c.get(k))} {t}" for k, t in CHECKS) + "</p>"
               "<p class='m'>The answers, notes, impression and recommendation are the agent's words, shortened, with "
               "version labels and function names replaced by what had been done. The repair kept and the check on "
               "the final scan come from the run log.</p>")
    p = d / "card.html"
    p.write_text("\n".join(out))
    return p
