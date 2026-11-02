"""The four Task 3 repairs with their gate replaced (Stage 2 plan R6): ②u, every repair on every image (D152), and ⑤,
the oracle (D151). The Task 3 code is called, never changed: each function reads its Task 2 tool as the fix does,
skips only the gate, then calls the fix's own repair step.

- spike: the tool's spikes. The tool lists none below its line, so ②u equals ② here; ⑤ gets the label's positions
  (`ks`) when the tool lists none.
- ghosting: the tool's period. Below its line the tool reports none: ②u takes the period of the strongest A–P rhythm
  the tool computed (rounded, ≥ 2); ⑤ takes the label's number of ghosts (`n`), also when the tool's period is off a
  whole number by more than the fix's 0.1 (planes the fix cannot undo).
- noise: the tool's σ, even below the 3.0 % line; ⑤ gets the label's total σ (`sigma`) only if the tool has none.
- bias: N4's field divided out (the fix's own formula), even below the 0.083 line.
With no number to repair with, the volume comes back unchanged with the reason. Labels never reach this module: ⑤'s
numbers are passed in by the scorer, outside the guard.
"""
import numpy as np

from bamr.fixes import ghosting as fg
from bamr.fixes import noise as fn
from bamr.fixes import spike as fs
from bamr.fixes.bias import _reading
from bamr.fixes.report import FixReport, head_change, unchanged
from bamr.tools import ghosting as tg
from bamr.tools.bias import LINE as BIAS_LINE
from bamr.tools.bias import MIN_BRAIN_ML, MIN_SIDE, brain_mask, n4_logfield
from bamr.tools.finding import PROTOCOL_V2, Protocol, check_volume
from bamr.tools.noise import noise_tool
from bamr.tools.regions import head_mask
from bamr.tools.spike import spike_tool


def _safe(tool: str, artifact: str, x, f):
    try:
        return f()
    except Exception as e:  # no crash inside a run (D68): the error becomes the reason
        return unchanged(x, tool, 1, artifact, f"fix error: {type(e).__name__}: {e}")


def spike_u(x: np.ndarray, protocol: Protocol = PROTOCOL_V2, ks=None) -> tuple[np.ndarray, FixReport]:
    def run():
        f = spike_tool(x, protocol)
        before = {"present": f.present, "best_z": f.evidence.get("best_z"), "spikes": len(f.evidence.get("spikes", [])),
                  "echoes": len(f.evidence.get("echoes", []))}
        if f.evidence.get("spikes"):
            y, rep = fs._repair(np.asarray(x, np.float32), f, before, protocol)
            rep.evidence["source"] = "tool"
            return y, rep
        if ks:
            a = np.asarray(x, np.float32)
            y, info = fs.se_repair(a, ks, head_mask(a), a == 0)
            what = f"{len(ks)} stripe{'s' if len(ks) > 1 else ''} removed at the label's k-space positions"
            return y, FixReport(fs.TOOL, fs.VERSION, fs.ARTIFACT, True, what, f.why, head_change(y, a, head_mask(a)),
                                {"source": "label", "k": ks, **info, "spike_tool_before": before})
        return unchanged(x, fs.TOOL, fs.VERSION, fs.ARTIFACT, f"{f.why}; no spike position, nothing to repair",
                         {"spike_tool_before": before})
    return _safe(fs.TOOL, fs.ARTIFACT, x, run)


def ghosting_u(x: np.ndarray, protocol: Protocol = PROTOCOL_V2, n=None) -> tuple[np.ndarray, FixReport]:
    def run():
        f = tg.ghosting_tool(x, protocol)
        per = f.evidence.get("period_lines")
        ev = {"present_before": f.present, "value_before": f.value, "period_lines": per}
        whole = per is not None and abs(per - round(per)) <= fg.PERIOD_TOL
        if per is not None and (whole or n is None):
            k, source = int(round(per)), "tool"
        elif n is not None:  # ⑤: no period, or one off a whole number (not planes the fix can undo): the label's
            k, source = int(n), "label"
        else:
            a = np.asarray(x, np.float32)
            pv = tg.period_view(tg.spectrum(a), protocol.array_axes.index(protocol.phase_slow))
            if pv["period_lines"] is None:
                return unchanged(x, fg.TOOL, fg.VERSION, fg.ARTIFACT, f"{f.why}; no A–P rhythm at all", ev)
            k, source = int(round(pv["period_lines"])), "rhythm"
            ev["rhythm_period_lines"] = pv["period_lines"]
        if k < 2:
            return unchanged(x, fg.TOOL, fg.VERSION, fg.ARTIFACT, f"{f.why}; period {k}: no planes to undo", ev)
        return fg._fix(x, k, f.why, ev | {"source": source}, protocol)
    return _safe(fg.TOOL, fg.ARTIFACT, x, run)


def noise_u(x: np.ndarray, protocol: Protocol = PROTOCOL_V2, sigma=None) -> tuple[np.ndarray, FixReport]:
    def run():
        f = noise_tool(x, protocol)
        s, source = (f.value, "tool") if f.value is not None else (sigma, "label")
        if s is None or not s > 0:
            return unchanged(x, fn.TOOL, fn.VERSION, fn.ARTIFACT, f"{f.why}; no σ to denoise with")
        y, rep = fn._fix(np.asarray(x).astype(np.float32, copy=False), float(s), f.why, protocol)
        rep.evidence["source"] = source
        return y, rep
    return _safe(fn.TOOL, fn.ARTIFACT, x, run)


def bias_u(x: np.ndarray, protocol: Protocol = PROTOCOL_V2) -> tuple[np.ndarray, FixReport]:
    """fix_bias's own steps (HD-BET brain, N4 in it, division centred over the brain) without the 0.083 line."""
    tool, art = "fix_bias", "bias"

    def run():
        a = np.asarray(x).astype(np.float32, copy=False)
        why = check_volume(a, protocol) or (f"volume {a.shape} is too small for N4's field" if min(a.shape) < MIN_SIDE
                                            else None)
        if why:
            return unchanged(x, tool, 1, art, why)
        brain = brain_mask(a, protocol)
        ml = float(brain.sum()) * protocol.voxel_mm ** 3 / 1000
        if ml < MIN_BRAIN_ML:
            return unchanged(x, tool, 1, art, f"HD-BET found {ml:.0f} ml of brain (need ≥ {MIN_BRAIN_ML} ml)")
        m = brain & (a > 0)
        logf = n4_logfield(a, m, protocol.voxel_mm)
        r0 = _reading(logf, brain)
        y = (a / np.exp(logf - logf[m].mean())).astype(np.float32)
        r1 = _reading(n4_logfield(y, m & (y > 0), protocol.voxel_mm), brain)
        what = f"smooth shading divided out without the {BIAS_LINE} line: N4 reading {r0:.3f} → {r1:.3f}"
        return y, FixReport(tool, 1, art, True, what, f"N4 reads {r0:.3f}", head_change(y, a, head_mask(a)),
                            {"reading_before": r0, "reading_after": r1, "line": BIAS_LINE, "brain_ml": ml,
                             "source": "always"})
    return _safe(tool, art, x, run)


def label_ks(spike_params: dict, shape) -> list[list[int]]:
    """TorchIO spike positions in [0, 1)³ of the (LR, SI, AP) spectrum → k offsets in array axes (LR, AP, SI)."""
    st = np.array([shape[0], shape[2], shape[1]])
    o = np.floor(np.asarray(spike_params["spikes_positions"]) * st).astype(int) - st // 2  # TorchIO's own indexing
    return [[int(v[0]), int(v[2]), int(v[1])] for v in o]


def label_fallbacks(label: dict, shape) -> dict:
    """⑤'s numbers for the repairs whose tool may not measure them, for the labelled artifacts only."""
    p, arts, fb = label.get("params") or {}, label.get("artifacts") or [], {}
    if "spike" in arts:
        fb["spike"] = {"ks": label_ks(p["spike"], shape)}
    if "ghosting" in arts:
        fb["ghosting"] = {"n": int(p["ghosting"]["num_ghosts"])}
    if "noise" in arts:
        fb["noise"] = {"sigma": float(p["noise"]["sigma_total"])}
    return fb
