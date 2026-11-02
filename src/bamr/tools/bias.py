"""The bias tool (D102): N4 inside the brain, and how far its smooth field spreads.

Brain: HD-BET on the input (D101, `bamr.tools.brain`). N4 (tustison2010; sled1998's N3 made robust): a B-spline
field fitted so the brain's intensity histogram sharpens; 3 fitting levels, SimpleITK defaults otherwise (D24: 50
iterations at each, convergence 0.001, 4 control points), on a 4× shrunk copy inside the brain. Our references were
N4-corrected the same way (D14), so on a clean brain N4 finds almost nothing; on an input it finds the added field.
Reading: e^(max − min of its log field over the brain) − 1, the label's own form (D86); a bias if ≥ 0.083.
Value: reading ÷ one scale factor (N4 recovers ~0.6 of the field on real brains, ~0.9 on a phantom).
Sides: the brightest and darkest 5 % of the field, seen from the brain's centre, named by the protocol's orientation.
"""
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from bamr.tools.brain import brain_mask
from bamr.tools.finding import PROTOCOL_V2, Finding, Protocol, by_line, check_volume

VERSION = 1
UNIT = "max/min − 1 of the shading over the brain"
LINE = 0.083  # D102: geometric middle of the dev sweep's gap (highest reading without bias 0.074, lowest with 0.093)
LEVELS = 3
ITERATIONS = 50
SHRINK = 4
NEAR = 0.10  # reading within ±10 % of the line → low confidence
MIN_BRAIN_ML = 100
TAIL = 0.05  # the brightest / darkest 5 % of the brain's field give the sides
MIN_SIDE = 32
SCALE = 0.595  # D102: median reading/target_maxmin on the 30 dev-sweep bias volumes (pilot; score_tools.py calibrate)
WORDS = {"R": "right", "L": "left", "A": "front", "P": "back", "S": "top", "I": "bottom"}


def _cannot(why: str, evidence: dict | None = None) -> Finding:
    return Finding("bias", VERSION, "bias", None, None, UNIT, "low", why, evidence or {}, None)


def _confidence(reading: float) -> str:
    return "low" if abs(reading / LINE - 1) < NEAR else "high"


def n4_logfield(x: np.ndarray, mask: np.ndarray, voxel_mm: float) -> np.ndarray:
    """N4's log bias field at full size, fitted on a shrunk copy inside `mask` (array axes in, array axes out)."""
    img = sitk.GetImageFromArray(np.ascontiguousarray(np.maximum(x, 0).transpose(2, 1, 0)).astype(np.float32))
    img.SetSpacing([float(voxel_mm)] * 3)
    m = sitk.GetImageFromArray(np.ascontiguousarray(mask.transpose(2, 1, 0)).astype(np.uint8))
    m.CopyInformation(img)
    f = sitk.N4BiasFieldCorrectionImageFilter()
    f.SetMaximumNumberOfIterations([ITERATIONS] * LEVELS)
    f.Execute(sitk.Shrink(img, [SHRINK] * 3), sitk.Shrink(m, [SHRINK] * 3))
    return sitk.GetArrayFromImage(f.GetLogBiasFieldAsImage(img)).transpose(2, 1, 0)


def _side(offset: np.ndarray, protocol: Protocol) -> str:
    """The anatomical letter of the offset's biggest component: the orientation's letter if it points toward the end
    the axis grows to, the axis' other letter if away."""
    i = int(np.argmax(np.abs(offset)))
    grow = protocol.orientation[i]
    return grow if offset[i] > 0 else protocol.array_axes[i].replace(grow, "")


def sides(logf: np.ndarray, brain: np.ndarray, protocol: Protocol) -> dict:
    v = logf[brain]
    idx = np.argwhere(brain).astype(np.float64)
    centre = idx.mean(axis=0)
    hi, lo = v >= np.quantile(v, 1 - TAIL), v <= np.quantile(v, TAIL)
    b = (idx[hi].mean(axis=0) - centre) * protocol.voxel_mm
    d = (idx[lo].mean(axis=0) - centre) * protocol.voxel_mm
    return {"bright_side": _side(b, protocol), "dark_side": _side(d, protocol), "bright_offset_mm": b.tolist(),
            "dark_offset_mm": d.tolist()}


def bias_tool(x: np.ndarray, protocol: Protocol = PROTOCOL_V2, picture: Path | None = None) -> Finding:
    x = np.asarray(x)
    why = check_volume(x, protocol)
    if not why and min(x.shape) < MIN_SIDE:
        why = f"volume {x.shape} is too small for N4's field (need ≥ {MIN_SIDE} samples per side)"
    if why:
        return _cannot(why)
    try:
        return _measure(x.astype(np.float32, copy=False), protocol, picture)
    except Exception as e:  # no crash inside an agent loop (D68): the error becomes the reason
        return _cannot(f"tool error: {type(e).__name__}: {e}")


def _measure(x: np.ndarray, protocol: Protocol, picture: Path | None) -> Finding:
    brain = brain_mask(x, protocol)
    ml = float(brain.sum()) * protocol.voxel_mm ** 3 / 1000
    if not brain.any():
        return _cannot("no brain found (HD-BET), so no shading over it to measure")
    if ml < MIN_BRAIN_ML:
        return _cannot(f"HD-BET found only {ml:.0f} ml of brain (need ≥ {MIN_BRAIN_ML} ml for N4's field)",
                       {"brain_ml": ml})
    logf = n4_logfield(x, brain & (x > 0), protocol.voxel_mm)
    v = logf[brain]
    raw = round(float(np.exp(v.max() - v.min()) - 1), 6)
    present = bool(raw >= LINE)
    value = round(raw / SCALE, 6)
    sd = sides(logf, brain, protocol)
    rt = by_line(raw, LINE, 3)  # never printed on the wrong side of the line
    if present:
        why = (f"smooth shading over the brain, about {value:.2f} max/min − 1 (N4 reads {rt}, line {LINE}): "
               f"brightest at the {WORDS[sd['bright_side']]}, darkest at the {WORDS[sd['dark_side']]}")
    else:
        why = f"no shading beyond what a clean brain has: N4 reads {rt} (line {LINE})"
    ev = {"raw": raw, "line": LINE, "scale": SCALE, "brain_ml": ml} | sd
    f = Finding("bias", VERSION, "bias", present, value, UNIT, _confidence(raw), why, ev)
    if picture is not None:
        from bamr.tools.picture import bias_picture
        bias_picture(x, brain, logf, f"bias: {why}", Path(picture))
        f.picture = Path(picture).name
    return f
