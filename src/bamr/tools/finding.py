"""The tool contract (D68): what every tool gets besides the array, and the one finding it returns."""
import json
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Protocol:
    """Scanner settings, like DICOM header fields: not artifact information, so the setup stays blind (D68, D75)."""
    readout: str
    phase_slow: str
    phase_fast: str
    array_axes: tuple[str, str, str]  # anatomical direction of array axes 0, 1, 2
    voxel_mm: float
    wm: float  # inputs are scaled so white matter = 1 (D56)
    orientation: str = "RAS"  # the end each array axis grows toward, NIfTI axis codes (Stage 3 R1: names the sides)


PROTOCOL_V2 = Protocol(readout="SI", phase_slow="AP", phase_fast="LR", array_axes=("LR", "AP", "SI"), voxel_mm=1.0,
                       wm=1.0)


def _plain(v):
    """JSON-ready copy: numpy scalars → Python, floats rounded to 6 decimals (byte-identical output, D68); NaN and
    infinity → None, so the output is always strict JSON."""
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return round(float(v), 6) if np.isfinite(v) else None
    return v


@dataclass
class Finding:
    tool: str
    version: int
    artifact: str
    present: bool | None  # None: the tool could not measure (graceful, D68)
    value: float | None  # what the agent acts on; the dataset's band names stay in the scorer (D95)
    unit: str
    confidence: str  # high · low
    why: str
    evidence: dict = field(default_factory=dict)
    picture: str | None = None

    def to_json(self) -> str:
        keys = ("tool", "version", "artifact", "present", "value", "unit", "confidence", "why", "evidence", "picture")
        return json.dumps({k: _plain(getattr(self, k)) for k in keys}, ensure_ascii=False, allow_nan=False)


def check_volume(x: np.ndarray, protocol: Protocol) -> str | None:
    """Why this input cannot be read, or None: a reason instead of a crash inside an agent loop (D68)."""
    if x.ndim != 3 or x.dtype.kind not in "biuf" or not np.isfinite(x).all():
        return "input is not a finite 3-D array of real numbers" + (" (complex: pass the magnitude)"
                                                                     if x.dtype.kind == "c" else "")
    if sorted(protocol.array_axes) != ["AP", "LR", "SI"]:
        return f"protocol array_axes {protocol.array_axes} is not an order of LR, AP, SI"
    o = protocol.orientation
    if len(o) != 3 or any(c not in a for c, a in zip(o, protocol.array_axes)):
        return f"protocol orientation {o!r} does not name one end of each axis {protocol.array_axes}"
    return None


def by_line(v: float, line: float, digits: int) -> str:
    """v with `digits` decimals, or more where fewer would print it on the other side of its line (a reason must
    never read "z 7.0 (line 7.0)" for a no): the same rule as deciding on the value as reported (D99 #9)."""
    d = digits
    while d < 6 and (float(f"{v:.{d}f}") >= line) != (v >= line):
        d += 1
    return f"{v:.{d}f}"
