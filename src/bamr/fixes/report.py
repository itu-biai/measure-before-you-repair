"""The report every fix returns with its volume (D112): what was found and changed, measured, without verdict words.

The same rules as a Task 2 finding (D68, D95): strict JSON, floats to 6 decimals, numbers instead of judgements. The
Task 2 tool's own reason goes in `why`; `change` is how far the volume moved, RMS over the head, WM = 1.
"""
import json
from dataclasses import dataclass, field

import numpy as np

from bamr.tools.finding import _plain

UNIT = "RMS change in the head, WM = 1"


@dataclass
class FixReport:
    tool: str
    version: int
    artifact: str
    changed: bool
    what: str  # what was changed, measured; "nothing changed" when the volume comes back as given
    why: str  # the Task 2 tool's reason, read before the fix (or why the fix could not run)
    change: float
    evidence: dict = field(default_factory=dict)
    unit: str = UNIT

    def to_json(self) -> str:
        keys = ("tool", "version", "artifact", "changed", "what", "why", "change", "unit", "evidence")
        return json.dumps({k: _plain(getattr(self, k)) for k in keys}, ensure_ascii=False, allow_nan=False)


def head_change(y: np.ndarray, x: np.ndarray, head: np.ndarray) -> float:
    d = (np.asarray(y, np.float64) - np.asarray(x, np.float64))[head]
    return float(np.sqrt(np.mean(d ** 2))) if d.size else 0.0


def unchanged(x, tool: str, version: int, artifact: str, why: str,
              evidence: dict | None = None) -> tuple[np.ndarray, FixReport]:
    """The input itself (byte-identical: nothing copied, nothing cast) and a report that says why."""
    return x, FixReport(tool, version, artifact, False, "nothing changed", why, 0.0, evidence or {})
