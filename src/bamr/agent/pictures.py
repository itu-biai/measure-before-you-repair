"""What the model is shown besides the tools' own pictures (before-doc §5): look, compare, and the smoke test's digit.

look: three planes through the middle of the head (its centre of mass), in the tools' grey scale 0–1.35 × WM
(bamr.tools.picture.GRAY, VIEWS), or one plane at a chosen slice; window "air" shows 0–0.1 × WM instead, the faint
signal outside the head where noise, ghost copies and motion show (Stage 2 r03; mortamet2009 reads artifacts there). compare: a − b on the same planes (red: a brighter,
blue: b brighter), colour limit the 99.5th percentile of |a − b| in b's head, with the RMS change in the head (WM = 1).
The layout is the one shown in the before-doc (_src/T4S1_figs.py). No pyplot: the caller's matplotlib is left alone.
"""
import base64
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from bamr.fixes.report import head_change
from bamr.tools.picture import GRAY, VIEWS
from bamr.tools.regions import head_mask

AXIS = {name: axis for name, axis, _ in VIEWS}
WINDOWS = {"head": GRAY, "air": dict(GRAY, vmin=0, vmax=0.1)}


def middle(head: np.ndarray, shape) -> tuple[int, ...]:
    """The head's centre of mass, one index per array axis; the box centre when there is no head."""
    if not head.any():
        return tuple(s // 2 for s in shape)
    return tuple(int(round(c)) for c in np.argwhere(head).mean(axis=0))


def _cuts(x: np.ndarray, plane, slice) -> list[tuple[str, int]]:
    c = middle(head_mask(x), x.shape)
    if plane is None:
        return [(name, c[axis]) for name, axis, _ in VIEWS]
    if plane not in AXIS:
        raise ValueError(f"plane must be one of {', '.join(AXIS)}")
    n = x.shape[AXIS[plane]]
    k = c[AXIS[plane]] if slice is None else int(slice)
    if not 0 <= k < n:
        raise ValueError(f"slice {k} is outside the {plane} slices 0–{n - 1}")
    return [(plane, k)]


def _save(fig: Figure, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=90, bbox_inches="tight")


def look(x: np.ndarray, path: Path, plane: str | None = None, slice: int | None = None,
         window: str | None = None) -> dict:
    window = window or "head"
    if window not in WINDOWS:
        raise ValueError(f"window must be one of {', '.join(WINDOWS)}")
    cuts = _cuts(np.asarray(x), plane, slice)
    fig = Figure(figsize=(3.5 * len(cuts), 3.9))
    FigureCanvasAgg(fig)
    axs = np.atleast_1d(fig.subplots(1, len(cuts), gridspec_kw=dict(wspace=0.03)))
    for ax, (name, k) in zip(axs, cuts):
        ax.imshow(dict((n, f) for n, _, f in VIEWS)[name](x, k), **WINDOWS[window])
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_title(f"{name} {k}" + (" · air window 0–0.1" if window == "air" else ""), fontsize=9)
    _save(fig, path)
    return {"planes": {name: k for name, k in cuts}, "window": window}


def compare(a: np.ndarray, b: np.ndarray, path: Path) -> dict:
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        raise ValueError(f"the versions differ in shape: {a.shape} and {b.shape}")
    head = head_mask(b)
    d = a.astype(np.float64) - b.astype(np.float64)
    lim = float(np.percentile(np.abs(d[head]), 99.5)) if head.any() else 0.0
    lim = lim or 1e-3
    rms = head_change(a, b, head)
    fig = Figure(figsize=(10.5, 3.9))
    FigureCanvasAgg(fig)
    for ax, (name, k) in zip(fig.subplots(1, 3, gridspec_kw=dict(wspace=0.03)), _cuts(b, None, None)):
        ax.imshow(dict((n, f) for n, _, f in VIEWS)[name](d, k), cmap="RdBu_r", vmin=-lim, vmax=lim,
                  interpolation="nearest")
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_title(f"{name} {k}", fontsize=9)
    fig.suptitle(f"a − b · RMS change in the head {rms:.4f} (WM = 1) · colour ±{lim:.3f}", fontsize=10)
    _save(fig, path)
    return {"rms_change": rms, "colour_limit": lim}


def digit(d: int, path: Path) -> None:
    """The smoke test's picture (D135): a large digit on a grey montage of the same size as a look."""
    fig = Figure(figsize=(10.5, 3.9))
    FigureCanvasAgg(fig)
    rng = np.random.default_rng(d)
    for ax in fig.subplots(1, 3, gridspec_kw=dict(wspace=0.03)):
        ax.imshow(0.3 + 0.05 * rng.standard_normal((64, 64)), **GRAY)
        ax.set_xticks([]), ax.set_yticks([])
    fig.text(0.5, 0.45, str(d), fontsize=150, color="white", ha="center", va="center", weight="bold")
    _save(fig, path)


def data_url(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(Path(path).read_bytes()).decode("ascii")
