"""Evidence picture shared by the tools: three slices through where a tool measured (D68)."""
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

GRAY = dict(cmap="gray", vmin=0, vmax=1.35, interpolation="nearest")
VIEWS = (  # (LR, AP, SI) volume → radiological axial, coronal, sagittal
    ("axial", 2, lambda v, k: v[:, :, k].T[::-1, ::-1]),
    ("coronal", 1, lambda v, k: v[:, k, :].T[::-1, ::-1]),
    ("sagittal", 0, lambda v, k: v[k].T[::-1, ::-1]),
)


def evidence_picture(x: np.ndarray, head: np.ndarray, flat: np.ndarray, title: str, path: Path) -> None:
    """Head outline (yellow) and measuring region (blue) on the slices with the most of that region."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig = Figure(figsize=(10.5, 3.9))  # no pyplot: the caller's matplotlib set-up is left alone (D99 #10)
    FigureCanvasAgg(fig)
    axs = fig.subplots(1, 3, gridspec_kw=dict(wspace=0.03))
    for ax, (name, axis, cut) in zip(axs, VIEWS):
        k = int(np.argmax(flat.sum(axis=tuple(a for a in range(3) if a != axis))))
        img = cut(x, k)
        ax.imshow(img, **GRAY)
        rgba = np.zeros(img.shape + (4,))
        rgba[cut(flat, k)] = [0.31, 0.82, 1.0, 0.5]
        ax.imshow(rgba, interpolation="nearest")
        if cut(head, k).any():
            ax.contour(cut(head, k).astype(float), [0.5], colors=["#f5c242"], linewidths=1.0)
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_title(name, fontsize=9)
    fig.suptitle(title, fontsize=10)
    fig.savefig(path, dpi=90, bbox_inches="tight")


def _figure(n: int):
    fig = Figure(figsize=(3.5 * n, 3.9))
    FigureCanvasAgg(fig)
    return fig, fig.subplots(1, n, gridspec_kw=dict(wspace=0.12))


def spike_picture(x: np.ndarray, K: np.ndarray, peaks: list, title: str, path: Path) -> None:
    """K-space plane through the top peak (circled), an axial slice, and the stripe the peaks make on that slice."""
    import numpy.fft as nfft
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig, axs = _figure(3)
    c = np.array(K.shape) // 2
    top = np.array(peaks[0]) if peaks else c
    ax = axs[0]
    ax.imshow(np.log10(np.abs(K[:, :, top[2]]) ** 2 + 1e-6).T, cmap="magma", origin="lower", interpolation="nearest")
    for p in peaks:
        if p[2] == top[2]:
            ax.add_patch(matplotlib_circle((p[0], p[1])))
    ax.set_title(f"k-space, plane k_SI = {top[2] - c[2]}" if peaks else "k-space (no peak)", fontsize=9)
    ax.set_xlabel("k_LR", fontsize=8), ax.set_ylabel("k_AP", fontsize=8)
    k = x.shape[2] // 2
    axs[1].imshow(x[:, :, k].T[::-1, ::-1], **GRAY)
    axs[1].set_title("axial slice", fontsize=9)
    S = np.zeros_like(K)
    for p in peaks:
        for j in (np.array(p), 2 * c - np.array(p)):
            sl = tuple(slice(v - 1, v + 2) for v in j)
            S[sl] = K[sl]
    stripe = np.real(nfft.ifftn(nfft.ifftshift(S)))[:, :, k].T[::-1, ::-1]
    lim = max(float(np.abs(stripe).max()), 1e-6)
    axs[2].imshow(stripe, cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
    axs[2].set_title(f"the stripe alone (±{lim:.3f} of WM)" if peaks else "no stripe", fontsize=9)
    for ax in axs[1:]:
        ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle(title, fontsize=10)
    fig.savefig(path, dpi=90, bbox_inches="tight")


def matplotlib_circle(xy):
    from matplotlib.patches import Circle
    return Circle(xy, 4, fill=False, color="#4fd1ff", linewidth=1.5)


def ghost_picture(x: np.ndarray, ap: int, rz: np.ndarray, f: np.ndarray, rel: np.ndarray, period, copy_mm,
                  title: str, path: Path) -> None:
    """The line energies' wiggle, its periodogram with the 150 line, and a slice along A–P with the copies marked."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig, axs = _figure(3)
    ky = np.arange(rz.size) - rz.size // 2
    axs[0].plot(ky, rz, lw=0.8, color="#333")
    axs[0].set_title("A–P line energy, fast wiggle (log)", fontsize=9)
    axs[0].set_xlabel("k_AP (line)", fontsize=8)
    band = f >= 0.1
    axs[1].semilogy(f[band], np.maximum(rel[band], 1e-2), lw=0.9, color="#333")
    axs[1].axhline(150, color="#d9822b", lw=1, ls="--")
    if period:
        axs[1].axvline(1 / period, color="#4fd1ff", lw=1.2)
    axs[1].set_title("periodogram ÷ typical bin (line 150)", fontsize=9)
    axs[1].set_xlabel("cycles per line", fontsize=8)
    sl = np.moveaxis(x, ap, 1)[x.shape[0] // 2] if ap != 0 else x[:, x.shape[1] // 2, :]
    axs[2].imshow(sl.T, cmap="gray", vmin=0, vmax=0.08, origin="lower", interpolation="nearest")
    if copy_mm:
        mid = sl.shape[0] / 2
        for s in (-1, 1):
            axs[2].axvline(mid + s * copy_mm, color="#4fd1ff", lw=1, ls=":")
    axs[2].set_title("A–P × S–I slice, window 0–8 % of WM", fontsize=9)
    axs[2].set_xticks([]), axs[2].set_yticks([])
    fig.suptitle(title, fontsize=10)
    fig.savefig(path, dpi=90, bbox_inches="tight")


def bias_picture(x: np.ndarray, brain: np.ndarray, logf: np.ndarray, title: str, path: Path) -> None:
    """The N4 field over the brain (blue darker, red brighter than the brain's mean) on the slices through its centre,
    with the brain outline (yellow)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig, axs = _figure(3)
    c = np.argwhere(brain).mean(axis=0).round().astype(int) if brain.any() else np.array(x.shape) // 2
    gain = np.exp(logf - logf[brain].mean()) - 1 if brain.any() else np.zeros_like(logf)
    lim = max(float(np.abs(gain[brain]).max()) if brain.any() else 0.0, 1e-3)
    for ax, (name, axis, cut) in zip(axs, VIEWS):
        k = int(c[axis])
        ax.imshow(cut(x, k), **GRAY)
        g = np.ma.masked_where(~cut(brain, k), cut(gain, k))
        im = ax.imshow(g, cmap="RdBu_r", vmin=-lim, vmax=lim, alpha=0.55, interpolation="nearest")
        if cut(brain, k).any():
            ax.contour(cut(brain, k).astype(float), [0.5], colors=["#f5c242"], linewidths=1.0)
        ax.set_xticks([]), ax.set_yticks([])
        ax.set_title(name, fontsize=9)
    cb = fig.colorbar(im, ax=list(axs), fraction=0.02, pad=0.01)
    cb.set_label("N4 field − 1 (vs brain mean)", fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.savefig(path, dpi=90, bbox_inches="tight")


def motion_picture(x: np.ndarray, curves: dict, protocol, line: float, title: str, path: Path) -> None:
    """The leak g(Δ) along each axis (the window its median is read over shaded, the line dashed), and two slices
    through the head's centre at 0–8 % of WM, where a smear along A–P shows in the air."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig, axs = _figure(3)
    colours = {"AP": "#d9822b", "LR": "#4f7fbf", "SI": "#5a9e6f"}
    for a, g in curves.items():
        d = np.arange(g.size)[: g.size // 2 + 1] * protocol.voxel_mm
        axs[0].plot(d, g[: g.size // 2 + 1], lw=1.2 if a == protocol.phase_slow else 0.8, color=colours[a],
                    label=f"{a[0]}–{a[1]}")
    axs[0].axvspan(20 * protocol.voxel_mm, 120 * protocol.voxel_mm, color="#999", alpha=0.12, lw=0)
    axs[0].axhline(line, color="#333", lw=0.9, ls="--")
    axs[0].axhline(0, color="#bbb", lw=0.6)
    seen = np.concatenate([g[15: g.size // 2 + 1] for g in curves.values()])  # shifts < 15 hold the head's own edge
    seen = seen[np.isfinite(seen)]
    if seen.size:
        axs[0].set_ylim(min(seen.min(), -line) * 1.3, max(seen.max(), 2 * line) * 1.3)
    axs[0].set_xlabel("head shifted by Δ (mm)", fontsize=8)
    axs[0].set_title("leak into the air, per axis (line dashed)", fontsize=9)
    axs[0].legend(fontsize=7, frameon=False)
    c = np.array(x.shape) // 2
    for ax, (name, axis, cut) in zip(axs[1:], (VIEWS[2], VIEWS[0])):
        ax.imshow(cut(x, int(c[axis])), cmap="gray", vmin=0, vmax=0.08, interpolation="nearest")
        ax.set_title(f"{name}, window 0–8 % of WM", fontsize=9)
        ax.set_xticks([]), ax.set_yticks([])
    fig.suptitle(title, fontsize=10)
    fig.savefig(path, dpi=90, bbox_inches="tight")
