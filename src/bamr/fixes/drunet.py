"""DRUNet (zhang2022drunet): KAIR's grayscale denoiser with its published v1.0 weights, run on a 3-D volume (D118 A).

The network is KAIR's UNetRes (models/network_unet.py: 64·128·256·512 channels, 4 residual blocks per scale, strided
convolutions down, transposed convolutions up, no biases), written out here. KAIR: MIT licence, "Copyright (c) 2019
Kai Zhang" (github.com/cszn/KAIR/blob/master/LICENSE). Input: a slice in the network's [0, 1] range and a constant σ
map; output: the denoised slice. NF3 (before-doc §9): the slices of each of the three directions, the three volumes
averaged; intensities ÷ 1.5 into the network's range (WM = 1) and back. A side that is not a multiple of 8 is padded by
its edge values and cropped after (ours; the dataset's 192 × 240 × 240 needs none).
The weights (D120): fetched once by scripts/get_weights.py. Here they are only checked, by sha256 before torch reads the
file, then loaded with weights_only and every key and shape matched. Nothing is downloaded here. MPS when present,
else CPU.
"""
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn

WEIGHTS = Path.home() / ".cache" / "bamr" / "drunet_gray.pth"
URL = "https://github.com/cszn/KAIR/releases/download/v1.0/drunet_gray.pth"
SHA256 = "e27fb29456732c604c8ee3ac3e92ecadd2a7a4e36a8be675e92ebbf93a240de4"
SCALE = 1.5  # WM = 1 → the network's [0, 1] (as the pilot, NF3)
BATCH = 16  # slices per pass (as the pilot)
MULT = 8  # three 2× steps down
_NETS = {}


def _conv(i, o, k=3, s=1, p=1):
    return nn.Conv2d(i, o, k, s, p, bias=False)


class _Res(nn.Module):  # KAIR's ResBlock, mode 'CRC'
    def __init__(self, c):
        super().__init__()
        self.res = nn.Sequential(_conv(c, c), nn.ReLU(inplace=True), _conv(c, c))

    def forward(self, x):
        return x + self.res(x)


class UNetRes(nn.Module):
    """KAIR models/network_unet.py UNetRes(in_nc=2, out_nc=1, nc=[64, 128, 256, 512], nb=4, act_mode='R',
    downsample_mode='strideconv', upsample_mode='convtranspose'); the module names match the published weights."""

    def __init__(self, nc=(64, 128, 256, 512), nb=4):
        super().__init__()
        self.m_head = _conv(2, nc[0])
        for i in range(3):
            setattr(self, f"m_down{i + 1}", nn.Sequential(*[_Res(nc[i]) for _ in range(nb)],
                                                          _conv(nc[i], nc[i + 1], 2, 2, 0)))
        self.m_body = nn.Sequential(*[_Res(nc[3]) for _ in range(nb)])
        for i in (3, 2, 1):
            setattr(self, f"m_up{i}", nn.Sequential(nn.ConvTranspose2d(nc[i], nc[i - 1], 2, 2, 0, bias=False),
                                                    *[_Res(nc[i - 1]) for _ in range(nb)]))
        self.m_tail = _conv(nc[0], 1)

    def forward(self, x0):
        x1 = self.m_head(x0)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        return self.m_tail(x + x1)


def sha256(p: Path) -> str:
    with open(p, "rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def default_device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load(path: Path | None = None, device: str | None = None) -> tuple[UNetRes, str]:
    """The network with the checked weights, once per process; a missing or different file is a clear error."""
    p, dev = Path(path or WEIGHTS), device or default_device()
    key = (str(p), dev)
    if key not in _NETS:
        if not p.exists():
            raise FileNotFoundError(f"DRUNet weights not found at {p}: run scripts/get_weights.py once (the fix "
                                    "never downloads)")
        got = sha256(p)
        if got != SHA256:
            raise ValueError(f"{p} has sha256 {got[:8]}…{got[-6:]}, not KAIR v1.0 drunet_gray.pth's "
                             f"{SHA256[:8]}…{SHA256[-6:]}: run scripts/get_weights.py")
        m = UNetRes()
        m.load_state_dict(torch.load(p, map_location="cpu", weights_only=True), strict=True)
        _NETS[key] = m.eval().to(dev)
    return _NETS[key], dev


def _along(m, dev, x: np.ndarray, sigma: float, ax: int, batch: int) -> np.ndarray:
    a = np.moveaxis(x, ax, 0)[:, None] / SCALE
    h, w = a.shape[2:]
    pad = ((0, 0), (0, 0), (0, -h % MULT), (0, -w % MULT))
    if pad[2][1] or pad[3][1]:
        a = np.pad(a, pad, mode="edge")
    out = np.empty_like(a)
    with torch.no_grad():
        for i in range(0, len(a), batch):
            t = torch.from_numpy(np.ascontiguousarray(a[i:i + batch], np.float32))
            t = torch.cat([t, torch.full_like(t, sigma / SCALE)], 1).to(dev)
            out[i:i + batch] = m(t).cpu().numpy()
    return np.moveaxis(out[:, 0, :h, :w] * SCALE, 0, ax)


def denoise3(x: np.ndarray, sigma: float, path: Path | None = None, device: str | None = None,
             batch: int = BATCH) -> np.ndarray:
    """DRUNet on the slices of each direction with the σ map = sigma (WM = 1), the three volumes averaged."""
    m, dev = load(path, device)
    x = np.asarray(x, np.float32)
    return np.mean([_along(m, dev, x, sigma, ax, batch) for ax in (0, 1, 2)], axis=0).astype(np.float32)
