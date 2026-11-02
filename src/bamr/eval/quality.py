"""Image quality of one volume against the clean reference (D158): the metrics restoration papers report.

- PSNR and SSIM inside the reference's brain, with the definitions the dataset was calibrated with (D42,
  bamr.data.metrics): data range = the reference's brain maximum; SSIM's 3-D map averaged over the brain.
- LPIPS (Zhang et al. 2018; the authors' package, AlexNet, v0.1): a pretrained network's distance, 0 = same. It is a
  2-D metric, so it runs on axial slices (array axes LR, AP, SI: a fixed SI index); each slice is scaled by the same
  brain maximum to [0, 1], clipped, mapped to [−1, 1] and copied to three channels; the per-pixel map (spatial=True)
  is averaged over the brain, as SSIM's is. Slices without brain are skipped.
Scorer only (refs are read there, outside the guard); nothing here reaches the agent.
"""
import numpy as np

from bamr.data.metrics import masked_psnr, masked_ssim

AXIAL = 2  # array axes (LR, AP, SI)


class Lpips:
    def __init__(self, batch: int = 16):
        import warnings

        import lpips
        import torch
        self.torch, self.batch = torch, batch
        with warnings.catch_warnings():  # lpips 0.1.4 asks torchvision for pretrained=True: AlexNet IMAGENET1K_V1
            warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
            self.net = lpips.LPIPS(net="alex", version="0.1", spatial=True, verbose=False).eval()

    def _t(self, x: np.ndarray, peak: float):
        v = np.clip(np.moveaxis(x, AXIAL, 0) / peak, 0, 1) * 2 - 1  # slices first
        return self.torch.from_numpy(np.ascontiguousarray(v, np.float32))[:, None].expand(-1, 3, -1, -1)

    def __call__(self, ref: np.ndarray, out: np.ndarray, brain: np.ndarray) -> float:
        peak = float(ref[brain].max())
        ks = np.flatnonzero(brain.any(axis=tuple(i for i in range(3) if i != AXIAL)))
        total, n = 0.0, 0
        for i in range(0, len(ks), self.batch):
            k = ks[i:i + self.batch]
            m = self.torch.from_numpy(np.ascontiguousarray(np.moveaxis(np.take(brain, k, AXIAL), AXIAL, 0)))
            with self.torch.no_grad():
                d = self.net(self._t(np.take(ref, k, AXIAL), peak), self._t(np.take(out, k, AXIAL), peak))[:, 0]
            total += float(d[m].sum())
            n += int(m.sum())
        return total / n


def quality(ref: np.ndarray, out: np.ndarray, brain: np.ndarray, lp: Lpips) -> dict:
    out = np.asarray(out, np.float32)
    return {"psnr": masked_psnr(ref, out, brain), "ssim": masked_ssim(ref, out, brain), "lpips": lp(ref, out, brain)}
