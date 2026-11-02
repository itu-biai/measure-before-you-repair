"""Brain-masked image quality for Stage 2 calibration (D42). Data range = the reference's brain maximum."""
import numpy as np
from skimage.metrics import structural_similarity


def masked_psnr(ref: np.ndarray, out: np.ndarray, brain: np.ndarray) -> float:
    mse = float(np.mean((np.asarray(out, np.float64) - ref)[brain] ** 2))
    if mse == 0:
        return float("inf")
    return float(10 * np.log10(float(ref[brain].max()) ** 2 / mse))


def masked_ssim(ref: np.ndarray, out: np.ndarray, brain: np.ndarray) -> float:
    _, smap = structural_similarity(np.asarray(ref, np.float64), np.asarray(out, np.float64),
                                    data_range=float(ref[brain].max()), full=True)
    return float(smap[brain].mean())
