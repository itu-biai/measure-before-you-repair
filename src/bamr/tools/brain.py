"""Shared region (D70 deferred it here, D101): the brain, outlined by HD-BET (isensee2019) on the input itself.

The Stage 1 model (D03, D22): HD-BET v2's nnU-Net, test-time mirroring off, loaded once per process, on the Mac's
GPU (MPS) when there is one. The array goes in as SimpleITK reads a NIfTI (axes reversed), the way Stage 1 fed it,
and the mask comes back in the input's axes. Only the array and the voxel size are given; nothing is read from disk
but the network's own weights.
"""
import contextlib
import sys

import numpy as np

from bamr.tools.finding import PROTOCOL_V2, Protocol

_PREDICTORS: dict[str, object] = {}


def device_name() -> str:
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _import():
    import HD_BET.hd_bet_prediction  # noqa: F401  (on import it sets sys.stdout to sys.__stdout__)


def _load(device: str):
    import torch
    from HD_BET.checkpoint_download import maybe_download_parameters
    from HD_BET.hd_bet_prediction import get_hdbet_predictor
    maybe_download_parameters()
    p = get_hdbet_predictor(use_tta=False, device=torch.device(device))
    torch.set_num_threads(2)  # after: on the CPU, HD-BET sets every core
    p.allow_tqdm = False  # no progress bars inside an agent loop
    return p


def predictor(device: str | None = None):
    """HD-BET's predictor, loaded on first use and kept for the process. HD-BET's import resets sys.stdout and
    nnU-Net prints: the import runs first, on its own, so its reset is undone before anything prints; both steps send
    stdout to stderr and put the caller's stdout back (review I1)."""
    device = device or device_name()
    if device not in _PREDICTORS:
        with contextlib.redirect_stdout(sys.stderr):
            _import()
        with contextlib.redirect_stdout(sys.stderr):
            _PREDICTORS[device] = _load(device)
    return _PREDICTORS[device]


def brain_mask(x: np.ndarray, protocol: Protocol = PROTOCOL_V2) -> np.ndarray:
    arr = np.ascontiguousarray(np.asarray(x, np.float32).transpose(2, 1, 0))[None]
    p = predictor()
    with contextlib.redirect_stdout(sys.stderr):
        seg = p.predict_single_npy_array(arr, {"spacing": [float(protocol.voxel_mm)] * 3}, None, None, False)
    return np.asarray(seg).transpose(2, 1, 0) > 0
