"""Task 2 Stage 3: the shared brain region (D101): HD-BET on the input, mask back in the input's axes."""
from pathlib import Path

import numpy as np
import pytest

import bamr.tools.brain as B
from bamr.config import REPO
from bamr.tools.finding import PROTOCOL_V2


class StandIn:
    """Records what HD-BET would be given; 'segments' every voxel above 0.5 (nnU-Net's array order is z, y, x)."""
    def __init__(self):
        self.calls = []

    def predict_single_npy_array(self, arr, props, prev, out, probs):
        self.calls.append((arr.shape, arr.dtype, props))
        return (arr[0] > 0.5).astype(np.uint8)


def test_brain_axes_with_stand_in(monkeypatch):
    """A transposition bug would still look like a brain: a non-cubic box and an asymmetric pattern catch it."""
    fake = StandIn()
    monkeypatch.setattr(B, "predictor", lambda device=None: fake)
    x = np.random.default_rng(0).uniform(0, 1, (20, 30, 40)).astype(np.float32)
    m = B.brain_mask(x, PROTOCOL_V2)
    assert m.dtype == bool and m.shape == x.shape and np.array_equal(m, x > 0.5)
    shape, dtype, props = fake.calls[0]
    assert shape == (1, 40, 30, 20) and dtype == np.float32 and list(props["spacing"]) == [1.0, 1.0, 1.0]


def test_brain_leaves_the_callers_stdout_alone(monkeypatch, capfd):
    """Review I1: HD-BET's import sets sys.stdout to sys.__stdout__, then nnU-Net prints (on MPS, and while
    predicting): an agent host that captures stdout or speaks over stdio must keep its stdout, with nothing written to
    file descriptor 1."""
    import sys

    class Loud(StandIn):
        def predict_single_npy_array(self, *a):
            print("out of memory, retrying")
            return super().predict_single_npy_array(*a)

    def import_hdbet():
        sys.stdout = sys.__stdout__  # what hd_bet_prediction does on import

    def load(device):
        print("perform_everything_on_device=True is only supported for cuda devices!")
        return Loud()

    monkeypatch.setattr(B, "_PREDICTORS", {})
    monkeypatch.setattr(B, "_import", import_hdbet)
    monkeypatch.setattr(B, "_load", load)
    before = sys.stdout
    B.brain_mask(np.ones((8, 8, 8), np.float32), PROTOCOL_V2)
    assert sys.stdout is before and capfd.readouterr().out == ""


def test_brain_spacing_follows_protocol(monkeypatch):
    from dataclasses import replace
    fake = StandIn()
    monkeypatch.setattr(B, "predictor", lambda device=None: fake)
    B.brain_mask(np.ones((8, 8, 8), np.float32), replace(PROTOCOL_V2, voxel_mm=2.0))
    assert list(fake.calls[0][2]["spacing"]) == [2.0, 2.0, 2.0]


DS = REPO.parent / "dataset" / "blind-mri-v2"


def _weights() -> bool:
    try:
        from HD_BET.paths import folder_with_parameter_files
        return (Path(folder_with_parameter_files) / "fold_all").exists()
    except Exception:
        return False


@pytest.mark.skipif(not (_weights() and (DS / "inputs" / "s0003.nii.gz").exists()),
                    reason="HD-BET weights or blind-mri-v2 not on this machine")
def test_brain_real_dev_volume(monkeypatch, capfd):
    """HD-BET on a corrupted input (s0003, strong bias) against Stage 1's mask of the clean scan (analysis only).
    Loaded fresh here, it must leave the caller's stdout as it was and write nothing to file descriptor 1 (review I1:
    its import resets sys.stdout, then nnU-Net prints on MPS)."""
    import sys
    import nibabel as nib
    from bamr.data.blindio import read_volume
    x = read_volume(DS / "inputs" / "s0003.nii.gz")
    ref = np.asarray(nib.load(DS / "refs" / "s0003" / "brain_mask.nii.gz").dataobj) > 0
    monkeypatch.setattr(B, "_PREDICTORS", {})
    before = sys.stdout
    m = B.brain_mask(x)
    assert sys.stdout is before and capfd.readouterr().out == ""
    assert m.shape == x.shape
    assert 2 * (m & ref).sum() / (m.sum() + ref.sum()) >= 0.95


@pytest.mark.skipif(not _weights(), reason="HD-BET weights not downloaded")
def test_brain_fresh_process_writes_nothing_to_stdout():
    """Review I1, in a fresh process (inside pytest HD-BET may already be imported): its import resets sys.stdout and
    nnU-Net then prints on MPS; the caller's stdout must stay as it was, with nothing written to it."""
    import subprocess
    import sys
    code = ("import sys, numpy as np, bamr.tools.brain as B; s = sys.stdout; "
            "B.brain_mask(np.ones((64, 64, 64), np.float32)); assert sys.stdout is s")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-500:]
    assert r.stdout == ""
