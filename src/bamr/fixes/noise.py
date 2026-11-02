"""The noise fix (D118 A, NF3): the grain thermal noise leaves, lowered by DRUNet with the σ the noise tool measured.

1. The noise tool (Task 2, D71) reads σ in the flat tissue. Under its 3.0 % line, or when it cannot measure, the volume
   comes back as given, with the tool's own reason.
2. At or above the line: DRUNet (zhang2022drunet) with its published weights on the slices of each of the three
   directions, σ map = the tool's σ, the three volumes averaged (bamr.fixes.drunet). One pass: unlike a spike, nothing
   hides behind noise.
3. Exact-zero voxels of the input stay 0 (D69).
4. The noise tool is asked again on the output, for the report.
Limits (before-doc §9): a 2-D network trained on photographs with Gaussian noise of known σ. In tissue the signal sits
several σ above zero, where Rician noise is close to Gaussian (gudbjartsson1995); in the air it is not. The scan's own
noise stays in the target, so no denoiser removes all of the error (before-doc §4).
"""
import numpy as np

from bamr.fixes.drunet import SHA256, default_device, denoise3
from bamr.fixes.report import FixReport, head_change, unchanged
from bamr.tools.finding import PROTOCOL_V2, Protocol, by_line
from bamr.tools.noise import LINE, noise_tool
from bamr.tools.regions import head_mask

TOOL, VERSION, ARTIFACT = "fix_noise", 1, "noise"
METHOD = "drunet_gray x3 (KAIR v1.0)"


def fix_noise(x: np.ndarray, protocol: Protocol = PROTOCOL_V2) -> tuple[np.ndarray, FixReport]:
    try:
        f = noise_tool(x, protocol)
        if f.present is not True:  # under the line, or the tool cannot measure: its own reason
            return unchanged(x, TOOL, VERSION, ARTIFACT, f.why,
                             {"sigma_before": f.value, "present_before": f.present, "line": LINE})
        return _fix(np.asarray(x).astype(np.float32, copy=False), f.value, f.why, protocol)
    except Exception as e:  # no crash inside an agent loop (D68): the error becomes the reason
        return unchanged(x, TOOL, VERSION, ARTIFACT, f"fix error: {type(e).__name__}: {e}")


def _fix(a: np.ndarray, s: float, why: str, protocol: Protocol) -> tuple[np.ndarray, FixReport]:
    dev = default_device()
    y = np.where(a == 0, 0, denoise3(a, s, device=dev)).astype(np.float32)
    g = noise_tool(y, protocol)
    # never printed on the wrong side of the line (D99 #9): 0.029996 reads "2.9996 %", not "3.00 %"
    after = f"{by_line(100 * g.value, 100 * LINE, 2)} %" if g.value is not None else f"cannot measure ({g.why})"
    ev = {"sigma_before": s, "sigma_after": g.value, "present_after": g.present, "line": LINE, "method": METHOD,
          "weights_sha256": SHA256, "device": dev}
    if g.present is None:
        ev["why_after"] = g.why
    what = f"denoised with σ {100 * s:.2f} % (DRUNet, three directions); noise tool after: {after}"
    return y, FixReport(TOOL, VERSION, ARTIFACT, True, what, why, head_change(y, a, head_mask(a)), ev)
