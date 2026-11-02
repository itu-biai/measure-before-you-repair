"""Task 2 Stage 1: tool contract (D68) and shared regions (D70)."""
import json

import numpy as np

from bamr.tools.finding import PROTOCOL_V2, Finding
from bamr.tools.regions import flat_tissue, head_mask

BANDS = [(0.030, 0.038), (0.038, 0.048), (0.048, 0.061)]


def ellipsoid(shape=(96, 120, 120), frac=0.42):
    g = np.ogrid[tuple(slice(0, d) for d in shape)]
    r = np.sqrt(sum(((gi - d / 2) / (d * frac)) ** 2 for gi, d in zip(g, shape)))
    return r < 1


def test_protocol_v2():
    p = PROTOCOL_V2
    assert (p.readout, p.phase_slow, p.phase_fast) == ("SI", "AP", "LR")
    assert p.array_axes == ("LR", "AP", "SI") and p.array_axes.index("SI") == 2
    assert (p.voxel_mm, p.wm) == (1.0, 1.0)
    assert p.orientation == "RAS"  # Stage 3 R1: the end each array axis grows toward (the v2 inputs are RAS)


def test_check_volume_orientation():
    """R1: an orientation whose letters are not on the named axes is a reason, not a silent mirror."""
    from dataclasses import replace
    from bamr.tools.finding import check_volume
    x = np.zeros((8, 8, 8), np.float32)
    assert check_volume(x, PROTOCOL_V2) is None
    assert check_volume(x, replace(PROTOCOL_V2, orientation="LPI")) is None
    assert "orientation" in check_volume(x, replace(PROTOCOL_V2, orientation="RSA"))
    assert "orientation" in check_volume(x, replace(PROTOCOL_V2, orientation="RA"))


def test_finding_json_order_and_rounding():
    f = Finding(tool="noise", version=1, artifact="noise", present=True, value=0.0431234567,
                unit="sigma, fraction of WM", confidence="high", why="w", evidence={"raw_sigma": 0.0431234567},
                picture=None)
    d = json.loads(f.to_json())
    assert list(d) == ["tool", "version", "artifact", "present", "value", "unit", "confidence", "why", "evidence",
                       "picture"]
    assert d["value"] == 0.043123 and d["evidence"]["raw_sigma"] == 0.043123


def test_tools_never_name_levels():
    """D95: the agent sees value + yes/no; our simulation's level names live in the scorer only."""
    import re
    from pathlib import Path
    for p in (Path(__file__).resolve().parents[1] / "src" / "bamr" / "tools").glob("*.py"):
        assert not re.search(r"\b(mild|moderate|borderline|level)\b", p.read_text()), p.name


def test_head_mask_phantom():
    obj = ellipsoid()
    x = (obj + np.random.default_rng(0).normal(0, 0.02, obj.shape)).astype(np.float32)
    h = head_mask(x)
    from scipy import ndimage as ndi
    assert h[obj].mean() >= 0.99
    assert ndi.distance_transform_edt(~obj)[h].max() <= 3  # smoothing + half Otsu grows the outline by ~1 voxel only


def test_head_mask_empty():
    h = head_mask(np.zeros((40, 50, 60), np.float32))
    assert h.dtype == bool and not h.any()


def test_flat_tissue_inset_and_band():
    obj = ellipsoid()
    x = obj.astype(np.float32)
    x[40:44, 50:54, 50:54] = 0.80
    x[50:54, 60:64, 60:64] = 1.20
    head = head_mask(x)
    flat = flat_tissue(x, head)
    from scipy import ndimage as ndi
    assert (ndi.distance_transform_edt(head)[flat] >= 8).all()
    assert not flat[40:44, 50:54, 50:54].any() and not flat[50:54, 60:64, 60:64].any()
    assert flat.sum() > 10000


def test_finding_json_strict():
    import json
    import math
    from bamr.tools.finding import Finding
    f = Finding("t", 1, "t", None, math.nan, "u", "low", "w", {"a": math.inf, "b": [np.float32("nan"), 1.0]})
    d = json.loads(f.to_json(), parse_constant=lambda c: (_ for _ in ()).throw(ValueError(c)))
    assert d["value"] is None and d["evidence"] == {"a": None, "b": [None, 1.0]}


# ---------------------------------------------------------------- Stage 1 leftovers (D99)
def test_flat_tissue_inset_at_array_edge():
    """#7: where the head runs off the array, the 8-voxel inset is kept from the array's face too."""
    g = np.ogrid[:96, :120, :120]
    obj = np.sqrt(((g[0] - 10) / 40) ** 2 + ((g[1] - 60) / 50) ** 2 + ((g[2] - 60) / 50) ** 2) < 1
    x = obj.astype(np.float32)
    flat = flat_tissue(x, head_mask(x))
    assert flat.sum() > 1000 and not flat[:7].any()  # the face is outside: index 7 is 8 voxels in
