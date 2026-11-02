from collections import Counter
from pathlib import Path

import pytest

from bamr.config import load_config
from bamr.data.subjects import Subject, list_subjects, priority_order

QUOTA = {"Guys": 33, "HH": 19, "IOP": 8}


def _fake(n_guys=322, n_hh=185, n_iop=74):
    out, i = [], 0
    for site, n in (("Guys", n_guys), ("HH", n_hh), ("IOP", n_iop)):
        for _ in range(n):
            out.append(Subject(f"IXI{i:03d}", site, Path(f"IXI{i:03d}-{site}-0000-T1.nii.gz")))
            i += 1
    return out


def test_parse_site(tmp_path):
    (tmp_path / "IXI002-Guys-0828-T1.nii.gz").touch()
    (tmp_path / "IXI012-HH-1211-T1.nii.gz").touch()
    (tmp_path / "notes.txt").touch()
    subs = list_subjects(tmp_path)
    assert [(s.id, s.site) for s in subs] == [("IXI002", "Guys"), ("IXI012", "HH")]
    assert subs[0].path == tmp_path / "IXI002-Guys-0828-T1.nii.gz"


def test_priority_quota():
    order = priority_order(_fake(), seed=7, quota=QUOTA)
    assert Counter(s.site for s in order[:60]) == Counter(QUOTA)


def test_priority_deterministic():
    subs = _fake()
    a, b = priority_order(subs, 7, QUOTA), priority_order(list(reversed(subs)), 7, QUOTA)
    c = priority_order(subs, 8, QUOTA)
    assert a == b
    assert a != c
    assert sorted(s.id for s in a) == sorted(s.id for s in subs)


def test_real_dataset():
    cfg = load_config()
    if not cfg["paths"]["raw"].exists():
        pytest.skip("raw IXI not present")
    subs = list_subjects(cfg["paths"]["raw"])
    assert len(subs) == 581
    assert len({s.id for s in subs}) == 581
    assert {s.site for s in subs} <= {"Guys", "HH", "IOP"}
