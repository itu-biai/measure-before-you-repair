"""IXI subject listing and the fixed-seed, site-stratified processing order (spec Stage 1, D04)."""
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_NAME = re.compile(r"^(IXI\d{3})-(Guys|HH|IOP)-\d{4}-T1\.nii\.gz$")


@dataclass(frozen=True)
class Subject:
    id: str
    site: str
    path: Path


def list_subjects(raw_dir: Path) -> list[Subject]:
    out = []
    for p in sorted(Path(raw_dir).iterdir()):
        m = _NAME.match(p.name)
        if m:
            out.append(Subject(m.group(1), m.group(2), p))
    return out


def priority_order(subjects: list[Subject], seed: int, quota: dict[str, int]) -> list[Subject]:
    """Priority block (`quota` subjects per site, shuffled) followed by all others (shuffled)."""
    rng = np.random.default_rng(seed)
    head, tail = [], []
    for site in sorted({s.site for s in subjects}):
        group = sorted((s for s in subjects if s.site == site), key=lambda s: s.id)
        group = [group[i] for i in rng.permutation(len(group))]
        n = quota.get(site, 0)
        head += group[:n]
        tail += group[n:]
    return [head[i] for i in rng.permutation(len(head))] + [tail[i] for i in rng.permutation(len(tail))]
