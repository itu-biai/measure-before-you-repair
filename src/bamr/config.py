"""Load configs/*.yaml; `paths` entries become absolute Paths relative to the repo root."""
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]


def load_config(path: Path | str | None = None) -> dict:
    path = Path(path) if path else REPO / "configs" / "dataset_v1.yaml"
    cfg = yaml.safe_load(path.read_text())
    cfg["paths"] = {k: (REPO / v).resolve() for k, v in cfg["paths"].items()}
    return cfg
