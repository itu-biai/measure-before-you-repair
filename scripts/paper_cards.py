"""Task 4 (D159): write the paper view (card.html) of every finished run in a results folder; reads result.json and the
pictures only (no label, no model call).

  uv run python scripts/paper_cards.py --out stage2-heldout [--folder agent-haiku]
"""
import argparse

from bamr.agent.paper_view import card
from bamr.config import REPO, load_config

RESULTS = load_config(REPO / "configs" / "dataset_v2.yaml")["paths"]["out"].parent / "task4-results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--folder", action="append", help="system folders (default: every folder holding result.json)")
    a = ap.parse_args()
    root = RESULTS / a.out
    folders = [root / f for f in a.folder] if a.folder else sorted({p.parent.parent for p in root.glob("*/*/result.json")})
    n = 0
    for f in folders:
        for d in sorted(p.parent for p in f.glob("*/result.json")):
            card(d)
            n += 1
    print(f"{n} cards → {root}")


if __name__ == "__main__":
    main()
