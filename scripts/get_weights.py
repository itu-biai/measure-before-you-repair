"""Fetch the pretrained weights the fixes use, once, and check them (D120); the fixes themselves never download.

  uv run python scripts/get_weights.py   # → ~/.cache/bamr/drunet_gray.pth (KAIR v1.0 release, sha256 checked)

A file already there with the right sha256 is kept as it is. A download with a different sha256 is deleted, nothing is
kept, and the script stops with an error.
"""
import urllib.request
from pathlib import Path

from bamr.fixes.drunet import SHA256, URL, WEIGHTS, sha256


def get(dest: Path = WEIGHTS, fetch=urllib.request.urlretrieve) -> Path:
    if dest.exists() and sha256(dest) == SHA256:
        print(f"{dest}: already there, sha256 checked")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    try:
        fetch(URL, part)
        got = sha256(part)
        if got != SHA256:
            raise ValueError(f"{URL} gave sha256 {got}, expected {SHA256}: nothing kept")
        part.replace(dest)
    finally:
        part.unlink(missing_ok=True)
    print(f"{dest}: fetched from {URL}, sha256 checked")
    return dest


if __name__ == "__main__":
    get()
