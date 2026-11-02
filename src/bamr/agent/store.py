"""Versions of one volume as handles (D139, before-doc §6): v0 as received, then one per repair that changed something.

Each version is a read-only float32 .npy file with its parent, the repair that made it and a content hash (sha256 of
the shape and the bytes), which keys the result cache. An old version never changes, so the model may go on from any
of them (AgenticIR's roll-back, zhu2025agenticir). The model sees only the handles.
"""
import hashlib
import json
import threading
from pathlib import Path

import numpy as np


class VersionStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._meta: dict[str, dict] = {}
        self._lock = threading.Lock()

    @staticmethod
    def content_hash(x: np.ndarray) -> str:
        a = np.ascontiguousarray(x, np.float32)
        h = hashlib.sha256(json.dumps(list(a.shape)).encode())
        h.update(a.tobytes())
        return h.hexdigest()

    def add(self, x: np.ndarray, parent: str | None, repair: str | None) -> str:
        a = np.ascontiguousarray(x, np.float32)
        with self._lock:
            h = f"v{len(self._meta)}"
            p = self.root / f"{h}.npy"
            np.save(p, a)
            p.chmod(0o444)
            self._meta[h] = {"handle": h, "parent": parent, "repair": repair, "hash": self.content_hash(a)}
        return h

    def _m(self, h: str) -> dict:
        if h not in self._meta:
            raise KeyError(f"unknown version {h!r}")
        return self._meta[h]

    def get(self, h: str) -> np.ndarray:
        self._m(h)
        a = np.load(self.path(h))
        a.setflags(write=False)
        return a

    def path(self, h: str) -> Path:
        self._m(h)
        return self.root / f"{h}.npy"

    def hash(self, h: str) -> str:
        return self._m(h)["hash"]

    def handles(self) -> list[str]:
        return list(self._meta)

    def lineage(self, h: str) -> list[dict]:
        out = []
        while h is not None:
            m = self._m(h)
            out.append(dict(m))
            h = m["parent"]
        return out[::-1]
