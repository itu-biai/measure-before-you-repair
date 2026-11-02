"""The compute queue and its cache (D139, before-doc §2): every tool, fix and picture the model, the chain or the router
asks for runs here, once per (function, arguments, volume hash).

Workers are spawned processes (the Task 2 tools and Task 3 fixes, unchanged; HD-BET and DRUNet load once per worker),
each inside blind_guard for its whole life, so refs, labels and the manifest cannot be opened even by code built at
run time (D99 #10). workers=0 runs inline in the calling thread (tests). A result is a folder
cache/<key[:2]>/<key>/ with result.json (written last, by renaming the finished folder into place), the picture and,
for a fix that changed the volume, output.npy. A second caller of the same key waits for the first. Errors are
reported to the caller and not cached, so a passing failure does not stick.
"""
import dataclasses
import hashlib
import importlib
import json
import shutil
import threading
import time
import uuid
from collections import Counter
from concurrent.futures import Future, ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from multiprocessing import get_context
from pathlib import Path

import numpy as np

TOOLS = ("noise", "spike", "ghosting", "bias", "motion")
FIXES = ("spike", "ghosting", "noise", "bias")
TABLE = {**{f"{a}_tool": ("tool", f"bamr.tools.{a}:{a}_tool") for a in TOOLS},
         **{f"fix_{a}": ("fix", f"bamr.fixes.{a}:fix_{a}") for a in FIXES},
         "look": ("look", "bamr.agent.pictures:look"), "compare": ("compare", "bamr.agent.pictures:compare"),
         **{f"{a}_u": ("fixp", f"bamr.agent.ungated:{a}_u") for a in FIXES}}  # ②u and ⑤ (D151, D152)


@dataclasses.dataclass
class Outcome:
    data: dict  # the Finding / FixReport JSON, the picture's numbers, or {"error": ...}
    picture: Path | None
    output: Path | None  # a fix's new volume (.npy), only when it changed the input
    seconds: float
    cached: bool


_GUARD = None


def _init(blocked: tuple[str, ...]) -> None:
    """Enters the guard for the worker's whole life. The context is kept in a global: once collected, its generator
    would close and lift the block."""
    global _GUARD
    from bamr.eval.guard import blind_guard
    _GUARD = blind_guard(*blocked)
    _GUARD.__enter__()


def _resolve(target: str):
    mod, attr = target.split(":")
    return getattr(importlib.import_module(mod), attr)


def _work(kind: str, target: str, inputs: list[str], args: dict, out: str) -> dict:
    """Runs in a worker: loads the input volume(s), calls the function, writes the picture / output into `out`."""
    from bamr.tools.finding import PROTOCOL_V2
    out = Path(out)
    t = time.perf_counter()
    try:
        fn = _resolve(target)
        xs = [np.load(p) for p in inputs]
        pic = out / "picture.png"
        output = False
        if kind == "tool":
            data = json.loads(fn(xs[0], PROTOCOL_V2, picture=pic).to_json())
        elif kind in ("fix", "fixp"):  # fixp: a repair with arguments (⑤'s numbers from the label)
            y, rep = fn(xs[0], PROTOCOL_V2, **args) if kind == "fixp" else fn(xs[0], PROTOCOL_V2)
            data = json.loads(rep.to_json())
            if rep.changed:
                np.save(out / "output.npy", np.ascontiguousarray(y, np.float32))
                output = True
        elif kind == "look":
            data = fn(xs[0], pic, **args)
        elif kind == "compare":
            data = fn(xs[0], xs[1], pic)
        else:
            raise ValueError(f"unknown kind {kind}")
    except Exception as e:  # no crash inside an agent loop (D68): the error becomes the answer
        data, output = {"error": f"{type(e).__name__}: {e}"}, False
    return {"data": data, "picture": (out / "picture.png").exists(), "output": output,
            "seconds": round(time.perf_counter() - t, 3)}


class Compute:
    def __init__(self, cache: Path, workers: int, blocked: tuple[Path, ...], table: dict = TABLE):
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.workers, self.blocked, self.table = workers, tuple(str(p) for p in blocked), table
        self.computed: Counter = Counter()
        self._lock = threading.Lock()
        self._inflight: dict[str, Future] = {}
        self._pool = self._new_pool() if workers > 0 else None
        self.pools = int(self._pool is not None)  # pools started (a dead worker breaks its pool)

    def _new_pool(self):
        return ProcessPoolExecutor(max_workers=self.workers, mp_context=get_context("spawn"), initializer=_init,
                                   initargs=(self.blocked,))

    def close(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True, cancel_futures=True)
            self._pool = None

    @staticmethod
    def key(fn: str, args: dict, hashes: list[str]) -> str:
        return hashlib.sha256(json.dumps([fn, args, hashes], sort_keys=True).encode()).hexdigest()

    def _dir(self, key: str) -> Path:
        return self.cache / key[:2] / key

    @staticmethod
    def _load(d: Path, cached: bool) -> Outcome:
        r = json.loads((d / "result.json").read_text())
        return Outcome(r["data"], d / "picture.png" if r["picture"] else None,
                       d / "output.npy" if r["output"] else None, r["seconds"], cached)

    def call(self, fn: str, inputs: list[tuple[Path, str]], args: dict | None = None) -> Outcome:
        """fn on the volume(s) (npy path, content hash); from the cache when it was computed before."""
        if fn not in self.table:
            raise KeyError(f"unknown function {fn}")
        args = args or {}
        key = self.key(fn, args, [h for _, h in inputs])
        d = self._dir(key)
        with self._lock:
            if (d / "result.json").exists():
                return self._load(d, cached=True)
            fut = self._inflight.get(key)
            owner = fut is None
            if owner:
                fut = self._inflight[key] = Future()
                self.computed[fn] += 1
        if not owner:
            return dataclasses.replace(fut.result(), cached=True)
        try:
            res = self._compute(fn, inputs, args, d)
            fut.set_result(res)
            return res
        except BaseException as e:
            fut.set_exception(e)
            raise
        finally:
            with self._lock:
                self._inflight.pop(key, None)

    def _submit(self, job: tuple, tries: int = 3) -> dict:
        """A worker that dies (memory) breaks the pool and every job in it. The first thread to see a given broken pool
        replaces it, once, and shuts the old one down; the others go on to the new one. After `tries` deaths on this
        call, the call ends as an error (not raised, not cached)."""
        for _ in range(tries):
            with self._lock:
                pool = self._pool
            try:
                return pool.submit(_work, *job).result()
            except BrokenProcessPool:
                with self._lock:
                    old = pool if self._pool is pool else None
                    if old is not None:
                        self._pool = self._new_pool()
                        self.pools += 1
                if old is not None:
                    old.shutdown(wait=False, cancel_futures=True)
        return {"data": {"error": f"a compute worker died {tries} times on this call (out of memory?)"},
                "picture": False, "output": False, "seconds": 0.0}

    def _compute(self, fn: str, inputs, args: dict, d: Path) -> Outcome:
        kind, target = self.table[fn]
        tmp = d.parent / f"{d.name}.tmp-{uuid.uuid4().hex[:8]}"
        tmp.mkdir(parents=True)
        job = (kind, target, [str(p) for p, _ in inputs], args, str(tmp))
        r = _work(*job) if self._pool is None else self._submit(job)
        if "error" in r["data"]:
            shutil.rmtree(tmp, ignore_errors=True)
            return Outcome(r["data"], None, None, r["seconds"], False)
        (tmp / "result.json").write_text(json.dumps({"fn": fn, "args": args, "inputs": [h for _, h in inputs], **r},
                                                    ensure_ascii=False))
        try:
            tmp.rename(d)
        except OSError:  # another process finished the same key first: keep theirs
            shutil.rmtree(tmp, ignore_errors=True)
        return self._load(d, cached=False)
