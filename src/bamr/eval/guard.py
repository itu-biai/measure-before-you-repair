"""Runtime path guard (D99 #10): while a tool runs, refs and labels cannot be opened or listed.

The static test keeps those words out of the tools' source; this guard also catches a path built at run time. One
process-wide audit hook (audit hooks cannot be removed, PEP 578) reads the roots blocked at the moment.
"""
import os
import sys
from contextlib import contextmanager
from pathlib import Path

_BLOCKED: tuple[str, ...] = ()
_EVENTS = {"open", "os.listdir", "os.scandir"}
_installed = False


def _norm(p) -> str:
    """Real path; case-folded on macOS, whose file names ignore case (labels/ and LABELS/ are one folder)."""
    p = os.path.realpath(os.fsdecode(p))
    return p.casefold() if sys.platform == "darwin" else p


def _hook(event, args):
    if not _BLOCKED or event not in _EVENTS or not args or isinstance(args[0], int):
        return
    p = _norm(os.getcwd() if args[0] is None else args[0])  # os.listdir() with no argument lists the cwd
    if any(p == r or p.startswith(r + os.sep) for r in _BLOCKED):
        raise PermissionError(f"blind guard: a tool tried to open {p}")


@contextmanager
def blind_guard(*roots: Path):
    """Block opening, listing and scanning anything under these roots for the length of the block."""
    global _BLOCKED, _installed
    if not _installed:
        sys.addaudithook(_hook)
        _installed = True
    before = _BLOCKED
    _BLOCKED = before + tuple(_norm(r) for r in roots)
    try:
        yield
    finally:
        _BLOCKED = before
