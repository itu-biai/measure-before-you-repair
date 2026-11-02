"""Task 4 CLI: the digit smoke test, and the systems on dev volumes of blind-mri-v2 (D136–D140; Stage 2 D144–D152):
② chain, ②u chain_u, ③ router, ③b router_readings, ④ agent, ④a agent_eyes. ⑤ (oracle) runs in score_agent.py.

  uv run python scripts/run_agent.py smoke --model haiku              # → ../dataset/task4-results/<out>/smoke/haiku.json
  uv run python scripts/run_agent.py run --system chain                # the 10 check volumes (D140), no model
  uv run python scripts/run_agent.py run --system agent --model haiku  # → <out>/agent-haiku/<case>/
  options: --cases check|develop|heldout|test|d0007,d0042 · --effort none|low|medium|high · --threads 6 · --workers 2
           · --out stage1-check

Everything runs inside blind_guard (refs, labels, manifest, splits), here and in every compute worker; the runner reads
inputs/<case>.nii.gz and hands the loop the array alone. A case with result.json is done and skipped, so a stopped run
restarts where it was; a case folder without one is cleared and run again. A case that ended on our network (no
reply from the server after the retries) moves to <out>/_network/ with its files, so it runs again (D165); one that
ended on OpenRouter's rate limit for our account (429 after the retries) moves to <out>/_ratelimit/ the same way (D167).
Each case gets messages.jsonl (pictures as files), api.jsonl, generations.jsonl (OpenRouter's per-call record:
provider, tokens, cost), result.json and output.nii.gz. A model runs only after its smoke test passed today with the
same settings (D135). The test set (t*) was refused until Task 5 (opened 30 Oct, after the Task 4 after-doc was
approved).
"""
import argparse
import concurrent.futures as cf
import dataclasses
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")

from bamr.agent import prompts  # noqa: E402
from bamr.agent.baselines import run_chain, run_router  # noqa: E402
from bamr.agent.compute import Compute  # noqa: E402
from bamr.agent.functions import ANSWER, TOOLS, TOOLS_EYES  # noqa: E402
from bamr.agent.loop import FUSE_CALLS, FUSE_SECONDS, CaseLog, run_agent  # noqa: E402
from bamr.agent.openrouter import MODELS, Client, load_key, request_body  # noqa: E402
from bamr.agent.report import page  # noqa: E402
from bamr.agent.smoke import run_smoke, same_settings  # noqa: E402
from bamr.config import REPO, load_config  # noqa: E402
from bamr.data.blindio import read_volume  # noqa: E402
from bamr.eval.guard import blind_guard  # noqa: E402

CFG = load_config(REPO / "configs" / "dataset_v2.yaml")
DS = CFG["paths"]["out"]
RESULTS = DS.parent / "task4-results"
BLOCKED = (DS / "refs", DS / "labels", DS / "manifest.csv", DS / "splits.json")
CHECK = ["d0058", "d0017", "d0034", "d0019", "d0028", "d0042", "d0007", "d0039", "d0040", "d0041"]  # D140, seed 20261029
# D144: develop on Stage 1's 10 + 10 drawn with seed 20261030; the other 40 held out, run once after the freeze
DEVELOP = ["d0002", "d0007", "d0015", "d0017", "d0019", "d0021", "d0027", "d0028", "d0032", "d0033", "d0034",
           "d0035", "d0037", "d0039", "d0040", "d0041", "d0042", "d0045", "d0055", "d0058"]
HELD_OUT = ["d0001", "d0003", "d0004", "d0005", "d0006", "d0008", "d0009", "d0010", "d0011", "d0012", "d0013",
           "d0014", "d0016", "d0018", "d0020", "d0022", "d0023", "d0024", "d0025", "d0026", "d0029", "d0030",
           "d0031", "d0036", "d0038", "d0043", "d0044", "d0046", "d0047", "d0048", "d0049", "d0050", "d0051",
           "d0052", "d0053", "d0054", "d0056", "d0057", "d0059", "d0060"]
SYSTEMS = ("chain", "chain_u", "router", "router_readings", "agent", "agent_eyes")
MODEL_FREE = ("chain", "chain_u")
_PRINT = threading.Lock()


def say(*a):
    with _PRINT:
        print(*a, flush=True)


def model_cfg(name: str, effort: str | None):
    cfg = MODELS[name]
    return dataclasses.replace(cfg, effort=effort) if effort else cfg


def cases(spec: str) -> list[str]:
    named = {"check": CHECK, "develop": DEVELOP, "heldout": HELD_OUT}
    if spec == "test":  # Task 5 (opened 30 Oct): the 148 test volumes, by their input file names
        return sorted(p.name[:5] for p in (DS / "inputs").glob("t*.nii.gz") if re.fullmatch(r"t\d{4}", p.name[:5]))
    ids = list(named[spec]) if spec in named else [c.strip() for c in spec.split(",") if c.strip()]
    bad = [c for c in ids if not re.fullmatch(r"[dst]\d{4}", c)]
    if bad:
        sys.exit(f"case ids are d*, s* or t* followed by four digits; refused: {bad}")
    return ids


def git_state() -> dict:
    run = lambda *a: subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()  # noqa
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain", "--", "src", "scripts"))}


def texts() -> dict:
    """Everything the model reads, word for word (R7): each round's run.json keeps it."""
    return {"system": prompts.SYSTEM, "first": prompts.FIRST, "router": prompts.ROUTER,
            "router_readings": prompts.ROUTER_READINGS, "system_eyes": prompts.SYSTEM_EYES, "tools": TOOLS,
            "tools_eyes": TOOLS_EYES, "answer": ANSWER}


def text_hashes() -> dict:
    h = lambda o: hashlib.sha256(json.dumps(o, ensure_ascii=False, sort_keys=True).encode()).hexdigest()  # noqa
    return {k: h(v) for k, v in texts().items()}


def smoke_file(out, cfg_name: str):
    return out / "smoke" / f"{cfg_name}.json"


def cmd_smoke(a):
    cfg = model_cfg(a.model, a.effort)
    out = RESULTS / a.out
    client = Client(load_key(REPO / ".env"), cfg)
    res = run_smoke(client, out / "smoke" / f"{a.model}-files")
    recs = client.generations(res["gen_ids"])
    res |= {"model": dataclasses.asdict(cfg), "date": datetime.date.today().isoformat(), "generations": recs,
            "body": {k: v for k, v in request_body(cfg, [], None).items() if k != "messages"}}
    p = smoke_file(out, a.model)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    say(f"smoke {a.model} (effort {cfg.effort}): {'PASS' if res['pass'] else 'FAIL'} · {res['reason']} · "
        f"{res['text'][:200]!r} · {res['seconds']} s · → {p}")
    return 0 if res["pass"] else 1


def smoke_ok(out, name: str, cfg) -> bool:
    p = smoke_file(out, name)
    if not p.exists():
        return False
    s = json.loads(p.read_text())
    return s["pass"] and s["date"] == datetime.date.today().isoformat() and same_settings(s["model"], cfg)


def network_failure(res: dict) -> bool:
    """The case ended on a transport error after the client's retries (OpenRouter 0: no reply from the server)."""
    return str(res.get("error") or "").startswith("OpenRouter 0:")


def rate_limited(res: dict) -> bool:
    """The case ended on OpenRouter's rate limit for our account after the client's retries (D167)."""
    return str(res.get("error") or "").startswith("OpenRouter 429:")


def set_aside(d: Path, res: dict) -> bool:
    """D165: our network is not the model's answer, nor (D167) our account's rate limit. The case folder moves to
    <out>/_network/ or <out>/_ratelimit/, as <system>/<case>-<n> (its log and cost kept, result.json →
    result.network.json), so the case runs again."""
    if network_failure(res):
        kind = "_network"
    elif rate_limited(res):
        kind = "_ratelimit"
    else:
        return False
    (d / "result.json").replace(d / "result.network.json")
    n = 1
    while (dest := d.parent.parent / kind / d.parent.name / f"{d.name}-{n}").exists():
        n += 1
    dest.parent.mkdir(parents=True, exist_ok=True)
    d.replace(dest)
    return True


def one_case(case, a, cfg, key, compute, folder, work_root) -> dict | None:
    d = folder / case
    if (d / "result.json").exists():
        return None
    if d.exists():
        shutil.rmtree(d)
    work = work_root / case
    if work.exists():
        shutil.rmtree(work)
    x0 = read_volume(DS / "inputs" / f"{case}.nii.gz")
    forbidden = [case, str(DS), str(REPO), "blind-mri", f"{case}.nii"]
    log = CaseLog(d)
    client = Client(key, cfg) if cfg else None
    t = time.perf_counter()
    if a.system in MODEL_FREE:
        res = run_chain(x0, compute, work, log, mode="gated" if a.system == "chain" else "ungated")
    elif a.system.startswith("router"):
        res = run_router(x0, client, compute, work, log, forbidden, readings=a.system == "router_readings")
    else:
        res = run_agent(x0, client, compute, work, log, forbidden, eyes=a.system == "agent_eyes")
    if client is not None and res["gen_ids"]:
        recs = client.generations(res["gen_ids"])
        with open(d / "generations.jsonl", "w") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in recs)
        res["generation"] = {"cost": round(sum(r.get("total_cost") or 0 for r in recs), 6),
                             "providers": sorted({r.get("provider_name") for r in recs if r.get("provider_name")}),
                             "missing": sum("error" in r for r in recs)}
        log.result(res)
    if a.system.startswith("agent"):
        page(d)
    shutil.rmtree(work, ignore_errors=True)
    say(f"{case} {folder.name}: {res['version']} · {res['calls']} calls ({res['bad_calls']} bad) · "
        f"fused {res['fused']} · error {res['error']} · ${res['usage']['cost']:.4f} · {time.perf_counter() - t:.0f} s"
        + (f" · {'RATE LIMIT' if rate_limited(res) else 'NETWORK'}: set aside, runs again" if set_aside(d, res)
           else ""))
    return res


def cmd_run(a):
    out = RESULTS / a.out
    ids = cases(a.cases)
    cfg = None if a.system in MODEL_FREE else model_cfg(a.model, a.effort)
    if cfg is not None and not smoke_ok(out, a.model, cfg):
        sys.exit(f"run the smoke test for {a.model} (effort {cfg.effort}) first: today, the same settings (D135)")
    key = load_key(REPO / ".env") if cfg else None
    folder = out / (a.system if a.system in MODEL_FREE else f"{a.system}-{a.model}")
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "run.json").write_text(json.dumps({
        "system": a.system, "model": dataclasses.asdict(cfg) if cfg else None,
        "body": {k: v for k, v in request_body(cfg, [], None).items() if k != "messages"} if cfg else None,
        "git": git_state(), "texts_sha256": text_hashes(), "texts": texts(), "cases": ids,
        "fuse": [FUSE_CALLS, FUSE_SECONDS],
        "threads": a.threads, "workers": a.workers, "started": datetime.datetime.now().isoformat(timespec="seconds")},
        indent=1))
    with blind_guard(*BLOCKED):
        compute = Compute(RESULTS / "cache", workers=a.workers, blocked=BLOCKED)
        try:
            with cf.ThreadPoolExecutor(a.threads) as ex:
                futs = {ex.submit(one_case, c, a, cfg, key, compute, folder, RESULTS / "work" / folder.name): c
                        for c in ids}
                for f in cf.as_completed(futs):
                    try:
                        f.result()
                    except Exception as e:  # one case failing stops nothing else; it reruns next time
                        say(f"{futs[f]} {folder.name}: FAILED {type(e).__name__}: {e}")
                        if isinstance(e, KeyboardInterrupt):
                            raise
        finally:
            compute.close()
    done = sum((folder / c / "result.json").exists() for c in ids)
    say(f"{folder.name}: {done}/{len(ids)} cases done; computed {dict(compute.computed)}")
    return 0 if done == len(ids) else 1


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("smoke")
    s.add_argument("--model", choices=list(MODELS), required=True)
    r = sub.add_parser("run")
    r.add_argument("--system", choices=SYSTEMS, required=True)
    r.add_argument("--model", choices=list(MODELS))
    r.add_argument("--cases", default="check")
    r.add_argument("--threads", type=int, default=6)
    r.add_argument("--workers", type=int, default=2)
    for q in (s, r):
        q.add_argument("--effort", choices=["none", "minimal", "low", "medium", "high"])
        q.add_argument("--out", default="stage1-check")
    a = p.parse_args()
    if a.cmd == "run" and a.system not in MODEL_FREE and not a.model:
        p.error("--model is needed for the routers and the agents")
    sys.exit(cmd_smoke(a) if a.cmd == "smoke" else cmd_run(a))


if __name__ == "__main__":
    main()
