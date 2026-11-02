"""OpenRouter, pinned (D134): one chat-completions client for every candidate model, with httpx.

The body fixes what the route page fixed (2_models_route.html §6–§8): the provider pinned (`only`, no fallbacks,
`require_parameters`), the precision for open weights, reasoning effort fixed per model, tool choice auto, one call
per turn (enforced by the loop: parallel_tool_calls is listed by no endpoint, so require_parameters would
refuse every one), prompt caching on where the provider wants it asked for. No temperature, top_p, top_k or seed is ever sent
(Claude refuses them since API 2026-10-21; repeats show the spread). The assistant message is kept as returned:
content, tool_calls and reasoning_details, which must go back unchanged on the next turn. Each call's generation id is
kept; its /generation record (provider, tokens, cost) is fetched after the case. The key comes from the git-ignored
.env and goes only into the Authorization header.
"""
import dataclasses
import time
from pathlib import Path

import httpx

BASE = "https://openrouter.ai/api/v1"
RETRY = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529}
KEEP = ("role", "content", "tool_calls", "reasoning_details")


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    name: str
    slug: str
    provider: str
    quantizations: tuple[str, ...] | None
    effort: str
    cache: bool
    max_tokens: int = 12000


MODELS = {
    "haiku": ModelConfig("Claude Haiku 5.5", "anthropic/claude-haiku-5.5", "anthropic", None, "medium", True),
    "luna": ModelConfig("GPT-6 luna", "openai/gpt-6-luna", "openai", None, "medium", False),  # OpenAI caches itself
    "qwen": ModelConfig("Qwen3.8-27B", "qwen/qwen3.8-27b", "deepinfra", ("bf16",), "medium", True),
    # Task 5 (D157): the stronger models, on ④ only
    "terra": ModelConfig("GPT-6.1 Terra", "openai/gpt-6.1-terra", "openai", None, "medium", False),
    "sonnet": ModelConfig("Claude Sonnet 5.5", "anthropic/claude-sonnet-5.5", "anthropic", None, "medium", True),
}


class ApiError(RuntimeError):
    def __init__(self, status: int, text: str):
        super().__init__(f"OpenRouter {status}: {text[:500]}")
        self.status, self.text = status, text


@dataclasses.dataclass
class Reply:
    message: dict
    gen_id: str
    usage: dict
    finish_reason: str
    seconds: float


def load_key(env: Path) -> str:
    """OPENROUTER_API_KEY from a KEY=VALUE file (# comments and blank lines skipped)."""
    for line in Path(env).read_text().splitlines():
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() == "OPENROUTER_API_KEY" and v.strip():
            return v.strip().strip('"').strip("'")
    raise RuntimeError(f"no OPENROUTER_API_KEY in {env}")


def request_body(cfg: ModelConfig, messages: list, tools: list | None, tool_choice="auto") -> dict:
    provider = {"only": [cfg.provider], "allow_fallbacks": False, "require_parameters": True}
    if cfg.quantizations:
        provider["quantizations"] = list(cfg.quantizations)
    body = {"model": cfg.slug, "messages": messages, "provider": provider, "reasoning": {"effort": cfg.effort},
            "max_tokens": cfg.max_tokens}
    if tools:
        body |= {"tools": tools, "tool_choice": tool_choice}  # no parallel_tool_calls: no endpoint lists it
    if cfg.cache:
        body["cache_control"] = {"type": "ephemeral"}
    return body


class Client:
    def __init__(self, key: str, cfg: ModelConfig, transport=None, retries: int = 5, sleep=time.sleep,
                 timeout: float = 300.0):
        self.cfg, self.retries, self._sleep = cfg, retries, sleep
        self._http = httpx.Client(base_url=BASE, transport=transport, timeout=httpx.Timeout(timeout, connect=30.0),
                                  headers={"Authorization": f"Bearer {key}"})

    def __repr__(self):
        return f"Client({self.cfg.slug} @ {self.cfg.provider})"

    def _wait(self, k: int, resp: httpx.Response | None) -> None:
        after = resp.headers.get("Retry-After") if resp is not None else None
        try:
            s = float(after) if after else 2.0 * 2 ** k
        except ValueError:
            s = 2.0 * 2 ** k
        self._sleep(min(s, 120.0))

    def _post(self, body: dict) -> dict:
        last = None
        for k in range(self.retries + 1):
            resp = None
            try:
                resp = self._http.post("/chat/completions", json=body)
                if resp.status_code == 200:
                    try:
                        data = resp.json()
                    except ValueError:  # a gateway page instead of JSON: try again
                        data = {"error": {"code": 502, "message": f"reply is not JSON: {resp.text[:200]}"}}
                    err = data.get("error") or (data.get("choices") or [{}])[0].get("error")
                    if not err:
                        return data
                    err = err if isinstance(err, dict) else {"message": str(err)}
                    code = int(err.get("code") or 0) if str(err.get("code") or "").isdigit() else 0
                    last = ApiError(code or 200, str(err))
                    if code not in RETRY:
                        raise last
                elif resp.status_code in RETRY or resp.status_code == 402 and "Retry-After" in resp.headers:
                    last = ApiError(resp.status_code, resp.text)
                else:
                    raise ApiError(resp.status_code, resp.text)
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last = ApiError(0, f"{type(e).__name__}: {e}")
            if k < self.retries:
                self._wait(k, resp)
        raise last

    def chat(self, messages: list, tools: list | None, tool_choice="auto") -> Reply:
        t = time.perf_counter()
        data = self._post(request_body(self.cfg, messages, tools, tool_choice))
        ch = (data.get("choices") or [None])[0]
        if not ch or "message" not in ch:
            raise ApiError(200, f"no message in the reply: {str(data)[:300]}")
        msg = {k: ch["message"].get(k) for k in KEEP if k in ch["message"]}
        msg["role"] = "assistant"
        if not msg.get("tool_calls"):  # [] or null: sent back, OpenAI refuses an empty array
            msg.pop("tool_calls", None)
        return Reply(msg, data.get("id", ""), data.get("usage") or {}, ch.get("finish_reason") or "",
                     round(time.perf_counter() - t, 3))

    def generations(self, gen_ids: list[str]) -> list[dict]:
        """Every call's record; a record that cannot be fetched is kept as {"id", "error"} (D134: none dropped)."""
        out = []
        for g in gen_ids:
            try:
                out.append(self.generation(g))
            except Exception as e:  # noqa: BLE001 — after the case: nothing may stop the records being written
                out.append({"id": g, "error": f"{type(e).__name__}: {e}"})
        return out

    def generation(self, gen_id: str, wait: float = 60.0) -> dict:
        """The /generation record of one call (provider, native tokens, cost); it appears a little after the call."""
        waited = 0.0
        while True:
            resp = self._http.get("/generation", params={"id": gen_id})
            if resp.status_code == 200:
                return resp.json().get("data") or {}
            if resp.status_code not in (404, *RETRY) or waited >= wait:
                raise ApiError(resp.status_code, resp.text)
            self._sleep(2.0)
            waited += 2.0
