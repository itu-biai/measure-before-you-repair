"""Task 4 Stage 1: the OpenRouter client (D134) — pinned provider, no sampling settings, retries; no API call."""
import json

import httpx
import pytest

from bamr.agent.openrouter import MODELS, ApiError, Client, load_key, request_body

MSGS = [{"role": "user", "content": "hi"}]
TOOLS = [{"type": "function", "function": {"name": "f", "description": "d",
                                           "parameters": {"type": "object", "properties": {}}}}]


def _ok(msg=None, gen="gen-1"):
    return {"id": gen, "choices": [{"finish_reason": "stop", "message": msg or {"role": "assistant", "content": "7"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "cost": 0.0001}}


def _client(handler, model="haiku"):
    sleeps = []
    c = Client("sk-or-secret", MODELS[model], transport=httpx.MockTransport(handler), sleep=sleeps.append)
    return c, sleeps


def test_body_pinned_no_sampling():
    want = {"haiku": ("anthropic/claude-haiku-5.5", "anthropic"), "luna": ("openai/gpt-6-luna", "openai"),
            "qwen": ("qwen/qwen3.8-27b", "deepinfra"),  # Task 5 (D157): ④ only, after the others
            "terra": ("openai/gpt-6.1-terra", "openai"), "sonnet": ("anthropic/claude-sonnet-5.5", "anthropic")}
    for name, (slug, prov) in want.items():
        b = request_body(MODELS[name], MSGS, TOOLS)
        assert b["model"] == slug
        assert b["provider"]["only"] == [prov]
        assert b["provider"]["allow_fallbacks"] is False and b["provider"]["require_parameters"] is True
        assert not {"temperature", "top_p", "top_k", "seed"} & set(b)
        assert b["tool_choice"] == "auto"
        assert "parallel_tool_calls" not in b  # no endpoint lists it: require_parameters would refuse all
        assert b["reasoning"] == {"effort": MODELS[name].effort} and MODELS[name].effort == "medium"
        assert b["messages"] is MSGS and b["tools"] is TOOLS
    assert request_body(MODELS["qwen"], MSGS, TOOLS)["provider"]["quantizations"] == ["bf16"]
    assert "quantizations" not in request_body(MODELS["haiku"], MSGS, TOOLS)["provider"]
    assert request_body(MODELS["haiku"], MSGS, TOOLS)["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in request_body(MODELS["luna"], MSGS, TOOLS)
    assert request_body(MODELS["sonnet"], MSGS, TOOLS)["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in request_body(MODELS["terra"], MSGS, TOOLS)
    assert "tools" not in request_body(MODELS["haiku"], MSGS, None)


def test_retry_then_ok():
    seen = []

    def h(req):
        seen.append(req)
        if len(seen) == 1:
            return httpx.Response(429, headers={"Retry-After": "3"}, json={"error": {"code": 429}})
        if len(seen) == 2:
            return httpx.Response(502, json={"error": {"code": 502}})
        if len(seen) == 3:
            return httpx.Response(200, json={"error": {"code": 503, "message": "provider down"}})
        return httpx.Response(200, json=_ok())

    c, sleeps = _client(h)
    r = c.chat(MSGS, TOOLS)
    assert len(seen) == 4 and r.message["content"] == "7" and r.gen_id == "gen-1"
    assert sleeps[0] == 3 and len(sleeps) == 3
    assert seen[0].headers["Authorization"] == "Bearer sk-or-secret"
    body = json.loads(seen[0].content)
    assert body["provider"]["only"] == ["anthropic"]


def test_400_raises():
    c, _ = _client(lambda req: httpx.Response(400, json={"error": {"code": 400, "message": "bad tool"}}))
    with pytest.raises(ApiError) as e:
        c.chat(MSGS, TOOLS)
    assert e.value.status == 400 and "bad tool" in e.value.text


def test_gives_up_after_retries():
    c, sleeps = _client(lambda req: httpx.Response(503, json={"error": {"code": 503}}))
    with pytest.raises(ApiError):
        c.chat(MSGS, TOOLS)
    assert len(sleeps) == 5


def test_timeout_retried():
    n = []

    def h(req):
        n.append(1)
        if len(n) == 1:
            raise httpx.ReadTimeout("slow", request=req)
        return httpx.Response(200, json=_ok())

    c, _ = _client(h)
    assert c.chat(MSGS, TOOLS).message["content"] == "7"


def test_reasoning_details_kept():
    rd = [{"type": "reasoning.encrypted", "data": "abc", "id": "r1"}]
    msg = {"role": "assistant", "content": None, "reasoning": "hidden", "reasoning_details": rd, "refusal": None,
           "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "f", "arguments": "{}"}}]}
    c, _ = _client(lambda req: httpx.Response(200, json=_ok(msg)))
    r = c.chat(MSGS, TOOLS)
    assert r.message == {"role": "assistant", "content": None, "tool_calls": msg["tool_calls"],
                         "reasoning_details": rd}
    assert r.usage["cost"] == 0.0001 and r.finish_reason == "stop"


def test_generation_waits_for_record():
    n = []

    def h(req):
        n.append(req.url.params["id"])
        return httpx.Response(404) if len(n) < 3 else httpx.Response(200, json={"data": {"provider_name": "Anthropic"}})

    c, sleeps = _client(h)
    assert c.generation("gen-9") == {"provider_name": "Anthropic"} and n == ["gen-9"] * 3


def test_load_key(tmp_path):
    p = tmp_path / ".env"
    p.write_text("# comment\n\nOTHER=1\nOPENROUTER_API_KEY=sk-or-v1-abc\n")
    assert load_key(p) == "sk-or-v1-abc"
    p.write_text("OPENROUTER_API_KEY=\n")
    with pytest.raises(RuntimeError):
        load_key(p)
    p.write_text("# OPENROUTER_API_KEY=sk-x\n")
    with pytest.raises(RuntimeError):
        load_key(p)


def test_key_not_in_repr_or_errors():
    c, _ = _client(lambda req: httpx.Response(401, json={"error": {"code": 401, "message": "no auth"}}))
    assert "secret" not in repr(c)
    with pytest.raises(ApiError) as e:
        c.chat(MSGS, TOOLS)
    assert "secret" not in str(e.value)


def test_non_json_and_string_errors_are_api_errors():
    """Review: a 200 that is not JSON is retried; an error given as a plain string ends as ApiError, not a crash."""
    c, sleeps = _client(lambda req: httpx.Response(200, text="<html>gateway</html>"))
    with pytest.raises(ApiError):
        c.chat(MSGS, TOOLS)
    assert len(sleeps) == 5
    c, _ = _client(lambda req: httpx.Response(200, json={"error": "something broke"}))
    with pytest.raises(ApiError) as e:
        c.chat(MSGS, TOOLS)
    assert "something broke" in str(e.value)


def test_empty_tool_calls_dropped():
    """Review: a text reply with tool_calls [] or null goes back without the key (OpenAI refuses an empty array)."""
    for tc in ([], None):
        msg = {"role": "assistant", "content": "hi", "tool_calls": tc}
        c, _ = _client(lambda req, m=msg: httpx.Response(200, json=_ok(m)))
        assert c.chat(MSGS, TOOLS).message == {"role": "assistant", "content": "hi"}


def test_generations_never_raise():
    def h(req):
        if req.url.params["id"] == "gen-bad":
            raise httpx.ConnectError("down", request=req)
        return httpx.Response(200, json={"data": {"id": req.url.params["id"], "total_cost": 0.1}})

    c, _ = _client(h)
    recs = c.generations(["gen-1", "gen-bad"])
    assert recs[0]["total_cost"] == 0.1 and recs[1]["id"] == "gen-bad" and "ConnectError" in recs[1]["error"]
