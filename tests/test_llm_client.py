import json

import httpx
import pytest
from pydantic import BaseModel

from residual_room.llm.client import (
    Endpoint,
    LLMError,
    RateLimiter,
    ResponseCache,
    RoleLLM,
    _check_judge_differs,
    load_role,
    parse_json,
)


class Verdict(BaseModel):
    supported: bool
    reason: str


def reply(content, status=200):
    return httpx.Response(status, json={"choices": [{"message": {"content": content}}]})


def make_llm(handler, tmp_path, endpoints=("a",)):
    calls = []

    def record(request):
        calls.append(json.loads(request.content)["model"])
        return handler(request, len(calls))

    eps = [Endpoint(p, f"model-{p}", f"https://{p}.test/v1", "key", RateLimiter(6000), 1) for p in endpoints]
    llm = RoleLLM("judge", eps, 0, 100, ResponseCache(tmp_path), httpx.MockTransport(record), retries=2)
    return llm, calls


@pytest.mark.parametrize(
    "text",
    [
        '{"supported": false, "reason": "x"}',
        '```json\n{"supported": false, "reason": "x"}\n```',
        'Here is my answer: {"supported": false, "reason": "x"} hope that helps',
    ],
)
def test_parse_json_tolerates_fences_and_preamble(text):
    assert parse_json(text) == {"supported": False, "reason": "x"}


def test_invalid_json_gets_one_corrective_retry(tmp_path):
    llm, calls = make_llm(
        lambda req, n: reply("not json") if n == 1 else reply('{"supported": true, "reason": "ok"}'),
        tmp_path,
    )
    assert llm.ask_json("sys", "user", Verdict).supported is True
    assert len(calls) == 2


def test_still_invalid_after_retry_raises(tmp_path):
    llm, _ = make_llm(lambda req, n: reply("still not json"), tmp_path)
    with pytest.raises(LLMError):
        llm.ask_json("sys", "user", Verdict)


def test_cache_avoids_second_call(tmp_path):
    llm, calls = make_llm(lambda req, n: reply('{"supported": true, "reason": "ok"}'), tmp_path)
    llm.ask_json("sys", "same question", Verdict)
    llm.ask_json("sys", "same question", Verdict)
    assert len(calls) == 1


def test_falls_back_when_primary_is_not_found(tmp_path):
    def handler(req, n):
        model = json.loads(req.content)["model"]
        return reply("", 404) if model == "model-a" else reply("from b")

    llm, calls = make_llm(handler, tmp_path, endpoints=("a", "b"))
    assert llm.ask_text("sys", "user") == "from b"
    assert calls == ["model-a", "model-b"]


def test_truncated_reply_is_not_cached_and_falls_back(tmp_path):
    def handler(req, n):
        if json.loads(req.content)["model"] == "model-a":
            return httpx.Response(200, json={"choices": [{"message": {"content": "Let me think..."},
                                                          "finish_reason": "length"}]})
        return reply('{"supported": true, "reason": "ok"}')

    llm, calls = make_llm(handler, tmp_path, endpoints=("a", "b"))
    assert llm.ask_json("sys", "user", Verdict).supported is True
    assert calls == ["model-a", "model-b"]
    assert len(list(tmp_path.glob("*.json"))) == 1  # only the complete answer was cached


def test_retryable_status_is_retried(tmp_path):
    llm, calls = make_llm(lambda req, n: reply("", 503) if n == 1 else reply("ok"), tmp_path)
    llm.endpoints[0].limiter = RateLimiter(6000)
    assert llm.ask_text("sys", "u") == "ok" and len(calls) == 2


def test_judge_and_generator_must_differ():
    cfg = {"roles": {"judge": [{"model": "m"}], "generator": [{"model": "m:free"}]}}
    with pytest.raises(ValueError):
        _check_judge_differs(cfg)


def test_configured_roles_load_and_differ():
    judge, generator = load_role("judge"), load_role("generator")
    assert judge.model != generator.model
    assert judge.temperature == 0
