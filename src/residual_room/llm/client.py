"""A small client for OpenAI-compatible chat APIs (NVIDIA, OpenRouter).

One `RoleLLM` per role (judge, generator) from config/models.yaml. It tries the
role's endpoints in order (primary, then fallback), rate-limits per provider,
caches responses on disk, and `ask_json` validates replies against a pydantic
model with one corrective retry. A judge failure raises LLMError; callers turn
that into a blocking result, never a silent pass.
"""

import hashlib
import json
import os
import re
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import TypeVar

import httpx
import yaml
from dotenv import dotenv_values
from pydantic import BaseModel, ValidationError

from residual_room.corpus.manifest import REPO_ROOT

MODELS_PATH = REPO_ROOT / "config" / "models.yaml"
CACHE_DIR = REPO_ROOT / ".cache" / "llm"
RETRYABLE = {408, 409, 429, 500, 502, 503, 504}

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


class RateLimiter:
    """At most `rpm` request starts per minute, shared by all threads."""

    def __init__(self, rpm: int) -> None:
        self.interval = 60.0 / rpm
        self.lock = threading.Lock()
        self.next_at = 0.0

    def wait(self) -> None:
        with self.lock:
            now = time.monotonic()
            start = max(now, self.next_at)
            self.next_at = start + self.interval
        time.sleep(max(0.0, start - now))


@dataclass
class Endpoint:
    provider: str
    model: str
    base_url: str
    api_key: str | None
    limiter: RateLimiter
    max_concurrency: int


class ResponseCache:
    def __init__(self, directory: Path = CACHE_DIR, enabled: bool = True) -> None:
        self.directory = directory
        self.enabled = enabled

    def key(self, payload: dict) -> str:
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def get(self, key: str) -> dict | None:
        path = self.directory / f"{key}.json"
        if self.enabled and path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
        return None

    def put(self, key: str, value: dict) -> None:
        if self.enabled:
            self.directory.mkdir(parents=True, exist_ok=True)
            (self.directory / f"{key}.json").write_text(json.dumps(value), encoding="utf-8")


class RoleLLM:
    def __init__(
        self,
        role: str,
        endpoints: list[Endpoint],
        temperature: float,
        max_tokens: int,
        cache: ResponseCache | None = None,
        transport: httpx.BaseTransport | None = None,
        retries: int = 3,
        timeout: float = 180,
    ) -> None:
        self.role = role
        self.endpoints = endpoints
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cache = cache or ResponseCache()
        self.http = httpx.Client(timeout=timeout, transport=transport)
        self.retries = retries

    @property
    def model(self) -> str:
        return self.endpoints[0].model

    @property
    def max_concurrency(self) -> int:
        return self.endpoints[0].max_concurrency

    def ask_text(self, system: str, user: str) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        # Keyed on the primary model: a fallback answer is cached as the role's answer.
        key = self.cache.key(
            {"model": self.model, "messages": messages, "t": self.temperature, "max": self.max_tokens}
        )
        if hit := self.cache.get(key):
            return hit["text"]
        errors = []
        for ep in self.endpoints:
            try:
                text = self._call(ep, messages)
            except LLMError as exc:
                errors.append(f"{ep.provider}/{ep.model}: {exc}")
                continue
            self.cache.put(key, {"text": text, "served_by": f"{ep.provider}/{ep.model}"})
            return text
        raise LLMError("; ".join(errors))

    def ask_json(self, system: str, user: str, schema: type[T]) -> T:
        text = self.ask_text(system, user)
        try:
            return schema.model_validate(parse_json(text))
        except (ValueError, ValidationError) as exc:
            # One corrective retry, with the parse error shown to the model.
            fix = f"{user}\n\nYour previous reply was not valid JSON for the schema ({exc}). Reply with only the JSON object."
            text = self.ask_text(system, fix)
            try:
                return schema.model_validate(parse_json(text))
            except (ValueError, ValidationError) as exc2:
                raise LLMError(f"invalid JSON from {self.role} after retry: {exc2}") from exc2

    def _call(self, ep: Endpoint, messages: list[dict]) -> str:
        if not ep.api_key:
            raise LLMError("no API key set")
        body = {
            "model": ep.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        for attempt in range(self.retries):
            ep.limiter.wait()
            try:
                r = self.http.post(
                    f"{ep.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {ep.api_key}"},
                    json=body,
                )
            except httpx.TransportError as exc:
                if attempt == self.retries - 1:
                    raise LLMError(f"network error: {exc}") from exc
                time.sleep(2**attempt)
                continue
            if r.status_code == 200:
                content = (r.json()["choices"][0]["message"].get("content") or "").strip()
                if not content:
                    raise LLMError("empty reply (max_tokens may be too low for a reasoning model)")
                return content
            if r.status_code not in RETRYABLE or attempt == self.retries - 1:
                raise LLMError(f"HTTP {r.status_code}: {r.text[:200]}")
            time.sleep(float(r.headers.get("retry-after", 2**attempt * 3)))
        raise LLMError("retries exhausted")


def parse_json(text: str) -> dict:
    """The first JSON object in a reply, tolerating ``` fences and preamble."""
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object in reply")
    obj, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(obj, dict):
        raise ValueError("reply is not a JSON object")
    return obj


def load_role(role: str, path: Path = MODELS_PATH, **kwargs) -> RoleLLM:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    _check_judge_differs(cfg)
    keys = {**dotenv_values(REPO_ROOT / ".env"), **os.environ}
    limiters = _limiters(path)
    endpoints = []
    for entry in cfg["roles"][role]:
        prov = cfg["providers"][entry["provider"]]
        endpoints.append(
            Endpoint(
                provider=entry["provider"],
                model=entry["model"],
                base_url=prov["base_url"],
                api_key=keys.get(prov["api_key_env"]),
                limiter=limiters[entry["provider"]],
                max_concurrency=prov["max_concurrency"],
            )
        )
    return RoleLLM(role, endpoints, **cfg["defaults"][role], **kwargs)


@lru_cache
def _limiters(path: Path) -> dict[str, RateLimiter]:
    # Shared per provider across roles: the judge and generator on NVIDIA share its limit.
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {name: RateLimiter(p["rpm"]) for name, p in cfg["providers"].items()}


def _check_judge_differs(cfg: dict) -> None:
    def base(model: str) -> str:
        return model.removesuffix(":free")

    judge = {base(e["model"]) for e in cfg["roles"]["judge"]}
    generator = {base(e["model"]) for e in cfg["roles"]["generator"]}
    if judge & generator:
        raise ValueError(f"judge and generator share a model: {judge & generator}")
