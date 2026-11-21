"""Minimal provider-agnostic LLM client (standard library only).

Provider is chosen from environment variables:
    ANTHROPIC_API_KEY  -> Anthropic Messages API   (model: ANTHROPIC_MODEL, default claude-sonnet-5-5)
    OPENAI_API_KEY     -> OpenAI Chat Completions  (model: OPENAI_MODEL, default gpt-4o-mini)
    (neither)          -> "mock": projects fall back to their offline logic
Force a provider with LLM_PROVIDER=anthropic|openai|mock.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass


class LLMError(RuntimeError):
    """Raised when an API call fails or no provider is configured."""


@dataclass
class Result:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0


def provider() -> str:
    forced = os.getenv("LLM_PROVIDER", "").strip().lower()
    if forced in {"anthropic", "openai", "mock"}:
        return forced
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    return "mock"


def is_live() -> bool:
    return provider() != "mock"


def model_name() -> str:
    p = provider()
    if p == "anthropic":
        return os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
    if p == "openai":
        return os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    return "offline-demo"


def _post(url: str, headers: dict, body: dict, timeout: int = 90) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"content-type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise LLMError(f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}") from e
    except urllib.error.URLError as e:
        raise LLMError(f"Network error: {e.reason}") from e


def complete(prompt: str, system: str = "", max_tokens: int = 1000, temperature: float = 0.0) -> Result:
    """Send one prompt and return the model's text plus token usage."""
    p = provider()
    if p == "anthropic":
        body = {
            "model": model_name(),
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            body["system"] = system
        data = _post(
            "https://api.anthropic.com/v1/messages",
            {"x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01"},
            body,
        )
        text = "".join(block.get("text", "") for block in data.get("content", []))
        usage = data.get("usage", {})
        return Result(text, usage.get("input_tokens", 0), usage.get("output_tokens", 0))

    if p == "openai":
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": prompt}
        ]
        data = _post(
            "https://api.openai.com/v1/chat/completions",
            {"authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
            {
                "model": model_name(),
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )
        usage = data.get("usage", {})
        return Result(
            data["choices"][0]["message"]["content"] or "",
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
        )

    raise LLMError("No API key set (ANTHROPIC_API_KEY or OPENAI_API_KEY); running in offline mode.")


def extract_json(text: str):
    """Pull the first JSON object/array out of a model reply (handles ```json fences)."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    if not starts:
        raise ValueError("no JSON found in model output")
    start = min(starts)
    end = max(text.rfind("}"), text.rfind("]"))
    if end < start:
        raise ValueError("unterminated JSON in model output")
    return json.loads(text[start : end + 1])
