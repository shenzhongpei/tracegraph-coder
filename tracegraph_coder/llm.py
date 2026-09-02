from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .models import LLMResponse, Message


@dataclass(slots=True)
class LLMConfig:
    api_key: str
    model: str
    base_url: str = "https://api.openai.com/v1"
    temperature: float = 0.1
    timeout: float = 120.0
    max_retries: int = 2

    @classmethod
    def from_env(cls) -> "LLMConfig":
        tracegraph_key = os.getenv("TRACEGRAPH_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        deepseek_key = os.getenv("DEEPSEEK_API_KEY")
        api_key = tracegraph_key or openai_key or deepseek_key
        model = os.getenv("TRACEGRAPH_MODEL")
        if not api_key:
            raise RuntimeError(
                "Missing API key. Set TRACEGRAPH_API_KEY, OPENAI_API_KEY, or DEEPSEEK_API_KEY."
            )
        if not model:
            raise RuntimeError("Missing model. Set TRACEGRAPH_MODEL, for example 'gpt-4o-mini'.")
        base_url = os.getenv("TRACEGRAPH_BASE_URL")
        if not base_url and deepseek_key and api_key == deepseek_key and not tracegraph_key and not openai_key:
            base_url = "https://api.deepseek.com/v1"
        return cls(api_key=api_key, model=model, base_url=base_url or "https://api.openai.com/v1")


class OpenAICompatibleLLM:
    """Minimal OpenAI-compatible Chat Completions client.

    It intentionally avoids any agent framework. The model only decides which local
    function tool should be called; this process executes every tool by itself.
    """

    def __init__(self, config: LLMConfig):
        self.config = config

    def chat(self, messages: list[Message], tools: list[dict[str, Any]]) -> LLMResponse:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [m.to_wire() for m in messages],
            "temperature": self.config.temperature,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        last_error: Exception | None = None
        for attempt in range(self.config.max_retries + 1):
            try:
                data = self._post_json("/chat/completions", payload)
                choice = data["choices"][0]
                msg = choice.get("message", {})
                return LLMResponse(
                    content=msg.get("content") or "",
                    tool_calls=msg.get("tool_calls") or None,
                    usage=data.get("usage"),
                    finish_reason=choice.get("finish_reason"),
                    raw=data,
                )
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, KeyError) as exc:
                last_error = exc
                if attempt >= self.config.max_retries:
                    break
                time.sleep(0.8 * (2**attempt))
        raise RuntimeError(f"LLM request failed: {last_error}")

    def _post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        url = self.config.base_url.rstrip("/") + path
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=body,
            method="POST",
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
            raw = resp.read().decode("utf-8")
        return json.loads(raw)
