from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ChatConfig:
    provider: str
    api_key: str
    base_url: str
    model: str
    temperature: float = 0.0
    timeout: int = 120
    max_retries: int = 3
    retry_sleep: float = 2.0


class ChatClient:
    """Small OpenAI-compatible chat completions client using only stdlib."""

    def __init__(self, config: ChatConfig) -> None:
        self.config = config

    def chat_json(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        url = self.config.base_url.rstrip("/") + "/chat/completions"
        body = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "response_format": {"type": "json_object"},
        }
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }

        last_error: Exception | None = None
        for attempt in range(1, self.config.max_retries + 1):
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.config.timeout) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                content = payload["choices"][0]["message"]["content"]
                return parse_json_object(content)
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
                last_error = e
                if attempt >= self.config.max_retries:
                    break
                time.sleep(self.config.retry_sleep * attempt)

        raise RuntimeError(f"{self.config.provider} chat completion failed: {last_error!r}")


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        obj = json.loads(text[start : end + 1])
    if not isinstance(obj, dict):
        raise TypeError(f"expected JSON object, got {type(obj).__name__}")
    return obj
