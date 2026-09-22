"""Minimal OpenAI-compatible course client and deterministic test client."""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass
from typing import Protocol, Sequence


class ChatClient(Protocol):
    def chat(self, messages: Sequence[dict[str, str]]) -> str:
        """Return assistant text for the supplied bounded conversation."""


@dataclass
class OpenAICompatibleClient:
    model: str
    api_key: str
    base_url: str = "http://golem:8000/v1"
    temperature: float = 0.0
    seed: int = 0
    timeout_seconds: float = 120.0
    max_tokens: int = 128

    def chat(self, messages: Sequence[dict[str, str]]) -> str:
        if not self.api_key:
            raise ValueError("api_key must be provided")
        payload = json.dumps(
            {
                "model": self.model,
                "messages": list(messages),
                "stream": False,
                "temperature": self.temperature,
                "seed": self.seed,
                "max_tokens": self.max_tokens,
            }
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/chat/completions",
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            data = json.loads(response.read().decode("utf-8"))
        return str(data["choices"][0]["message"]["content"])


class ScriptedClient:
    """Deterministic fake for tests; responses are returned in order."""

    def __init__(self, responses: Sequence[str]) -> None:
        self._responses = iter(responses)
        self.messages_seen: list[tuple[dict[str, str], ...]] = []

    def chat(self, messages: Sequence[dict[str, str]]) -> str:
        self.messages_seen.append(tuple(messages))
        try:
            return next(self._responses)
        except StopIteration as error:
            raise RuntimeError("scripted responses exhausted") from error
