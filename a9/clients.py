"""Embedding and chat model client protocols, live HTTP clients, and offline test doubles."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Protocol, Sequence

import httpx

from config import PLACEHOLDER_KEY


class ClientServiceError(RuntimeError):
    """Sanitized runtime error raised when an embedding or chat service call fails."""


class EmbeddingClient(Protocol):
    """Protocol for embedding document chunks and search queries."""

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one float vector per input text in the exact input order."""


class ChatClient(Protocol):
    """Protocol for bounded ReAct chat completions."""

    def chat(self, messages: Sequence[dict[str, str]]) -> str:
        """Return the model's raw text response for the given message sequence."""


@dataclass
class OpenAICompatibleEmbeddingClient:
    """OpenAI-compatible `/embeddings` client (default: Qwen3-Embedding-8B)."""

    base_url: str
    model: str
    api_key: str
    timeout_seconds: float = 300.0
    batch_size: int = 32
    max_workers: int = 2

    def _embed_single_batch(
        self, client: httpx.Client, url: str, batch: list[str]
    ) -> list[list[float]]:
        last_exc: Exception | None = None
        for attempt in range(5):
            try:
                resp = client.post(
                    url,
                    json={"model": self.model, "input": batch},
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
                if resp.status_code >= 400:
                    raise ClientServiceError(
                        f"Embedding HTTP error: status {resp.status_code}"
                    )
                body = resp.json()
                items = sorted(body["data"], key=lambda item: int(item["index"]))
                vectors = [[float(x) for x in item["embedding"]] for item in items]
                if len(vectors) != len(batch):
                    raise ClientServiceError(
                        f"Embedding count mismatch: expected {len(batch)}, got {len(vectors)}"
                    )
                return vectors
            except ClientServiceError as exc:
                last_exc = exc
                if "status 4" in str(exc) and "429" not in str(exc):
                    raise
            except Exception as exc:
                last_exc = exc
            if attempt < 4:
                time.sleep(1.5 * (attempt + 1))

        if isinstance(last_exc, ClientServiceError):
            raise last_exc
        raise ClientServiceError(
            f"Embedding request failed: {type(last_exc).__name__}"
        ) from None

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not self.api_key or self.api_key == PLACEHOLDER_KEY:
            raise ClientServiceError(
                "Embedding API key is missing or set to placeholder."
            )
        if not texts:
            return []

        url = f"{self.base_url.rstrip('/')}/embeddings"
        batches = [
            list(texts[start : start + self.batch_size])
            for start in range(0, len(texts), self.batch_size)
        ]

        with httpx.Client(timeout=self.timeout_seconds) as client:
            if len(batches) == 1 or self.max_workers <= 1:
                results = [self._embed_single_batch(client, url, b) for b in batches]
            else:
                with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                    results = list(
                        pool.map(
                            lambda b: self._embed_single_batch(client, url, b),
                            batches,
                        )
                    )

        all_vectors: list[list[float]] = []
        for vec_batch in results:
            all_vectors.extend(vec_batch)
        return all_vectors


@dataclass
class OpenAICompatibleChatClient:
    """OpenAI-compatible `/chat/completions` client (default: Qwen3.8-27B)."""

    base_url: str
    model: str
    api_key: str
    temperature: float = 0.0
    seed: int = 4300
    max_tokens: int = 4096
    timeout_seconds: float = 300.0

    def chat(self, messages: Sequence[dict[str, str]]) -> str:
        if not self.api_key or self.api_key == PLACEHOLDER_KEY:
            raise ClientServiceError("LLM API key is missing or set to placeholder.")

        url = f"{self.base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self.model,
            "messages": list(messages),
            "stream": False,
            "temperature": self.temperature,
            "seed": self.seed,
            "max_tokens": self.max_tokens,
        }
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                with httpx.Client(timeout=self.timeout_seconds) as client:
                    resp = client.post(
                        url,
                        json=payload,
                        headers={"Authorization": f"Bearer {self.api_key}"},
                    )
                if resp.status_code >= 400:
                    raise ClientServiceError(
                        f"Chat HTTP error: status {resp.status_code}"
                    )
                body = resp.json()
                choice = body["choices"][0]["message"]
                content = choice.get("content")
                reasoning = choice.get("reasoning_content")
                if content is None and not reasoning:
                    raise ClientServiceError("Chat response contained null content")
                text = str(content or "")
                if reasoning and "<think>" not in text:
                    text = f"<think>{str(reasoning).strip()}</think>\n{text}".strip()
                return text
            except ClientServiceError as exc:
                last_exc = exc
                if "status 4" in str(exc) and "429" not in str(exc):
                    raise
            except Exception as exc:
                last_exc = exc
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))

        if isinstance(last_exc, ClientServiceError):
            raise last_exc
        raise ClientServiceError(
            f"Chat request failed: {type(last_exc).__name__}"
        ) from None


_TOKEN_RE = re.compile(r"[a-z0-9]+")


class DeterministicHashEmbeddingClient:
    """Offline lexical feature-hashing embedder for deterministic unit tests without live services."""

    def __init__(self, dimensions: int = 128) -> None:
        if dimensions < 8:
            raise ValueError("dimensions must be >= 8")
        self.dimensions = dimensions
        self.calls: list[tuple[str, ...]] = []

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        self.calls.append(tuple(texts))
        vectors: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dimensions
            tokens = _TOKEN_RE.findall(text.lower())
            if not tokens:
                vec[0] = 1.0
                vectors.append(vec)
                continue
            for tok in tokens:
                digest = hashlib.sha256(tok.encode("utf-8")).digest()
                bucket = int.from_bytes(digest[:4], "big") % self.dimensions
                sign = 1.0 if (digest[4] % 2 == 0) else -1.0
                vec[bucket] += sign
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            vectors.append([v / norm for v in vec])
        return vectors


class ScriptedChatClient:
    """Deterministic fake chat client for offline tests; yields strings or raises Exceptions."""

    def __init__(self, responses: Sequence[str | Exception]) -> None:
        self._responses = iter(responses)
        self.messages_seen: list[tuple[dict[str, str], ...]] = []

    def chat(self, messages: Sequence[dict[str, str]]) -> str:
        self.messages_seen.append(tuple(messages))
        try:
            item = next(self._responses)
        except StopIteration as exc:
            raise RuntimeError("ScriptedChatClient responses exhausted") from exc
        if isinstance(item, Exception):
            raise item
        return item

