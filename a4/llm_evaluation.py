"""Model-call lifecycle for a student-designed LLM evaluation function."""

import math
from collections.abc import Callable, Sequence

from evaluations import EvaluationFunction
from jetan import JetanBoard, Player
from llm_client import ChatClient

ChatMessages = Sequence[dict[str, str]]
EvaluationPromptBuilder = Callable[[JetanBoard, Player], ChatMessages]
EvaluationResponseParser = Callable[[str], float]


class LLMEvaluationFunction:
    """Evaluate cutoff states through student-supplied prompt and parser functions."""

    def __init__(
        self,
        client: ChatClient,
        prompt_builder: EvaluationPromptBuilder,
        response_parser: EvaluationResponseParser,
        fallback: EvaluationFunction,
        max_model_calls: int = 64,
    ) -> None:
        if max_model_calls < 0:
            raise ValueError("max_model_calls cannot be negative")
        self.client = client
        self.prompt_builder = prompt_builder
        self.response_parser = response_parser
        self.fallback = fallback
        self.max_model_calls = max_model_calls
        self.cache: dict[tuple[JetanBoard, Player], float] = {}
        self.model_calls = 0
        self.fallback_calls = 0
        self.cache_hits = 0
        self.total_model_calls = 0
        self.total_fallback_calls = 0
        self.total_cache_hits = 0

    def begin_search(self) -> None:
        """Reset metrics that describe one complete move search."""
        self.model_calls = 0
        self.fallback_calls = 0
        self.cache_hits = 0

    def __call__(self, board: JetanBoard, perspective: Player) -> float:
        key = board, perspective
        if key in self.cache:
            self.cache_hits += 1
            self.total_cache_hits += 1
            return self.cache[key]
        if self.model_calls >= self.max_model_calls:
            return self._fallback(board, perspective)

        try:
            messages = self.prompt_builder(board, perspective)
            self.model_calls += 1
            self.total_model_calls += 1
            response = self.client.chat(messages)
            value = float(self.response_parser(response))
            if not math.isfinite(value):
                raise ValueError("evaluation response must be finite")
        except Exception:
            return self._fallback(board, perspective)
        self.cache[key] = value
        return value

    def _fallback(self, board: JetanBoard, perspective: Player) -> float:
        self.fallback_calls += 1
        self.total_fallback_calls += 1
        value = float(self.fallback(board, perspective))
        if not math.isfinite(value):
            raise ValueError("fallback evaluation must be finite")
        self.cache[board, perspective] = value
        return value
