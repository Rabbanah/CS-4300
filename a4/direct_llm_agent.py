"""Model-call lifecycle for a student-designed direct move chooser."""

from collections.abc import Callable, Sequence

from jetan import JetanBoard, Move
from llm_client import ChatClient

ChatMessages = Sequence[dict[str, str]]
MovePromptBuilder = Callable[
    [JetanBoard, tuple[Move, ...], tuple[Move, ...]], ChatMessages
]
MoveResponseParser = Callable[[str, tuple[Move, ...]], Move]
FallbackPolicy = Callable[[JetanBoard, tuple[Move, ...]], Move]


def first_legal_action(board: JetanBoard, actions: tuple[Move, ...]) -> Move:
    """Return a deterministic infrastructure fallback."""
    del board
    return actions[0]


class DirectLLMAgent:
    """Request one move using student-supplied prompting and response handling."""

    def __init__(
        self,
        client: ChatClient,
        prompt_builder: MovePromptBuilder,
        response_parser: MoveResponseParser,
        name: str = "Direct LLM",
        fallback: FallbackPolicy = first_legal_action,
        history_limit: int = 6,
    ) -> None:
        if history_limit < 0:
            raise ValueError("history_limit cannot be negative")
        self.client = client
        self.prompt_builder = prompt_builder
        self.response_parser = response_parser
        self.name = name
        self.fallback = fallback
        self.history_limit = history_limit
        self.model_calls = 0
        self.fallback_calls = 0
        self._recent_moves: list[Move] = []

    @property
    def recent_moves(self) -> tuple[Move, ...]:
        return tuple(self._recent_moves)

    def reset(self) -> None:
        """Clear match-specific history while preserving cumulative metrics."""
        self._recent_moves.clear()

    def choose_action(self, board: JetanBoard) -> Move:
        actions = board.actions()
        if not actions:
            raise RuntimeError("DirectLLMAgent received a terminal board")
        if board.ply == 0:
            self.reset()

        try:
            messages = self.prompt_builder(board, actions, self.recent_moves)
            self.model_calls += 1
            response = self.client.chat(messages)
            action = self.response_parser(response, actions)
            if action not in actions:
                raise ValueError("model response did not select a legal action")
        except Exception:
            self.fallback_calls += 1
            action = self.fallback(board, actions)
            if action not in actions:
                raise ValueError("fallback policy did not select a legal action")

        if self.history_limit:
            self._recent_moves.append(action)
            self._recent_moves = self._recent_moves[-self.history_limit :]
        return action
