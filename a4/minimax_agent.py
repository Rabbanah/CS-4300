"""Reusable depth-limited minimax agent."""

import math
from dataclasses import dataclass

from evaluations import EvaluationFunction
from jetan import JetanBoard, Move, Player


@dataclass(frozen=True)
class SearchMetrics:
    """Work performed during the most recent move search."""

    generated: int = 0
    evaluated: int = 0
    maximum_depth: int = 0

    def __add__(self, other: "SearchMetrics") -> "SearchMetrics":
        return SearchMetrics(
            self.generated + other.generated,
            self.evaluated + other.evaluated,
            max(self.maximum_depth, other.maximum_depth),
        )


class DepthLimitedMinimaxAgent:
    """Run exhaustive minimax using a supplied cutoff evaluation function."""

    def __init__(
        self, name: str, depth: int, evaluation_function: EvaluationFunction
    ) -> None:
        if depth < 1:
            raise ValueError("depth must be at least one ply")
        self.name = name
        self.depth = depth
        self.evaluation_function = evaluation_function
        self.last_metrics = SearchMetrics()
        self.total_metrics = SearchMetrics()
        self._generated = 0
        self._evaluated = 0
        self._maximum_depth = 0

    def choose_action(self, board: JetanBoard) -> Move:
        actions = board.actions()
        if not actions:
            raise RuntimeError("minimax received a terminal board")
        perspective = board.player()
        self._generated = len(actions)
        self._evaluated = 0
        self._maximum_depth = 0
        best_action = actions[0]
        best_value = -math.inf
        for action in actions:
            value = self._value(
                board.result(action),
                self.depth - 1,
                perspective,
                current_depth=1,
            )
            if value > best_value:
                best_value = value
                best_action = action
        self.last_metrics = SearchMetrics(
            self._generated, self._evaluated, self._maximum_depth
        )
        self.total_metrics += self.last_metrics
        return best_action

    def _value(
        self,
        board: JetanBoard,
        depth_remaining: int,
        perspective: Player,
        current_depth: int,
    ) -> float:
        self._maximum_depth = max(self._maximum_depth, current_depth)
        if board.terminal_test():
            self._evaluated += 1
            return float(board.utility(perspective))
        if depth_remaining == 0:
            self._evaluated += 1
            value = float(self.evaluation_function(board, perspective))
            if not math.isfinite(value):
                raise ValueError("evaluation function must return a finite value")
            return value

        actions = board.actions()
        self._generated += len(actions)
        values = (
            self._value(
                board.result(action),
                depth_remaining - 1,
                perspective,
                current_depth + 1,
            )
            for action in actions
        )
        return max(values) if board.player() is perspective else min(values)
