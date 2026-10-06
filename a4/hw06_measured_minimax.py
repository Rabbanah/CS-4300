"""Supplied exhaustive minimax with Assignment 6 measurement semantics."""

from __future__ import annotations

import math
import time

from alpha_beta_agent import SearchMetrics
from evaluations import EvaluationFunction
from jetan import JetanBoard, Move, Player


class MeasuredMinimaxAgent:
    """Exhaustive reference using the same evaluator and semantics as alpha-beta."""

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
        self.last_value = -math.inf
        self._reset_counts()

    def _reset_counts(self) -> None:
        self._visited = 0
        self._expanded = 0
        self._generated = 0
        self._evaluated = 0
        self._maximum_depth = 0

    def choose_action(self, board: JetanBoard) -> Move:
        started = time.perf_counter()
        self._reset_counts()
        self._visit(0)
        if board.is_terminal():
            raise RuntimeError("minimax received a terminal board")
        actions = board.actions()
        self._expanded += 1
        self._generated += len(actions)
        perspective = board.player()
        best_action = actions[0]
        best_value = -math.inf
        for action in actions:
            value = self._value(board.result(action), 1, perspective)
            if value > best_value:
                best_action, best_value = action, value
        self.last_value = best_value
        self.last_metrics = SearchMetrics(
            self._visited,
            self._expanded,
            self._generated,
            self._evaluated,
            0,
            self._maximum_depth,
            time.perf_counter() - started,
        )
        self.total_metrics += self.last_metrics
        return best_action

    def _visit(self, depth: int) -> None:
        self._visited += 1
        self._maximum_depth = max(self._maximum_depth, depth)

    def _value(self, board: JetanBoard, depth: int, perspective: Player) -> float:
        self._visit(depth)
        if board.is_terminal():
            self._evaluated += 1
            return float(board.utility(perspective))
        if depth == self.depth:
            self._evaluated += 1
            value = float(self.evaluation_function(board, perspective))
            if not math.isfinite(value):
                raise ValueError("evaluation function must return a finite value")
            return value
        actions = board.actions()
        self._expanded += 1
        self._generated += len(actions)
        values = (
            self._value(board.result(action), depth + 1, perspective)
            for action in actions
        )
        return max(values) if board.player() is perspective else min(values)
