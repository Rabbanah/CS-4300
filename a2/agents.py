"""Provided interfaces and student-completed agents for U0-HW-03."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Protocol

from problem import RecoveryProblem
from recovery import Action, RecoveryState


@dataclass(frozen=True)
class SearchMetrics:
    generated: int = 0
    expanded: int = 0
    peak_frontier: int = 0
    planning_time_seconds: float = 0.0


@dataclass(frozen=True)
class SearchResult:
    plan: tuple[Action, ...] | None
    metrics: SearchMetrics


class SearchAgent(Protocol):
    def plan(self, problem: RecoveryProblem) -> SearchResult:
        """Return a complete plan and search metrics."""


class RandomRecoveryAgent:
    """Provided online baseline that chooses uniformly among legal actions."""

    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def act(
        self, state: RecoveryState, legal_actions: tuple[Action, ...]
    ) -> Action:
        del state
        if not legal_actions:
            raise RuntimeError("RandomRecoveryAgent received no legal actions")
        return self._rng.choice(legal_actions)


class UniformCostSearchAgent:
    def plan(self, problem: RecoveryProblem) -> SearchResult:
        # TODO: Implement uniform-cost graph search and metric accounting.
        raise NotImplementedError("Implement UniformCostSearchAgent.plan")


class AStarSearchAgent:
    def heuristic(self, state: RecoveryState) -> int:
        # TODO: Return a nonnegative, nontrivial admissible estimate.
        raise NotImplementedError("Implement AStarSearchAgent.heuristic")

    def plan(self, problem: RecoveryProblem) -> SearchResult:
        # TODO: Implement A* graph search and metric accounting.
        raise NotImplementedError("Implement AStarSearchAgent.plan")
