"""Provided interfaces and student-completed agents for U0-HW-03."""

from __future__ import annotations

import heapq
import random
import time
from dataclasses import dataclass
from typing import Protocol

from problem import RecoveryProblem
from recovery import (
    Action,
    BackupStatus,
    DatabaseStatus,
    RecoveryState,
    ServiceStatus,
    StorageStatus,
    TrafficMode,
    ValidationStatus,
)


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


def compute_admissible_heuristic(state: RecoveryState) -> int:
    """Compute a provably admissible lower-bound cost to reach the goal."""
    cost = 0
    services_healthy = (
        state.storage is StorageStatus.HEALTHY
        and state.database is DatabaseStatus.HEALTHY
        and state.auth is ServiceStatus.HEALTHY
        and state.api is ServiceStatus.HEALTHY
    )

    if state.storage is StorageStatus.DOWN:
        cost += 4
    if state.database is DatabaseStatus.DOWN:
        cost += 4
    elif state.database is DatabaseStatus.CORRUPT:
        cost += 7
    if state.auth in (ServiceStatus.DOWN, ServiceStatus.FAULTY):
        cost += 3
    if state.api in (ServiceStatus.DOWN, ServiceStatus.FAULTY):
        cost += 3

    if not services_healthy:
        if state.traffic is not TrafficMode.ISOLATED:
            cost += 1  # ISOLATE_TRAFFIC
        cost += 2 + 2 + 2  # VALIDATE_STACK + ENABLE_CANARY + ENABLE_NORMAL
    else:
        if state.validation is ValidationStatus.NOT_RUN:
            if state.traffic is not TrafficMode.ISOLATED:
                cost += 1
            cost += 2 + 2 + 2
        else:
            if state.traffic is TrafficMode.ISOLATED:
                cost += 2 + 2  # ENABLE_CANARY + ENABLE_NORMAL
            elif state.traffic is TrafficMode.CANARY:
                cost += 2  # ENABLE_NORMAL
            elif state.traffic is TrafficMode.NORMAL:
                cost += 0  # Goal

    return cost


class UniformCostSearchAgent:
    def plan(self, problem: RecoveryProblem) -> SearchResult:
        start_time = time.perf_counter()

        if problem.is_goal(problem.initial_state):
            elapsed = time.perf_counter() - start_time
            return SearchResult(
                plan=(),
                metrics=SearchMetrics(
                    generated=1,
                    expanded=0,
                    peak_frontier=1,
                    planning_time_seconds=elapsed,
                ),
            )

        counter = 0
        frontier: list[tuple[int, int, RecoveryState, tuple[Action, ...]]] = []
        heapq.heappush(frontier, (0, counter, problem.initial_state, ()))

        best_g: dict[RecoveryState, int] = {problem.initial_state: 0}
        generated = 1
        expanded = 0
        peak_frontier = 1

        while frontier:
            g, _, current_state, plan = heapq.heappop(frontier)

            if g > best_g.get(current_state, float("inf")):
                continue

            if problem.is_goal(current_state):
                elapsed = time.perf_counter() - start_time
                return SearchResult(
                    plan=plan,
                    metrics=SearchMetrics(
                        generated=generated,
                        expanded=expanded,
                        peak_frontier=peak_frontier,
                        planning_time_seconds=elapsed,
                    ),
                )

            expanded += 1

            for action in problem.actions(current_state):
                next_state = problem.result(current_state, action)
                step_cost = problem.step_cost(current_state, action, next_state)
                new_g = g + step_cost

                if new_g < best_g.get(next_state, float("inf")):
                    best_g[next_state] = new_g
                    counter += 1
                    heapq.heappush(frontier, (new_g, counter, next_state, plan + (action,)))
                    generated += 1
                    if len(frontier) > peak_frontier:
                        peak_frontier = len(frontier)

        elapsed = time.perf_counter() - start_time
        return SearchResult(
            plan=None,
            metrics=SearchMetrics(
                generated=generated,
                expanded=expanded,
                peak_frontier=peak_frontier,
                planning_time_seconds=elapsed,
            ),
        )

class AStarSearchAgent:
    def heuristic(self, state: RecoveryState) -> int:
        return compute_admissible_heuristic(state)

    def plan(self, problem: RecoveryProblem) -> SearchResult:
        start_time = time.perf_counter()

        if problem.is_goal(problem.initial_state):
            elapsed = time.perf_counter() - start_time
            return SearchResult(
                plan=(),
                metrics=SearchMetrics(
                    generated=1,
                    expanded=0,
                    peak_frontier=1,
                    planning_time_seconds=elapsed,
                ),
            )

        counter = 0
        h_init = self.heuristic(problem.initial_state)
        frontier: list[tuple[int, int, int, RecoveryState, tuple[Action, ...]]] = []
        heapq.heappush(frontier, (h_init, 0, counter, problem.initial_state, ()))

        best_g: dict[RecoveryState, int] = {problem.initial_state: 0}
        generated = 1
        expanded = 0
        peak_frontier = 1

        while frontier:
            f, g, _, current_state, plan = heapq.heappop(frontier)

            if g > best_g.get(current_state, float("inf")):
                continue

            if problem.is_goal(current_state):
                elapsed = time.perf_counter() - start_time
                return SearchResult(
                    plan=plan,
                    metrics=SearchMetrics(
                        generated=generated,
                        expanded=expanded,
                        peak_frontier=peak_frontier,
                        planning_time_seconds=elapsed,
                    ),
                )

            expanded += 1

            for action in problem.actions(current_state):
                next_state = problem.result(current_state, action)
                step_cost = problem.step_cost(current_state, action, next_state)
                new_g = g + step_cost

                if new_g < best_g.get(next_state, float("inf")):
                    best_g[next_state] = new_g
                    h = self.heuristic(next_state)
                    new_f = new_g + h
                    counter += 1
                    heapq.heappush(frontier, (new_f, new_g, counter, next_state, plan + (action,)))
                    generated += 1
                    if len(frontier) > peak_frontier:
                        peak_frontier = len(frontier)

        elapsed = time.perf_counter() - start_time
        return SearchResult(
            plan=None,
            metrics=SearchMetrics(
                generated=generated,
                expanded=expanded,
                peak_frontier=peak_frontier,
                planning_time_seconds=elapsed,
            ),
        )