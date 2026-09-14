"""Student-completed search problem formulation for U0-HW-03."""

from __future__ import annotations

from recovery import Action, RecoveryState


class RecoveryProblem:
    def __init__(self, initial_state: RecoveryState) -> None:
        self.initial_state = initial_state

    def actions(self, state: RecoveryState) -> tuple[Action, ...]:
        """Return all legal actions in the supplied Action enumeration order."""
        raise NotImplementedError("Implement RecoveryProblem.actions")

    def result(self, state: RecoveryState, action: Action) -> RecoveryState:
        """Return the new immutable state, or raise ValueError if action is illegal."""
        raise NotImplementedError("Implement RecoveryProblem.result")

    def is_goal(self, state: RecoveryState) -> bool:
        """Return whether state satisfies the complete recovery goal."""
        raise NotImplementedError("Implement RecoveryProblem.is_goal")

    def step_cost(
        self, state: RecoveryState, action: Action, next_state: RecoveryState
    ) -> int:
        """Return action cost, or raise ValueError if the transition is illegal."""
        raise NotImplementedError("Implement RecoveryProblem.step_cost")
