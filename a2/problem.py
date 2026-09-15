"""Student-completed search problem formulation for U0-HW-03."""

from __future__ import annotations

from dataclasses import replace
from recovery import (
    ACTION_COSTS,
    ACTION_ORDER,
    Action,
    BackupStatus,
    DatabaseStatus,
    RecoveryState,
    ServiceStatus,
    StorageStatus,
    TrafficMode,
    ValidationStatus,
)

class RecoveryProblem:
    def __init__(self, initial_state: RecoveryState) -> None:
        self.initial_state = initial_state

    def actions(self, state: RecoveryState) -> tuple[Action, ...]:
        """Return all legal actions in the supplied Action enumeration order."""
        actions: list[Action] = []
        isolated = state.traffic is TrafficMode.ISOLATED
        services_healthy = (
            state.storage is StorageStatus.HEALTHY
            and state.database is DatabaseStatus.HEALTHY
            and state.auth is ServiceStatus.HEALTHY
            and state.api is ServiceStatus.HEALTHY
        )

        if not isolated:
            actions.append(Action.ISOLATE_TRAFFIC)
        if isolated and state.storage is StorageStatus.DOWN:
            actions.append(Action.RESTART_STORAGE)
        if (
            isolated
            and state.storage is StorageStatus.HEALTHY
            and state.database is DatabaseStatus.DOWN
        ):
            actions.append(Action.RESTART_DATABASE)
        if (
            isolated
            and state.storage is StorageStatus.HEALTHY
            and state.database is DatabaseStatus.CORRUPT
            and state.backup is BackupStatus.AVAILABLE
        ):
            actions.append(Action.RESTORE_DATABASE)
        if (
            isolated
            and state.database is DatabaseStatus.HEALTHY
            and state.auth is not ServiceStatus.HEALTHY
        ):
            actions.append(Action.RESTART_AUTH)
        if (
            isolated
            and state.database is DatabaseStatus.HEALTHY
            and state.auth is ServiceStatus.FAULTY
        ):
            actions.append(Action.ROLLBACK_AUTH)
        if (
            isolated
            and state.database is DatabaseStatus.HEALTHY
            and state.auth is ServiceStatus.HEALTHY
            and state.api is not ServiceStatus.HEALTHY
        ):
            actions.append(Action.RESTART_API)
        if (
            isolated
            and state.database is DatabaseStatus.HEALTHY
            and state.auth is ServiceStatus.HEALTHY
            and state.api is ServiceStatus.FAULTY
        ):
            actions.append(Action.ROLLBACK_API)
        if (
            isolated
            and services_healthy
            and state.validation is ValidationStatus.NOT_RUN
        ):
            actions.append(Action.VALIDATE_STACK)
        if (
            isolated
            and services_healthy
            and state.validation is ValidationStatus.PASSED
        ):
            actions.append(Action.ENABLE_CANARY)
        if (
            state.traffic is TrafficMode.CANARY
            and services_healthy
            and state.validation is ValidationStatus.PASSED
        ):
            actions.append(Action.ENABLE_NORMAL)

        return tuple(action for action in ACTION_ORDER if action in actions)

    def result(self, state: RecoveryState, action: Action) -> RecoveryState:
        """Return the new immutable state, or raise ValueError if action is illegal."""
        if action not in self.actions(state):
            raise ValueError(f"Action {action} is illegal in state {state}")

        not_validated = ValidationStatus.NOT_RUN

        if action is Action.ISOLATE_TRAFFIC:
            return replace(state, traffic=TrafficMode.ISOLATED)
        if action is Action.RESTART_STORAGE:
            return replace(
                state, storage=StorageStatus.HEALTHY, validation=not_validated
            )
        if action is Action.RESTART_DATABASE:
            return replace(
                state, database=DatabaseStatus.HEALTHY, validation=not_validated
            )
        if action is Action.RESTORE_DATABASE:
            return replace(
                state,
                database=DatabaseStatus.HEALTHY,
                backup=BackupStatus.UNAVAILABLE,
                validation=not_validated,
            )
        if action is Action.RESTART_AUTH:
            next_status = (
                ServiceStatus.DOWN
                if state.auth is ServiceStatus.FAULTY
                else ServiceStatus.HEALTHY
            )
            return replace(state, auth=next_status, validation=not_validated)
        if action is Action.ROLLBACK_AUTH:
            return replace(state, auth=ServiceStatus.HEALTHY, validation=not_validated)
        if action is Action.RESTART_API:
            next_status = (
                ServiceStatus.DOWN
                if state.api is ServiceStatus.FAULTY
                else ServiceStatus.HEALTHY
            )
            return replace(state, api=next_status, validation=not_validated)
        if action is Action.ROLLBACK_API:
            return replace(state, api=ServiceStatus.HEALTHY, validation=not_validated)
        if action is Action.VALIDATE_STACK:
            return replace(state, validation=ValidationStatus.PASSED)
        if action is Action.ENABLE_CANARY:
            return replace(state, traffic=TrafficMode.CANARY)
        if action is Action.ENABLE_NORMAL:
            return replace(state, traffic=TrafficMode.NORMAL)

        raise AssertionError(f"Unhandled action: {action}")

    def is_goal(self, state: RecoveryState) -> bool:
        """Return whether state satisfies the complete recovery goal."""
        return (
            state.storage is StorageStatus.HEALTHY
            and state.database is DatabaseStatus.HEALTHY
            and state.auth is ServiceStatus.HEALTHY
            and state.api is ServiceStatus.HEALTHY
            and state.traffic is TrafficMode.NORMAL
            and state.validation is ValidationStatus.PASSED
        )

    def step_cost(
        self, state: RecoveryState, action: Action, next_state: RecoveryState
    ) -> int:
        """Return action cost, or raise ValueError if the transition is illegal."""
        if action not in self.actions(state):
            raise ValueError(f"Action {action} is illegal in state {state}")
        expected_next = self.result(state, action)
        if next_state != expected_next:
            raise ValueError("next_state does not match result(state, action)")
        return ACTION_COSTS[action]