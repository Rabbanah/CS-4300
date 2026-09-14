"""Provided Cloud Service Recovery environment for U0-HW-03.

Students should not modify this file.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum


class StorageStatus(Enum):
    DOWN = "down"
    HEALTHY = "healthy"


class DatabaseStatus(Enum):
    DOWN = "down"
    CORRUPT = "corrupt"
    HEALTHY = "healthy"


class ServiceStatus(Enum):
    DOWN = "down"
    FAULTY = "faulty"
    HEALTHY = "healthy"


class TrafficMode(Enum):
    NORMAL = "normal"
    ISOLATED = "isolated"
    CANARY = "canary"


class BackupStatus(Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class ValidationStatus(Enum):
    NOT_RUN = "not_run"
    PASSED = "passed"


class Action(Enum):
    ISOLATE_TRAFFIC = "isolate_traffic"
    RESTART_STORAGE = "restart_storage"
    RESTART_DATABASE = "restart_database"
    RESTORE_DATABASE = "restore_database"
    RESTART_AUTH = "restart_auth"
    ROLLBACK_AUTH = "rollback_auth"
    RESTART_API = "restart_api"
    ROLLBACK_API = "rollback_api"
    VALIDATE_STACK = "validate_stack"
    ENABLE_CANARY = "enable_canary"
    ENABLE_NORMAL = "enable_normal"


ACTION_COSTS: dict[Action, int] = {
    Action.ISOLATE_TRAFFIC: 1,
    Action.RESTART_STORAGE: 4,
    Action.RESTART_DATABASE: 4,
    Action.RESTORE_DATABASE: 7,
    Action.RESTART_AUTH: 3,
    Action.ROLLBACK_AUTH: 5,
    Action.RESTART_API: 3,
    Action.ROLLBACK_API: 5,
    Action.VALIDATE_STACK: 2,
    Action.ENABLE_CANARY: 2,
    Action.ENABLE_NORMAL: 2,
}

ACTION_ORDER = tuple(Action)


@dataclass(frozen=True)
class RecoveryState:
    storage: StorageStatus
    database: DatabaseStatus
    auth: ServiceStatus
    api: ServiceStatus
    traffic: TrafficMode
    backup: BackupStatus
    validation: ValidationStatus


@dataclass(frozen=True)
class Transition:
    before: RecoveryState
    action: Action
    after: RecoveryState
    cost: int
    recovered: bool
    done: bool


def _services_healthy(state: RecoveryState) -> bool:
    return (
        state.storage is StorageStatus.HEALTHY
        and state.database is DatabaseStatus.HEALTHY
        and state.auth is ServiceStatus.HEALTHY
        and state.api is ServiceStatus.HEALTHY
    )


def _is_goal(state: RecoveryState) -> bool:
    return (
        _services_healthy(state)
        and state.traffic is TrafficMode.NORMAL
        and state.validation is ValidationStatus.PASSED
    )


def _legal_actions(state: RecoveryState) -> tuple[Action, ...]:
    actions: list[Action] = []
    isolated = state.traffic is TrafficMode.ISOLATED

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
        and _services_healthy(state)
        and state.validation is ValidationStatus.NOT_RUN
    ):
        actions.append(Action.VALIDATE_STACK)
    if (
        isolated
        and _services_healthy(state)
        and state.validation is ValidationStatus.PASSED
    ):
        actions.append(Action.ENABLE_CANARY)
    if (
        state.traffic is TrafficMode.CANARY
        and _services_healthy(state)
        and state.validation is ValidationStatus.PASSED
    ):
        actions.append(Action.ENABLE_NORMAL)

    return tuple(action for action in ACTION_ORDER if action in actions)


def _apply_legal_action(state: RecoveryState, action: Action) -> RecoveryState:
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


class ServiceRecoveryEnvironment:
    """Deterministic simulator that independently validates a recovery plan."""

    def __init__(self, initial_state: RecoveryState, max_actions: int = 25) -> None:
        if max_actions <= 0:
            raise ValueError("max_actions must be positive")
        self.initial_state = initial_state
        self.max_actions = max_actions
        self.reset()

    def reset(self) -> RecoveryState:
        self.state = self.initial_state
        self.action_count = 0
        self.total_cost = 0
        self.trace: list[Transition] = []
        return self.state

    @property
    def recovered(self) -> bool:
        return _is_goal(self.state)

    @property
    def done(self) -> bool:
        return self.recovered or self.action_count >= self.max_actions

    def legal_actions(self) -> tuple[Action, ...]:
        return _legal_actions(self.state)

    def step(self, action: Action) -> Transition:
        if self.done:
            raise RuntimeError("Cannot act after an episode has ended")
        if not isinstance(action, Action):
            raise TypeError("action must be an Action value")
        if action not in self.legal_actions():
            raise ValueError(f"{action.name} is illegal in the current state")

        before = self.state
        self.state = _apply_legal_action(before, action)
        self.action_count += 1
        cost = ACTION_COSTS[action]
        self.total_cost += cost
        transition = Transition(
            before=before,
            action=action,
            after=self.state,
            cost=cost,
            recovered=self.recovered,
            done=self.done,
        )
        self.trace.append(transition)
        return transition
