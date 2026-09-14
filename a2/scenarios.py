"""Provided deterministic scenario suite for U0-HW-03.

Students should not modify this file.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from recovery import (
    BackupStatus,
    DatabaseStatus,
    RecoveryState,
    ServiceStatus,
    StorageStatus,
    TrafficMode,
    ValidationStatus,
)


@dataclass(frozen=True)
class Scenario:
    identifier: int
    initial_state: RecoveryState


_ANCHOR_STATES = (
    RecoveryState(
        StorageStatus.HEALTHY,
        DatabaseStatus.HEALTHY,
        ServiceStatus.HEALTHY,
        ServiceStatus.HEALTHY,
        TrafficMode.NORMAL,
        BackupStatus.AVAILABLE,
        ValidationStatus.PASSED,
    ),
    RecoveryState(
        StorageStatus.HEALTHY,
        DatabaseStatus.DOWN,
        ServiceStatus.HEALTHY,
        ServiceStatus.FAULTY,
        TrafficMode.NORMAL,
        BackupStatus.UNAVAILABLE,
        ValidationStatus.NOT_RUN,
    ),
    RecoveryState(
        StorageStatus.DOWN,
        DatabaseStatus.DOWN,
        ServiceStatus.DOWN,
        ServiceStatus.DOWN,
        TrafficMode.NORMAL,
        BackupStatus.AVAILABLE,
        ValidationStatus.NOT_RUN,
    ),
    RecoveryState(
        StorageStatus.DOWN,
        DatabaseStatus.CORRUPT,
        ServiceStatus.FAULTY,
        ServiceStatus.FAULTY,
        TrafficMode.CANARY,
        BackupStatus.AVAILABLE,
        ValidationStatus.NOT_RUN,
    ),
    RecoveryState(
        StorageStatus.HEALTHY,
        DatabaseStatus.HEALTHY,
        ServiceStatus.HEALTHY,
        ServiceStatus.HEALTHY,
        TrafficMode.ISOLATED,
        BackupStatus.UNAVAILABLE,
        ValidationStatus.NOT_RUN,
    ),
    RecoveryState(
        StorageStatus.HEALTHY,
        DatabaseStatus.HEALTHY,
        ServiceStatus.HEALTHY,
        ServiceStatus.HEALTHY,
        TrafficMode.ISOLATED,
        BackupStatus.AVAILABLE,
        ValidationStatus.PASSED,
    ),
    RecoveryState(
        StorageStatus.HEALTHY,
        DatabaseStatus.HEALTHY,
        ServiceStatus.HEALTHY,
        ServiceStatus.HEALTHY,
        TrafficMode.CANARY,
        BackupStatus.AVAILABLE,
        ValidationStatus.PASSED,
    ),
)


def _build_states() -> tuple[RecoveryState, ...]:
    states = list(_ANCHOR_STATES)
    seen = set(states)
    rng = random.Random(4300)

    while len(states) < 50:
        storage = rng.choice(tuple(StorageStatus))
        database = rng.choice(tuple(DatabaseStatus))
        auth = rng.choice(tuple(ServiceStatus))
        api = rng.choice(tuple(ServiceStatus))
        traffic = rng.choice(tuple(TrafficMode))
        backup = rng.choice(tuple(BackupStatus))

        if database is DatabaseStatus.CORRUPT:
            backup = BackupStatus.AVAILABLE

        all_healthy = (
            storage is StorageStatus.HEALTHY
            and database is DatabaseStatus.HEALTHY
            and auth is ServiceStatus.HEALTHY
            and api is ServiceStatus.HEALTHY
        )
        validation = (
            rng.choice(tuple(ValidationStatus))
            if all_healthy
            else ValidationStatus.NOT_RUN
        )
        state = RecoveryState(
            storage, database, auth, api, traffic, backup, validation
        )
        if state not in seen:
            seen.add(state)
            states.append(state)

    return tuple(states)


_EVALUATION_STATES = _build_states()


def get_scenario(identifier: int) -> Scenario:
    if not 0 <= identifier < len(_EVALUATION_STATES):
        raise ValueError("scenario identifier must be between 0 and 49")
    return Scenario(identifier, _EVALUATION_STATES[identifier])


def evaluation_scenarios(start: int = 0, stop: int = 50) -> tuple[Scenario, ...]:
    if not 0 <= start <= stop <= len(_EVALUATION_STATES):
        raise ValueError("scenario range must satisfy 0 <= start <= stop <= 50")
    return tuple(get_scenario(identifier) for identifier in range(start, stop))
