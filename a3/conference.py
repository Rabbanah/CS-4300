"""Immutable conference-scheduling model supplied for U0-HW-04."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping


@dataclass(frozen=True, order=True)
class Placement:
    room: str
    slot: str


@dataclass(frozen=True)
class Room:
    name: str
    capacity: int
    equipment: frozenset[str] = frozenset()


@dataclass(frozen=True)
class Talk:
    identifier: str
    title: str
    speaker: str
    track: str
    expected_audience: int
    required_equipment: frozenset[str]
    available_slots: frozenset[str]
    preferred_slots: frozenset[str] = frozenset()


@dataclass(frozen=True)
class ConferenceProblem:
    rooms: tuple[Room, ...]
    slots: tuple[str, ...]
    talks: tuple[Talk, ...]
    audience_conflicts: frozenset[frozenset[str]]

    def __post_init__(self) -> None:
        room_names = [room.name for room in self.rooms]
        talk_ids = [talk.identifier for talk in self.talks]
        if len(room_names) != len(set(room_names)):
            raise ValueError("room names must be unique")
        if len(self.slots) != len(set(self.slots)):
            raise ValueError("slots must be unique")
        if len(talk_ids) != len(set(talk_ids)):
            raise ValueError("talk identifiers must be unique")
        if not self.rooms or not self.slots or not self.talks:
            raise ValueError("a problem needs rooms, slots, and talks")
        known = set(talk_ids)
        for pair in self.audience_conflicts:
            if len(pair) != 2 or not pair <= known:
                raise ValueError("audience conflicts must be pairs of known talks")
        for talk in self.talks:
            if not self.domain(talk.identifier):
                raise ValueError(f"talk {talk.identifier} has an empty domain")

    def talk(self, identifier: str) -> Talk:
        for talk in self.talks:
            if talk.identifier == identifier:
                return talk
        raise KeyError(identifier)

    def domain(self, talk_id: str) -> tuple[Placement, ...]:
        """Return values satisfying capacity, equipment, and availability."""
        talk = self.talk(talk_id)
        return tuple(
            Placement(room.name, slot)
            for slot in self.slots
            if slot in talk.available_slots
            for room in self.rooms
            if room.capacity >= talk.expected_audience
            and talk.required_equipment <= room.equipment
        )


@dataclass(frozen=True)
class Schedule:
    """A complete talk-to-placement mapping in canonical talk order."""

    assignments: tuple[tuple[str, Placement], ...]

    def __post_init__(self) -> None:
        identifiers = [identifier for identifier, _ in self.assignments]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("a schedule cannot assign a talk more than once")

    def placement(self, talk_id: str) -> Placement:
        for identifier, placement in self.assignments:
            if identifier == talk_id:
                return placement
        raise KeyError(talk_id)

    def as_mapping(self) -> Mapping[str, Placement]:
        return MappingProxyType(dict(self.assignments))


def make_schedule(
    problem: ConferenceProblem, assignments: Mapping[str, Placement]
) -> Schedule:
    """Construct a complete, structurally valid schedule."""
    expected = {talk.identifier for talk in problem.talks}
    if set(assignments) != expected:
        missing = expected - set(assignments)
        extra = set(assignments) - expected
        raise ValueError(f"schedule talk mismatch: missing={missing}, extra={extra}")
    for talk_id, placement in assignments.items():
        if placement not in problem.domain(talk_id):
            raise ValueError(f"{placement!r} is outside {talk_id}'s domain")
    return Schedule(
        tuple((talk.identifier, assignments[talk.identifier]) for talk in problem.talks)
    )


def is_structurally_valid(problem: ConferenceProblem, schedule: Schedule) -> bool:
    try:
        return make_schedule(problem, schedule.as_mapping()) == schedule
    except (KeyError, ValueError):
        return False


def move(
    problem: ConferenceProblem,
    schedule: Schedule,
    talk_id: str,
    placement: Placement,
) -> Schedule:
    """Return a new complete schedule after one domain-valid talk move."""
    if not is_structurally_valid(problem, schedule):
        raise ValueError("input schedule is not structurally valid")
    if placement not in problem.domain(talk_id):
        raise ValueError(f"{placement!r} is outside {talk_id}'s domain")
    changed = dict(schedule.assignments)
    changed[talk_id] = placement
    return make_schedule(problem, changed)


def format_schedule(schedule: Schedule) -> str:
    return "\n".join(
        f"{talk_id}: {placement.room} @ {placement.slot}"
        for talk_id, placement in schedule.assignments
    )
