"""Deterministic supplied scenarios for U0-HW-04."""

from __future__ import annotations

from dataclasses import dataclass

from conference import ConferenceProblem, Placement, Room, Schedule, Talk, make_schedule


@dataclass(frozen=True)
class Scenario:
    identifier: str
    problem: ConferenceProblem
    initial_schedule: Schedule
    budget: int = 20


def _talk(
    identifier: str,
    speaker: str,
    track: str,
    audience: int,
    available: tuple[str, ...],
    preferred: tuple[str, ...],
    equipment: tuple[str, ...] = (),
) -> Talk:
    return Talk(
        identifier,
        f"Talk {identifier}",
        speaker,
        track,
        audience,
        frozenset(equipment),
        frozenset(available),
        frozenset(preferred),
    )


def evaluation_scenarios() -> tuple[Scenario, ...]:
    rooms = (
        Room("Auditorium", 180, frozenset({"projector", "recording"})),
        Room("Canyon", 80, frozenset({"projector"})),
        Room("Mesa", 45, frozenset({"projector", "lab"})),
    )
    slots = ("09:00", "10:30", "13:00")
    all_slots = slots
    talks = (
        _talk("T1", "Ada", "AI", 140, all_slots, ("09:00",), ("recording",)),
        _talk("T2", "Bo", "Systems", 70, all_slots, ("10:30",)),
        _talk("T3", "Ada", "AI", 40, all_slots, ("10:30",), ("lab",)),
        _talk("T4", "Cy", "Systems", 35, ("09:00", "13:00"), ("13:00",)),
        _talk("T5", "Dee", "AI", 65, all_slots, ("13:00",)),
        _talk("T6", "Eli", "Practice", 30, ("10:30", "13:00"), ("10:30",)),
    )
    problem = ConferenceProblem(
        rooms,
        slots,
        talks,
        frozenset({frozenset({"T1", "T2"}), frozenset({"T3", "T5"})}),
    )
    starts = (
        {
            "T1": Placement("Auditorium", "09:00"),
            "T2": Placement("Canyon", "09:00"),
            "T3": Placement("Mesa", "09:00"),
            "T4": Placement("Canyon", "09:00"),
            "T5": Placement("Auditorium", "09:00"),
            "T6": Placement("Mesa", "10:30"),
        },
        {
            "T1": Placement("Auditorium", "10:30"),
            "T2": Placement("Canyon", "10:30"),
            "T3": Placement("Mesa", "10:30"),
            "T4": Placement("Canyon", "13:00"),
            "T5": Placement("Canyon", "10:30"),
            "T6": Placement("Mesa", "10:30"),
        },
        {
            "T1": Placement("Auditorium", "13:00"),
            "T2": Placement("Auditorium", "13:00"),
            "T3": Placement("Mesa", "13:00"),
            "T4": Placement("Mesa", "13:00"),
            "T5": Placement("Canyon", "13:00"),
            "T6": Placement("Canyon", "13:00"),
        },
        {
            "T1": Placement("Auditorium", "09:00"),
            "T2": Placement("Canyon", "10:30"),
            "T3": Placement("Mesa", "09:00"),
            "T4": Placement("Canyon", "09:00"),
            "T5": Placement("Auditorium", "13:00"),
            "T6": Placement("Mesa", "13:00"),
        },
    )
    return tuple(
        Scenario(f"conference-{index + 1}", problem, make_schedule(problem, start))
        for index, start in enumerate(starts)
    )
