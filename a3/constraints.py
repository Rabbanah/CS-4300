"""Student-completed schedule evaluator for U0-HW-04."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from conference import ConferenceProblem, Schedule


@dataclass(frozen=True)
class Evaluation:
    hard_violations: int
    soft_penalty: int
    conflicted_talks: frozenset[str]
    details: tuple[str, ...]


def evaluate(problem: ConferenceProblem, schedule: Schedule) -> Evaluation:
    """Count hard violations and the precisely defined soft penalty.

    Hard violations are counted once per violated room-collision constraint,
    speaker-overlap constraint, and supplied audience-conflict constraint.
    Soft penalty is one per missed nonempty preferred-slot set plus, for each
    track, max(0, number of rooms used by that track - 1).
    """
    mapping = schedule.as_mapping()
    talk_by_id = {talk.identifier: talk for talk in problem.talks}
    talk_ids = tuple(talk.identifier for talk in problem.talks)

    conflicted: set[str] = set()
    details: list[str] = []
    hard_violations = 0

    n = len(talk_ids)
    for i in range(n):
        t1_id = talk_ids[i]
        p1 = mapping[t1_id]
        t1 = talk_by_id[t1_id]

        for j in range(i + 1, n):
            t2_id = talk_ids[j]
            p2 = mapping[t2_id]
            t2 = talk_by_id[t2_id]

            if p1.room == p2.room and p1.slot == p2.slot:
                hard_violations += 1
                conflicted.add(t1_id)
                conflicted.add(t2_id)
                details.append(f"Room collision: {t1_id} and {t2_id} in {p1.room} @ {p1.slot}")

            if t1.speaker == t2.speaker and p1.slot == p2.slot:
                hard_violations += 1
                conflicted.add(t1_id)
                conflicted.add(t2_id)
                details.append(f"Speaker overlap: {t1_id} and {t2_id} ({t1.speaker}) @ {p1.slot}")

            if p1.slot == p2.slot and frozenset({t1_id, t2_id}) in problem.audience_conflicts:
                hard_violations += 1
                conflicted.add(t1_id)
                conflicted.add(t2_id)
                details.append(f"Audience conflict: {t1_id} and {t2_id} @ {p1.slot}")

    soft_penalty = 0

    for talk in problem.talks:
        if talk.preferred_slots:
            assigned_slot = mapping[talk.identifier].slot
            if assigned_slot not in talk.preferred_slots:
                soft_penalty += 1

    track_rooms: dict[str, set[str]] = defaultdict(set)
    for talk in problem.talks:
        track_rooms[talk.track].add(mapping[talk.identifier].room)

    for track, rooms in track_rooms.items():
        soft_penalty += max(0, len(rooms) - 1)

    return Evaluation(
        hard_violations=hard_violations,
        soft_penalty=soft_penalty,
        conflicted_talks=frozenset(conflicted),
        details=tuple(details),
    )


def is_feasible(problem: ConferenceProblem, schedule: Schedule) -> bool:
    return evaluate(problem, schedule).hard_violations == 0