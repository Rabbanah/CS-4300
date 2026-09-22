"""Repair-agent interfaces and student implementation points."""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass

from conference import ConferenceProblem, Placement, Schedule, move
from constraints import evaluate
from llm_client import ChatClient


@dataclass(frozen=True)
class RepairMetrics:
    edits: int = 0
    evaluator_calls: int = 0
    model_calls: int = 0
    malformed_requests: int = 0
    invalid_requests: int = 0
    repeated_requests: int = 0
    client_failures: int = 0
    runtime_seconds: float = 0.0


@dataclass(frozen=True)
class TraceStep:
    step: int
    request: str
    outcome: str
    hard_violations: int
    soft_penalty: int


@dataclass(frozen=True)
class RepairResult:
    schedule: Schedule
    feasible: bool
    metrics: RepairMetrics
    trace: tuple[TraceStep, ...]


class RandomMoveAgent:
    """Provided seeded baseline using the same one-talk move and budget."""

    def repair(
        self, problem: ConferenceProblem, initial: Schedule, budget: int, seed: int
    ) -> RepairResult:
        started = time.perf_counter()
        rng = random.Random(seed)
        schedule = initial
        calls = 1
        trace: list[TraceStep] = []
        current = evaluate(problem, schedule)
        for step in range(1, budget + 1):
            if current.hard_violations == 0:
                break
            talk_id = rng.choice(tuple(t.identifier for t in problem.talks))
            options = [p for p in problem.domain(talk_id) if p != schedule.placement(talk_id)]
            if not options:
                continue
            placement = rng.choice(options)
            schedule = move(problem, schedule, talk_id, placement)
            current = evaluate(problem, schedule)
            calls += 1
            trace.append(
                TraceStep(
                    step,
                    f"{talk_id}->{placement}",
                    "applied",
                    current.hard_violations,
                    current.soft_penalty,
                )
            )
        return RepairResult(
            schedule,
            current.hard_violations == 0,
            RepairMetrics(
                edits=len(trace),
                evaluator_calls=calls,
                runtime_seconds=time.perf_counter() - started,
            ),
            tuple(trace),
        )


class MinConflictsAgent:
    def repair(
        self, problem: ConferenceProblem, initial: Schedule, budget: int, seed: int
    ) -> RepairResult:
        """Repair with conflicted-variable selection and lexicographic scoring."""
        started = time.perf_counter()
        rng = random.Random(seed)
        schedule = initial
        trace: list[TraceStep] = []

        current = evaluate(problem, schedule)
        evaluator_calls = 1

        for step in range(1, budget + 1):
            if current.hard_violations == 0:
                break

            conflicted_talks = sorted(current.conflicted_talks)
            if not conflicted_talks:
                break
            talk_id = rng.choice(conflicted_talks)

            current_placement = schedule.placement(talk_id)
            domain = problem.domain(talk_id)

            scored_placements: list[tuple[tuple[int, int], Placement]] = []
            for placement in domain:
                if placement == current_placement:
                    score = (current.hard_violations, current.soft_penalty)
                else:
                    cand_schedule = move(problem, schedule, talk_id, placement)
                    cand_eval = evaluate(problem, cand_schedule)
                    evaluator_calls += 1
                    score = (cand_eval.hard_violations, cand_eval.soft_penalty)
                scored_placements.append((score, placement))

            min_score = min(s[0] for s in scored_placements)
            best_placements = [p for s, p in scored_placements if s == min_score]

            alternatives = [p for p in best_placements if p != current_placement]
            chosen_placement = rng.choice(alternatives if alternatives else best_placements)

            if chosen_placement != current_placement:
                schedule = move(problem, schedule, talk_id, chosen_placement)
                current = evaluate(problem, schedule)
                evaluator_calls += 1
                outcome = "applied"
            else:
                outcome = "no-op"

            trace.append(
                TraceStep(
                    step=step,
                    request=f"{talk_id}->{chosen_placement}",
                    outcome=outcome,
                    hard_violations=current.hard_violations,
                    soft_penalty=current.soft_penalty,
                )
            )

        return RepairResult(
            schedule=schedule,
            feasible=(current.hard_violations == 0),
            metrics=RepairMetrics(
                edits=sum(1 for t in trace if t.outcome == "applied"),
                evaluator_calls=evaluator_calls,
                runtime_seconds=time.perf_counter() - started,
            ),
            trace=tuple(trace),
        )


def _no_duplicate_keys_decoder(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate key: {key}")
        result[key] = value
    return result


class LLMRepairAgent:
    def __init__(self, client: ChatClient, history_limit: int = 6) -> None:
        if history_limit < 0:
            raise ValueError("history_limit must be nonnegative")
        self.client = client
        self.history_limit = history_limit

    def repair(
        self, problem: ConferenceProblem, initial: Schedule, budget: int, seed: int
    ) -> RepairResult:
        """Request exactly one JSON move per model call and validate it in code."""
        del seed 
        started = time.perf_counter()
        schedule = initial
        trace: list[TraceStep] = []
        seen_moves: set[tuple[str, str, str]] = set()

        model_calls = 0
        malformed_requests = 0
        invalid_requests = 0
        repeated_requests = 0
        client_failures = 0

        current = evaluate(problem, schedule)
        known_talk_ids = {t.identifier for t in problem.talks}

        for step in range(1, budget + 1):
            if current.hard_violations == 0:
                break

            system_msg = (
                "You are an assistant repairing a conference schedule.\n"
                "Output ONLY a single raw JSON object with no explanations or markdown.\n"
                'Exact schema: {"tool":"move","talk_id":"<ID>","room":"<ROOM>","slot":"<SLOT>"}'
            )

            user_lines = [
                "Current Schedule:",
                "\n".join(
                    f"  {tid}: {schedule.placement(tid).room} @ {schedule.placement(tid).slot}"
                    for tid in sorted(known_talk_ids)
                ),
                f"\nHard Violations: {current.hard_violations}",
                f"Conflicted Talks: {', '.join(sorted(current.conflicted_talks)) if current.conflicted_talks else 'None'}",
                "Violation Details:",
                "\n".join(f"  - {d}" for d in current.details) if current.details else "  None",
                "\nLegal Domains for Conflicted Talks:",
            ]
            for tid in sorted(current.conflicted_talks):
                dom = problem.domain(tid)
                dom_str = ", ".join(f"({p.room}, {p.slot})" for p in dom)
                user_lines.append(f"  {tid}: [{dom_str}]")

            if self.history_limit > 0 and trace:
                user_lines.append("\nRecent Outcomes:")
                recent = trace[-self.history_limit:]
                for t in recent:
                    user_lines.append(f"  Step {t.step}: {t.request} -> {t.outcome}")

            user_lines.append("\nPropose the next move:")
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": "\n".join(user_lines)},
            ]

            model_calls += 1
            try:
                raw_response = self.client.chat(messages)
            except Exception as e:
                client_failures += 1
                trace.append(
                    TraceStep(
                        step=step,
                        request="client_call",
                        outcome=f"client failure: {type(e).__name__}",
                        hard_violations=current.hard_violations,
                        soft_penalty=current.soft_penalty,
                    )
                )
                continue

            clean_resp = raw_response.strip()
            if clean_resp.startswith("```"):
                lines = clean_resp.splitlines()
                if lines and lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                clean_resp = "\n".join(lines).strip()

            try:
                parsed = json.loads(clean_resp, object_pairs_hook=_no_duplicate_keys_decoder)
            except Exception:
                malformed_requests += 1
                trace.append(
                    TraceStep(
                        step=step,
                        request=clean_resp[:40],
                        outcome="malformed: invalid json",
                        hard_violations=current.hard_violations,
                        soft_penalty=current.soft_penalty,
                    )
                )
                continue

            if not isinstance(parsed, dict) or set(parsed.keys()) != {"tool", "talk_id", "room", "slot"}:
                malformed_requests += 1
                trace.append(
                    TraceStep(
                        step=step,
                        request=clean_resp[:40],
                        outcome="malformed: schema mismatch",
                        hard_violations=current.hard_violations,
                        soft_penalty=current.soft_penalty,
                    )
                )
                continue

            if parsed["tool"] != "move":
                malformed_requests += 1
                trace.append(
                    TraceStep(
                        step=step,
                        request=clean_resp[:40],
                        outcome="malformed: tool != move",
                        hard_violations=current.hard_violations,
                        soft_penalty=current.soft_penalty,
                    )
                )
                continue

            talk_id = str(parsed["talk_id"])
            room = str(parsed["room"])
            slot = str(parsed["slot"])
            req_str = f"{talk_id}->Placement({room!r}, {slot!r})"

            if talk_id not in known_talk_ids:
                invalid_requests += 1
                trace.append(
                    TraceStep(
                        step=step,
                        request=req_str,
                        outcome="invalid: unknown talk",
                        hard_violations=current.hard_violations,
                        soft_penalty=current.soft_penalty,
                    )
                )
                continue

            placement = Placement(room, slot)
            if placement not in problem.domain(talk_id):
                invalid_requests += 1
                trace.append(
                    TraceStep(
                        step=step,
                        request=req_str,
                        outcome="invalid: outside domain",
                        hard_violations=current.hard_violations,
                        soft_penalty=current.soft_penalty,
                    )
                )
                continue

            move_tuple = (talk_id, room, slot)
            if move_tuple in seen_moves:
                repeated_requests += 1
                trace.append(
                    TraceStep(
                        step=step,
                        request=req_str,
                        outcome="repeated: already proposed",
                        hard_violations=current.hard_violations,
                        soft_penalty=current.soft_penalty,
                    )
                )
                continue

            seen_moves.add(move_tuple)

            schedule = move(problem, schedule, talk_id, placement)
            current = evaluate(problem, schedule)
            trace.append(
                TraceStep(
                    step=step,
                    request=req_str,
                    outcome="applied",
                    hard_violations=current.hard_violations,
                    soft_penalty=current.soft_penalty,
                )
            )

        return RepairResult(
            schedule=schedule,
            feasible=(current.hard_violations == 0),
            metrics=RepairMetrics(
                edits=sum(1 for t in trace if t.outcome == "applied"),
                evaluator_calls=0,
                model_calls=model_calls,
                malformed_requests=malformed_requests,
                invalid_requests=invalid_requests,
                repeated_requests=repeated_requests,
                client_failures=client_failures,
                runtime_seconds=time.perf_counter() - started,
            ),
            trace=tuple(trace),
        )