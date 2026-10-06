"""Run the Assignment 6 search matrix and write raw and summary CSV."""

from __future__ import annotations

import argparse
import csv
import math
import multiprocessing
import statistics
from dataclasses import fields
from pathlib import Path
from typing import Any

from alpha_beta_agent import AlphaBetaAgent, SearchMetrics
from alpha_beta_ordering import order_actions_1, order_actions_2
import hw06_config
from hw06_measured_minimax import MeasuredMinimaxAgent
from hw06_positions import POSITIONS

NA = "N/A"
AGENT_NAMES = (
    "measured_minimax", "alpha_beta_order_1", "alpha_beta_order_2"
)
STAGE_AGENTS = {
    "A": AGENT_NAMES,
    "B": ("alpha_beta_order_1", "alpha_beta_order_2"),
}
METRIC_NAMES = tuple(field.name for field in fields(SearchMetrics))
RELATIVE_CHANGE_FIELD = "relative_expanded_nodes_change_order2_vs_order1"
RAW_FIELDS = (
    "stage", "position", "category", "depth", "agent", "repetition",
    "status", "selected_move", "root_value", *METRIC_NAMES,
    "error_type", "error_message",
)
SUMMARY_FIELDS = (
    "stage", "position", "depth", "agent", "attempts",
    "completed_searches", "timeouts", "errors", "selected_move",
    "root_value", "equivalent_move", "equivalent_value", *METRIC_NAMES,
    RELATIVE_CHANGE_FIELD,
)


def _empty_metrics() -> dict[str, object]:
    return {name: NA for name in METRIC_NAMES}


def _base_row(
    stage: str, position: object, depth: int, agent_name: str, repetition: int
) -> dict[str, object]:
    return {
        "stage": stage, "position": position.identifier,
        "category": position.category, "depth": depth, "agent": agent_name,
        "repetition": repetition, "status": "ok", "selected_move": NA,
        "root_value": NA, **_empty_metrics(), "error_type": NA,
        "error_message": NA,
    }


def _build_agent(agent_name: str, depth: int) -> object:
    evaluator = hw06_config.SELECTED_EVALUATOR
    if agent_name == "measured_minimax":
        return MeasuredMinimaxAgent(agent_name, depth, evaluator)
    ordering = (
        order_actions_1 if agent_name == "alpha_beta_order_1" else order_actions_2
    )
    return AlphaBetaAgent(agent_name, depth, evaluator, ordering)


def _search_worker(
    connection: Any, stage: str, position: object, depth: int,
    agent_name: str, repetition: int,
) -> None:
    """Run one search in a child and return a serializable result row."""
    row = _base_row(stage, position, depth, agent_name, repetition)
    try:
        agent = _build_agent(agent_name, depth)
        move = agent.choose_action(position.board)
    except Exception as error:
        row["status"] = "error"
        row["error_type"] = type(error).__name__
        row["error_message"] = str(error)
    else:
        row["selected_move"] = str(move)
        row["root_value"] = agent.last_value
        row.update(vars(agent.last_metrics))
    try:
        connection.send(row)
    finally:
        connection.close()


def _run_search(
    stage: str, position: object, depth: int, agent_name: str,
    repetition: int, timeout: float | None,
) -> dict[str, object]:
    """Run one search in a spawn child, terminating it at the time bound."""
    row = _base_row(stage, position, depth, agent_name, repetition)
    context = multiprocessing.get_context("spawn")
    parent_connection, child_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_search_worker,
        args=(child_connection, stage, position, depth, agent_name, repetition),
    )
    try:
        process.start()
        child_connection.close()
        process.join(timeout)
        if process.is_alive():
            process.terminate()
            process.join(1.0)
            if process.is_alive():
                process.kill()
                process.join()
            row["status"] = "timeout"
            row["error_type"] = "SearchTimeout"
            row["error_message"] = f"search exceeded {timeout:g} seconds"
        elif parent_connection.poll():
            row = parent_connection.recv()
        else:
            row["status"] = "error"
            row["error_type"] = "ChildProcessError"
            row["error_message"] = (
                f"search process exited with code {process.exitcode} without a result"
            )
    except Exception as error:
        if process.is_alive():
            process.terminate()
            process.join(1.0)
            if process.is_alive():
                process.kill()
                process.join()
        row["status"] = "error"
        row["error_type"] = type(error).__name__
        row["error_message"] = str(error)
    finally:
        parent_connection.close()
        child_connection.close()
    return row


def run_matrix(
    repeats: int = 3,
    timeout: float | None = 30.0,
    stage: str = "all",
    position_identifier: str | None = None,
    agent_name: str | None = None,
) -> list[dict[str, object]]:
    if repeats < 1:
        raise ValueError("repeats must be at least one")
    normalized_stage = stage.upper() if stage.lower() != "all" else "all"
    if normalized_stage not in ("A", "B", "all"):
        raise ValueError("stage must be A, B, or all")
    if agent_name is not None and agent_name not in AGENT_NAMES:
        raise ValueError(f"unknown agent: {agent_name}")
    if normalized_stage == "B" and agent_name == "measured_minimax":
        raise ValueError("measured_minimax is available only in Stage A")

    positions = [
        position for position in POSITIONS
        if position_identifier is None or position.identifier == position_identifier
    ]
    if not positions:
        raise ValueError(f"unknown position: {position_identifier}")

    stages = ("A", "B") if normalized_stage == "all" else (normalized_stage,)
    rows: list[dict[str, object]] = []
    for position in positions:
        for selected_stage in stages:
            depth = 2 if selected_stage == "A" else 3
            repetitions = 1 if selected_stage == "A" else repeats
            for selected_agent in STAGE_AGENTS[selected_stage]:
                if agent_name is not None and selected_agent != agent_name:
                    continue
                for repetition in range(1, repetitions + 1):
                    rows.append(_run_search(
                        selected_stage, position, depth, selected_agent,
                        repetition, timeout,
                    ))
    return rows


def _stable_completed_value(
    rows: list[dict[str, object]], name: str
) -> object:
    values = {row[name] for row in rows if row["status"] == "ok"}
    return values.pop() if len(values) == 1 else NA


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in rows:
        key = (row["stage"], row["position"], row["depth"], row["agent"])
        groups.setdefault(key, []).append(row)

    stage_a_equivalence: dict[str, tuple[object, object]] = {}
    for position in {str(row["position"]) for row in rows if row["stage"] == "A"}:
        comparison = [
            row for row in rows
            if row["stage"] == "A" and row["position"] == position
        ]
        complete = (
            {row["agent"] for row in comparison} == set(STAGE_AGENTS["A"])
            and len(comparison) == 3
            and all(row["status"] == "ok" for row in comparison)
        )
        if complete:
            moves = [row["selected_move"] for row in comparison]
            values = [float(row["root_value"]) for row in comparison]
            stage_a_equivalence[position] = (
                len(set(moves)) == 1,
                all(math.isclose(values[0], value) for value in values[1:]),
            )
        else:
            stage_a_equivalence[position] = (NA, NA)

    relative_changes: dict[str, object] = {}
    for position in {str(row["position"]) for row in rows if row["stage"] == "B"}:
        order_1 = groups.get(("B", position, 3, "alpha_beta_order_1"), [])
        order_2 = groups.get(("B", position, 3, "alpha_beta_order_2"), [])
        count_1 = _stable_completed_value(order_1, "expanded_nodes")
        count_2 = _stable_completed_value(order_2, "expanded_nodes")
        complete = (
            bool(order_1) and bool(order_2)
            and all(row["status"] == "ok" for row in order_1 + order_2)
        )
        relative_changes[position] = (
            (float(count_1) - float(count_2)) / float(count_1)
            if (
                complete and count_1 != NA and count_2 != NA
                and float(count_1) != 0
            )
            else NA
        )

    summary: list[dict[str, object]] = []
    for (stage, position, depth, agent), group in groups.items():
        completed = [row for row in group if row["status"] == "ok"]
        metric_summary = {
            name: _stable_completed_value(group, name)
            for name in METRIC_NAMES if name != "thinking_time"
        }
        metric_summary["thinking_time"] = (
            statistics.median(float(row["thinking_time"]) for row in completed)
            if completed else NA
        )
        equivalent_move, equivalent_value = (
            stage_a_equivalence[str(position)] if stage == "A" else (NA, NA)
        )
        summary.append({
            "stage": stage, "position": position, "depth": depth,
            "agent": agent, "attempts": len(group),
            "completed_searches": len(completed),
            "timeouts": sum(row["status"] == "timeout" for row in group),
            "errors": sum(row["status"] == "error" for row in group),
            "selected_move": _stable_completed_value(group, "selected_move"),
            "root_value": _stable_completed_value(group, "root_value"),
            "equivalent_move": equivalent_move,
            "equivalent_value": equivalent_value,
            **metric_summary,
            RELATIVE_CHANGE_FIELD: (
                relative_changes.get(str(position), NA) if stage == "B" else NA
            ),
        })
    return summary


def _write_csv(
    path: Path, rows: list[dict[str, object]], fieldnames: tuple[str, ...]
) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("hw06_results"))
    parser.add_argument("--stage", choices=("A", "B", "all"), default="all")
    parser.add_argument(
        "--position", choices=tuple(position.identifier for position in POSITIONS)
    )
    parser.add_argument("--agent", choices=AGENT_NAMES)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be at least one")
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.stage == "B" and args.agent == "measured_minimax":
        parser.error("--agent measured_minimax is incompatible with --stage B")
    args.output.mkdir(parents=True, exist_ok=True)
    rows = run_matrix(
        args.repeats, args.timeout, args.stage, args.position, args.agent
    )
    summary = summarize(rows)
    _write_csv(args.output / "raw.csv", rows, RAW_FIELDS)
    _write_csv(args.output / "summary.csv", summary, SUMMARY_FIELDS)
    print(f"wrote {len(rows)} attempts to {args.output / 'raw.csv'}")
    print(f"wrote {len(summary)} summaries to {args.output / 'summary.csv'}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
