"""Run the required Jetan experiment matrix and write one CSV row per match."""

from __future__ import annotations

import argparse
import csv
import math
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


COLORS = ("orange", "black")
SEEDS = (0, 1)
DEFAULT_MAX_THINK_SECONDS = 3600.0
METRICS = (
    "generated",
    "evaluated",
    "maximum_depth",
    "model_calls",
    "fallback_calls",
    "cache_hits",
)
FIELDNAMES = (
    "stage",
    "focal_agent",
    "focal_color",
    "orange_agent",
    "black_agent",
    "orange_depth",
    "black_depth",
    "orange_seed",
    "black_seed",
    "max_plies",
    "max_llm_calls",
    "max_think_seconds",
    "live",
    "status",
    "exit_status",
    "error",
    "result",
    "reason",
    "plies",
    "orange_utility",
    "black_utility",
    "orange_time",
    "black_time",
    *(f"{color}_{metric}" for color in COLORS for metric in METRICS),
    "stdout",
    "stderr",
)
CONFIGURATION_FIELDS = FIELDNAMES[: FIELDNAMES.index("status")]
SUMMARY_METRICS = (
    "utility",
    "plies",
    "think_time",
    "generated",
    "evaluated",
    "model_calls",
    "fallback_calls",
    "cache_hits",
)
SUMMARY_FIELDNAMES = (
    "stage",
    "agent",
    "configured_depth",
    "games_attempted",
    "games_with_results",
    "wins",
    "draws",
    "losses",
    "failures",
    *(field for metric in SUMMARY_METRICS for field in (f"mean_{metric}", f"{metric}_denominator")),
)


def _match(
    stage: str,
    focal_agent: str,
    focal_color: str,
    focal_depth: int,
    opponent: str,
    opponent_depth: int,
    max_plies: int,
    *,
    seed: int,
    live: bool = False,
    max_llm_calls: int | None = None,
    max_think_seconds: float | None = None,
) -> dict[str, Any]:
    agents = {focal_color: focal_agent}
    agents[{"orange": "black", "black": "orange"}[focal_color]] = opponent
    depths = {focal_color: focal_depth}
    depths[{"orange": "black", "black": "orange"}[focal_color]] = opponent_depth
    seeds = {"orange": 0, "black": 0}
    seeds[focal_color] = seed
    if opponent == "random":
        seeds = {"orange": seed, "black": seed}
    return {
        "stage": stage,
        "focal_agent": focal_agent,
        "focal_color": focal_color,
        "orange_agent": agents["orange"],
        "black_agent": agents["black"],
        "orange_depth": depths["orange"],
        "black_depth": depths["black"],
        "orange_seed": seeds["orange"],
        "black_seed": seeds["black"],
        "max_plies": max_plies,
        "max_llm_calls": max_llm_calls,
        "max_think_seconds": max_think_seconds,
        "live": live,
    }


def build_matrix(
    stage: str,
    evaluator: str | None = None,
    depth: int | None = None,
    max_think_seconds: float = DEFAULT_MAX_THINK_SECONDS,
) -> list[dict[str, Any]]:
    """Return the configured matches for one required experiment stage."""
    if stage == "a1":
        return [
            _match(stage, agent, color, 1, "random", 1, 100, seed=seed)
            for agent in ("minimax_1", "minimax_2", "minimax_3")
            for seed in SEEDS
            for color in COLORS
        ]
    if evaluator is None:
        raise ValueError(f"stage {stage} requires an evaluator")
    if stage == "a2":
        return [
            _match(stage, evaluator, color, 2, "random", 1, 100, seed=seed)
            for seed in SEEDS
            for color in COLORS
        ]
    if depth not in (1, 2):
        raise ValueError(f"stage {stage} requires depth 1 or 2")
    if stage == "b":
        configurations = (
            (evaluator, depth, False, None),
            ("llm_evaluator", 1, True, 100),
            ("llm_direct", 1, True, None),
        )
        return [
            _match(
                stage,
                agent,
                color,
                agent_depth,
                "random",
                1,
                60,
                seed=seed,
                live=live,
                max_llm_calls=max_calls,
                max_think_seconds=max_think_seconds,
            )
            for agent, agent_depth, live, max_calls in configurations
            for seed in SEEDS
            for color in COLORS
        ]
    if stage == "c":
        return [
            _match(
                stage,
                "llm_direct",
                color,
                1,
                evaluator,
                depth,
                60,
                seed=seed,
                live=True,
                max_think_seconds=max_think_seconds,
            )
            for seed in SEEDS
            for color in COLORS
        ]
    raise ValueError(f"unknown stage: {stage}")


def _key_values(line: str) -> dict[str, str]:
    values: dict[str, str] = {}
    try:
        tokens = shlex.split(line)
    except ValueError:
        tokens = line.split()
    for token in tokens:
        key, separator, value = token.partition("=")
        if separator and key:
            values[key] = value
    return values


def parse_output(stdout: str) -> dict[str, str]:
    """Parse result and agent metric lines, ignoring unrelated output."""
    parsed: dict[str, str] = {}
    for line in stdout.splitlines():
        values = _key_values(line.strip())
        if "result" in values:
            for key in ("result", "reason", "plies", "orange_utility", "black_utility"):
                if key in values:
                    parsed[key] = values[key]
            for color in COLORS:
                value = values.get(f"{color}_time")
                if value is not None:
                    parsed[f"{color}_time"] = value.removesuffix("s")
        color = values.get("agent")
        if color in COLORS:
            for metric in METRICS:
                if metric in values:
                    parsed[f"{color}_{metric}"] = values[metric]
    return parsed


def command_for(match: dict[str, Any], play_path: Path) -> list[str]:
    command = [
        sys.executable,
        str(play_path),
        "--view",
        "none",
        "--orange-agent",
        str(match["orange_agent"]),
        "--black-agent",
        str(match["black_agent"]),
        "--orange-depth",
        str(match["orange_depth"]),
        "--black-depth",
        str(match["black_depth"]),
        "--orange-seed",
        str(match["orange_seed"]),
        "--black-seed",
        str(match["black_seed"]),
        "--max-plies",
        str(match["max_plies"]),
    ]
    if match["max_llm_calls"] is not None:
        command.extend(("--max-llm-calls", str(match["max_llm_calls"])))
    if match["max_think_seconds"] is not None:
        command.extend(("--max-think-seconds", str(match["max_think_seconds"])))
    if match["live"]:
        command.append("--live")
    return command


def run_match(match: dict[str, Any], play_path: Path) -> dict[str, Any]:
    """Run one match and return a complete CSV row."""
    row = {field: "" for field in FIELDNAMES}
    row.update(match)
    try:
        completed = subprocess.run(
            command_for(match, play_path),
            cwd=play_path.parent,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as error:
        row.update(status="failed", error=f"{type(error).__name__}: {error}")
        return row

    row.update(
        parse_output(completed.stdout),
        exit_status=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )
    if completed.returncode == 0 and row["result"]:
        row["status"] = "ok"
    else:
        row["status"] = "failed"
        if completed.stderr.strip():
            row["error"] = completed.stderr.strip()
        elif completed.returncode != 0:
            row["error"] = f"process exited with status {completed.returncode}"
        else:
            row["error"] = "match produced no result line"
    return row


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _tested_agents(row: dict[str, Any]) -> tuple[str, ...]:
    if row["stage"] == "c":
        return tuple(
            dict.fromkeys(
                agent
                for agent in (str(row["orange_agent"]), str(row["black_agent"]))
                if agent != "random"
            )
        )
    agent = str(row["focal_agent"])
    return () if agent == "random" else (agent,)


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate tested agents from their own color perspective."""
    groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    values: dict[tuple[str, str, str], dict[str, list[float]]] = {}
    for row in rows:
        for agent in _tested_agents(row):
            color = "orange" if row["orange_agent"] == agent else "black"
            depth = str(row[f"{color}_depth"])
            key = (str(row["stage"]), agent, depth)
            if key not in groups:
                groups[key] = {
                    "stage": key[0],
                    "agent": agent,
                    "configured_depth": depth,
                    "games_attempted": 0,
                    "games_with_results": 0,
                    "wins": 0,
                    "draws": 0,
                    "losses": 0,
                    "failures": 0,
                }
                values[key] = {metric: [] for metric in SUMMARY_METRICS}
            summary = groups[key]
            summary["games_attempted"] += 1
            if row["status"] != "ok":
                summary["failures"] += 1

            result = str(row["result"]).upper()
            if result in ("ORANGE", "BLACK", "DRAW"):
                summary["games_with_results"] += 1
                if result == "DRAW":
                    summary["draws"] += 1
                elif result == color.upper():
                    summary["wins"] += 1
                else:
                    summary["losses"] += 1

            metric_fields = {
                "utility": f"{color}_utility",
                "plies": "plies",
                "think_time": f"{color}_time",
                "generated": f"{color}_generated",
                "evaluated": f"{color}_evaluated",
                "model_calls": f"{color}_model_calls",
                "fallback_calls": f"{color}_fallback_calls",
                "cache_hits": f"{color}_cache_hits",
            }
            for metric, field in metric_fields.items():
                number = _number(row.get(field))
                if number is not None:
                    values[key][metric].append(number)

    result_rows = []
    for key, summary in groups.items():
        for metric in SUMMARY_METRICS:
            metric_values = values[key][metric]
            summary[f"mean_{metric}"] = (
                sum(metric_values) / len(metric_values) if metric_values else ""
            )
            summary[f"{metric}_denominator"] = len(metric_values)
        result_rows.append(summary)
    return result_rows


def summary_path_for(output_path: Path) -> Path:
    suffix = output_path.suffix or ".csv"
    return output_path.with_name(f"{output_path.stem}-summary{suffix}")


def configuration_key(row: dict[str, Any]) -> tuple[str, ...]:
    """Return a stable key for every setting that changes a match invocation."""
    return tuple(
        "" if row.get(field) is None else str(row.get(field, ""))
        for field in CONFIGURATION_FIELDS
    )


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or not set(CONFIGURATION_FIELDS).issubset(
            reader.fieldnames
        ):
            raise ValueError(f"{path} is not a compatible raw results CSV")
        return [{field: row.get(field, "") for field in FIELDNAMES} for row in reader]


def write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    """Atomically replace a CSV and sync its contents before returning."""
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)


def progress_message(index: int, total: int, match: dict[str, Any]) -> str:
    color = str(match["focal_color"])
    return (
        f"[{index}/{total}] stage={match['stage']} agent={match['focal_agent']} "
        f"color={color} seed={match[f'{color}_seed']}"
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("a1", "a2", "b", "c"))
    parser.add_argument("--output", type=Path, required=True, help="output CSV path")
    parser.add_argument("--evaluator", help="selected deterministic evaluator")
    parser.add_argument("--depth", type=int, choices=(1, 2), help="selected depth")
    parser.add_argument(
        "--max-think-seconds",
        type=float,
        default=DEFAULT_MAX_THINK_SECONDS,
        help="per-player cumulative think limit for Stages B and C (default: 3600)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip exact configurations already recorded with status=ok",
    )
    args = parser.parse_args(argv)
    if args.stage in ("a2", "b", "c") and not args.evaluator:
        parser.error(f"stage {args.stage} requires --evaluator")
    if args.stage in ("b", "c") and args.depth is None:
        parser.error(f"stage {args.stage} requires --depth")
    if not math.isfinite(args.max_think_seconds) or args.max_think_seconds <= 0:
        parser.error("--max-think-seconds must be positive and finite")
    if args.resume and not args.output.is_file():
        parser.error(f"--resume requires an existing raw CSV: {args.output}")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    play_path = Path(__file__).with_name("play.py")
    matches = build_matrix(
        args.stage, args.evaluator, args.depth, args.max_think_seconds
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    summary_path = summary_path_for(args.output)
    print(f"Raw results: {args.output}")
    print(f"Summary: {summary_path}")

    existing_rows = read_rows(args.output) if args.resume else []
    rows_by_key = {configuration_key(row): row for row in existing_rows}
    write_csv(args.output, FIELDNAMES, list(rows_by_key.values()))
    write_csv(
        summary_path,
        SUMMARY_FIELDNAMES,
        summarize_rows(list(rows_by_key.values())),
    )

    for index, match in enumerate(matches, start=1):
        key = configuration_key(match)
        old_row = rows_by_key.get(key)
        if old_row is not None and old_row.get("status") == "ok":
            print(f"[{index}/{len(matches)}] skipping completed configuration")
            continue
        print(progress_message(index, len(matches), match), flush=True)
        rows_by_key[key] = run_match(match, play_path)
        rows = list(rows_by_key.values())
        write_csv(args.output, FIELDNAMES, rows)
        write_csv(summary_path, SUMMARY_FIELDNAMES, summarize_rows(rows))


if __name__ == "__main__":
    main()
