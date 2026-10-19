"""Run reproducible paired evaluations on generated Rescue Rover scenarios."""

from __future__ import annotations

import argparse
import csv
import random
import subprocess
from collections import Counter, defaultdict
from dataclasses import dataclass, fields
from pathlib import Path
from statistics import mean, median
from typing import Callable, Sequence

from rover import MissionStatus
from run_experiment import FACTORIES, EpisodeResult, run_scenario
from scenarios import generated_scenarios


@dataclass(frozen=True)
class EvaluationRecord:
    agent: str
    scenario: str
    seed: int
    status: str
    score: int
    actions: int
    moves: int
    scans: int
    traction_moves: int
    traction_toggles: int
    unprotected_slips: int
    invalid_actions: int
    code_version: str


def evaluate(
    agent_names: Sequence[str],
    scenario_seed: int,
    scenario_count: int,
    agent_seeds: Sequence[int],
    max_actions: int,
    code_version: str,
) -> tuple[EvaluationRecord, ...]:
    if not agent_names:
        raise ValueError("agent_names cannot be empty")
    if len(set(agent_names)) != len(agent_names):
        raise ValueError("agent_names must be unique")
    unknown_agents = [name for name in agent_names if name not in FACTORIES]
    if unknown_agents:
        raise ValueError(f"unknown agents: {', '.join(unknown_agents)}")
    if scenario_count < 2:
        raise ValueError("scenario_count must be at least 2")
    if not agent_seeds:
        raise ValueError("agent_seeds cannot be empty")
    if len(set(agent_seeds)) != len(agent_seeds):
        raise ValueError("agent_seeds must be unique")
    if max_actions <= 0:
        raise ValueError("max_actions must be positive")

    scenarios = generated_scenarios(scenario_seed, scenario_count)
    records: list[EvaluationRecord] = []
    for agent_name in agent_names:
        factory = FACTORIES[agent_name]
        for scenario in scenarios:
            for agent_seed in agent_seeds:
                result = run_scenario(factory, scenario, agent_seed, max_actions)
                records.append(
                    _record(agent_name, agent_seed, result, code_version)
                )
    return tuple(records)


def _record(
    agent_name: str,
    agent_seed: int,
    result: EpisodeResult,
    code_version: str,
) -> EvaluationRecord:
    return EvaluationRecord(
        agent=agent_name,
        scenario=result.scenario,
        seed=agent_seed,
        status=result.status.value,
        score=result.score,
        actions=result.actions,
        moves=result.moves,
        scans=result.scans,
        traction_moves=result.traction_moves,
        traction_toggles=result.traction_toggles,
        unprotected_slips=result.unprotected_slips,
        invalid_actions=result.invalid_actions,
        code_version=code_version,
    )


def write_csv(records: Sequence[EvaluationRecord], output_path: Path) -> None:
    field_names = [field.name for field in fields(EvaluationRecord)]
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=field_names)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {field_name: getattr(record, field_name) for field_name in field_names}
            )


def print_summary(
    records: Sequence[EvaluationRecord], agent_names: Sequence[str]
) -> None:
    if not agent_names:
        raise ValueError("agent_names cannot be empty")
    if len(set(agent_names)) != len(agent_names):
        raise ValueError("agent_names must be unique")
    grouped = {
        agent_name: [record for record in records if record.agent == agent_name]
        for agent_name in agent_names
    }
    empty_agents = [name for name, agent_records in grouped.items() if not agent_records]
    if empty_agents:
        raise ValueError(f"no evaluation records for: {', '.join(empty_agents)}")
    scenario_sets = {
        name: {record.scenario for record in agent_records}
        for name, agent_records in grouped.items()
    }
    expected_scenarios = scenario_sets[agent_names[0]]
    if any(scenarios != expected_scenarios for scenarios in scenario_sets.values()):
        raise ValueError("agents must have records for identical scenarios")
    print(
        "agent       episodes success mean_score median_score actions moves "
        "scans traction_moves traction_toggles unprotected_slips invalid"
    )
    for agent_name in agent_names:
        agent_records = grouped[agent_name]
        success_count = sum(
            record.status == MissionStatus.SUCCESS.value for record in agent_records
        )
        print(
            f"{agent_name:<11} {len(agent_records):>8} "
            f"{success_count / len(agent_records):>7.1%} "
            f"{mean(record.score for record in agent_records):>10.2f} "
            f"{median(record.score for record in agent_records):>12.2f} "
            f"{mean(record.actions for record in agent_records):>7.2f} "
            f"{mean(record.moves for record in agent_records):>5.2f} "
            f"{mean(record.scans for record in agent_records):>5.2f} "
            f"{mean(record.traction_moves for record in agent_records):>14.2f} "
            f"{mean(record.traction_toggles for record in agent_records):>16.2f} "
            f"{mean(record.unprotected_slips for record in agent_records):>17.2f} "
            f"{mean(record.invalid_actions for record in agent_records):>7.2f}"
        )
        status_counts = Counter(record.status for record in agent_records)
        print(
            "  status: "
            + ", ".join(
                f"{status.value}={status_counts[status.value]}"
                for status in MissionStatus
                if status is not MissionStatus.RUNNING
            )
        )
        scenario_success = _scenario_values(
            agent_records,
            lambda record: float(record.status == MissionStatus.SUCCESS.value),
        )
        success_mean, success_low, success_high = _bootstrap_interval(
            tuple(scenario_success.values())
        )
        print(
            f"  scenario success proportion: {success_mean:.3f} "
            f"(descriptive 95% bootstrap interval "
            f"[{success_low:.3f}, {success_high:.3f}])"
        )

    if len(agent_names) > 1:
        baseline_name = agent_names[0]
        baseline_scores = _scenario_values(
            grouped[baseline_name], lambda record: float(record.score)
        )
        print(f"paired scenario score differences relative to {baseline_name}:")
        for agent_name in agent_names[1:]:
            agent_scores = _scenario_values(
                grouped[agent_name], lambda record: float(record.score)
            )
            differences = tuple(
                agent_scores[scenario] - baseline_scores[scenario]
                for scenario in sorted(baseline_scores)
            )
            difference_mean, low, high = _bootstrap_interval(differences)
            print(
                f"  {agent_name}: {difference_mean:+.2f} "
                f"(descriptive 95% bootstrap interval [{low:+.2f}, {high:+.2f}])"
            )


def _scenario_values(
    records: Sequence[EvaluationRecord],
    value: Callable[[EvaluationRecord], float],
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        grouped[record.scenario].append(value(record))
    return {
        scenario: mean(scenario_values)
        for scenario, scenario_values in grouped.items()
    }


def _bootstrap_interval(
    values: Sequence[float], resamples: int = 5000, seed: int = 4300
) -> tuple[float, float, float]:
    if not values:
        raise ValueError("cannot summarize an empty sequence")
    center = mean(values)
    if len(values) < 2:
        return center, center, center
    rng = random.Random(seed)
    bootstrap_means = sorted(
        mean(rng.choices(values, k=len(values))) for _ in range(resamples)
    )
    low = bootstrap_means[int(0.025 * resamples)]
    high = bootstrap_means[int(0.975 * resamples)]
    return center, low, high


def repository_version(allow_dirty: bool) -> str:
    version = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ("git", "status", "--short"),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status and not allow_dirty:
        raise RuntimeError(
            "working tree is not clean; commit changes or pass --allow-dirty"
        )
    return f"{version}-dirty" if status else version


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("agents", nargs="+", choices=FACTORIES)
    parser.add_argument("--scenario-seed", type=int, default=4300)
    parser.add_argument("--scenarios", type=int, default=30)
    parser.add_argument("--agent-seeds", type=int, default=30)
    parser.add_argument("--seed-start", type=int, default=4300)
    parser.add_argument("--max-actions", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("results.csv"))
    parser.add_argument("--allow-dirty", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.scenarios < 2:
        raise ValueError("--scenarios must be at least 2")
    if args.agent_seeds <= 0:
        raise ValueError("--agent-seeds must be positive")
    if args.max_actions <= 0:
        raise ValueError("--max-actions must be positive")

    code_version = repository_version(args.allow_dirty)
    seeds = tuple(range(args.seed_start, args.seed_start + args.agent_seeds))
    records = evaluate(
        args.agents,
        args.scenario_seed,
        args.scenarios,
        seeds,
        args.max_actions,
        code_version,
    )
    write_csv(records, args.output)
    print_summary(records, args.agents)
    print(f"raw results: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
