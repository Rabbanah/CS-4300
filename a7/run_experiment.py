"""Run traces and comparisons for Rescue Rover agents."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from statistics import mean
from typing import Callable

from agents import LogicAgent, RandomAgent, RoverAgent, StopAgent
from rover import (
    Action,
    MissionStatus,
    Percept,
    RescueRoverEnvironment,
    RoverScenario,
)
from scenarios import get_scenario, training_scenarios


@dataclass(frozen=True)
class TraceRow:
    step: int
    percept: Percept
    action: Action
    event: str
    score_delta: int
    total_score: int


@dataclass(frozen=True)
class EpisodeResult:
    scenario: str
    status: MissionStatus
    score: int
    actions: int
    moves: int
    scans: int
    traction_moves: int
    traction_toggles: int
    unprotected_slips: int
    invalid_actions: int
    trace: tuple[TraceRow, ...]


AgentFactory = Callable[[int], RoverAgent]


FACTORIES: dict[str, AgentFactory] = {
    "logic": lambda seed: LogicAgent(seed),
    "random": lambda seed: RandomAgent(seed),
    "stop": lambda seed: StopAgent(),
}


def run_episode(
    factory: AgentFactory,
    scenario_id: int,
    seed: int = 0,
    max_actions: int = 100,
) -> EpisodeResult:
    return run_scenario(factory, get_scenario(scenario_id), seed, max_actions)


def run_scenario(
    factory: AgentFactory,
    scenario: RoverScenario,
    seed: int = 0,
    max_actions: int = 100,
) -> EpisodeResult:
    environment = RescueRoverEnvironment(scenario, max_actions)
    agent = factory(seed)
    agent.reset()
    percept = environment.percept()
    rows: list[TraceRow] = []

    while not environment.done:
        action = agent.act(percept)
        transition = environment.step(action)
        rows.append(
            TraceRow(
                step=environment.action_count,
                percept=percept,
                action=action,
                event=transition.outcome.event.value,
                score_delta=transition.outcome.score_delta,
                total_score=transition.total_score,
            )
        )
        percept = transition.percept

    return EpisodeResult(
        scenario=scenario.identifier,
        status=environment.status,
        score=environment.total_score,
        actions=environment.action_count,
        moves=environment.move_count,
        scans=environment.scan_count,
        traction_moves=environment.traction_moves,
        traction_toggles=environment.traction_toggles,
        unprotected_slips=environment.unprotected_slips,
        invalid_actions=environment.invalid_actions,
        trace=tuple(rows),
    )


def print_trace(result: EpisodeResult) -> None:
    for row in result.trace:
        position = row.percept.position
        print(
            f"{row.step:>3} ({position.row},{position.column}) "
            f"traction={str(row.percept.traction_enabled):<5} "
            f"action={row.action.kind.value:<18} event={row.event:<20} "
            f"delta={row.score_delta:>5} total={row.total_score:>5}"
        )
    print(
        f"scenario={result.scenario} status={result.status.value} "
        f"score={result.score} actions={result.actions} moves={result.moves} "
        f"scans={result.scans} traction_moves={result.traction_moves} "
        f"traction_toggles={result.traction_toggles} "
        f"unprotected_slips={result.unprotected_slips} "
        f"invalid={result.invalid_actions}"
    )


def print_experiment(factory: AgentFactory, seed: int) -> None:
    results = [
        run_episode(factory, scenario_id, seed + scenario_id)
        for scenario_id, _ in enumerate(training_scenarios())
    ]
    successes = sum(result.status is MissionStatus.SUCCESS for result in results)
    print(f"scenarios: {len(results)}")
    print(f"success rate: {successes / len(results):.1%}")
    print(f"mean score: {mean(result.score for result in results):.2f}")
    print(f"mean actions: {mean(result.actions for result in results):.2f}")
    print(f"mean moves: {mean(result.moves for result in results):.2f}")
    print(f"mean scans: {mean(result.scans for result in results):.2f}")
    print(
        f"mean traction moves: {mean(result.traction_moves for result in results):.2f}"
    )
    print(
        "mean traction toggles: "
        f"{mean(result.traction_toggles for result in results):.2f}"
    )
    print(
        "mean unprotected slips: "
        f"{mean(result.unprotected_slips for result in results):.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("agent", choices=FACTORIES)
    parser.add_argument("--scenario", type=int)
    parser.add_argument("--seed", type=int, default=4300)
    args = parser.parse_args()
    factory = FACTORIES[args.agent]

    if args.scenario is None:
        print_experiment(factory, args.seed)
    else:
        print_trace(run_episode(factory, args.scenario, args.seed))


if __name__ == "__main__":
    main()
