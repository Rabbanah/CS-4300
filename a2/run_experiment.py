"""Run traces and controlled experiments for U0-HW-03."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from statistics import mean
from time import perf_counter

from agents import (
    AStarSearchAgent,
    RandomRecoveryAgent,
    SearchAgent,
    SearchMetrics,
    UniformCostSearchAgent,
)
from problem import RecoveryProblem
from recovery import Action, RecoveryState, ServiceRecoveryEnvironment, Transition
from scenarios import Scenario, evaluation_scenarios, get_scenario


@dataclass(frozen=True)
class EpisodeResult:
    agent_name: str
    scenario_id: int
    initial_state: RecoveryState
    final_state: RecoveryState
    recovered: bool
    legal: bool
    actions: tuple[Action, ...]
    total_cost: int
    trace: tuple[Transition, ...]
    search_metrics: SearchMetrics | None


def _execute_plan(
    agent_name: str,
    scenario: Scenario,
    plan: tuple[Action, ...] | None,
    metrics: SearchMetrics,
) -> EpisodeResult:
    environment = ServiceRecoveryEnvironment(scenario.initial_state)
    legal = True

    if plan is not None:
        for action in plan:
            try:
                environment.step(action)
            except (TypeError, ValueError, RuntimeError):
                legal = False
                break

    return EpisodeResult(
        agent_name=agent_name,
        scenario_id=scenario.identifier,
        initial_state=scenario.initial_state,
        final_state=environment.state,
        recovered=environment.recovered and legal,
        legal=legal,
        actions=tuple(row.action for row in environment.trace),
        total_cost=environment.total_cost,
        trace=tuple(environment.trace),
        search_metrics=metrics,
    )


def run_search(agent_name: str, scenario: Scenario) -> EpisodeResult:
    factories: dict[str, type[SearchAgent]] = {
        "ucs": UniformCostSearchAgent,
        "astar": AStarSearchAgent,
    }
    agent = factories[agent_name]()
    problem = RecoveryProblem(scenario.initial_state)
    started = perf_counter()
    result = agent.plan(problem)
    elapsed = perf_counter() - started
    metrics = replace(result.metrics, planning_time_seconds=elapsed)
    return _execute_plan(agent_name, scenario, result.plan, metrics)


def run_random(scenario: Scenario) -> EpisodeResult:
    environment = ServiceRecoveryEnvironment(scenario.initial_state)
    agent = RandomRecoveryAgent(seed=10_000 + scenario.identifier)
    legal = True

    while not environment.done:
        action = agent.act(environment.state, environment.legal_actions())
        try:
            environment.step(action)
        except (TypeError, ValueError, RuntimeError):
            legal = False
            break

    return EpisodeResult(
        agent_name="random",
        scenario_id=scenario.identifier,
        initial_state=scenario.initial_state,
        final_state=environment.state,
        recovered=environment.recovered and legal,
        legal=legal,
        actions=tuple(row.action for row in environment.trace),
        total_cost=environment.total_cost,
        trace=tuple(environment.trace),
        search_metrics=None,
    )


def run_agent(agent_name: str, scenario: Scenario) -> EpisodeResult:
    if agent_name == "random":
        return run_random(scenario)
    return run_search(agent_name, scenario)


def print_trace(result: EpisodeResult) -> None:
    print(f"agent: {result.agent_name}")
    print(f"scenario: {result.scenario_id}")
    print(f"initial: {result.initial_state}")
    for number, row in enumerate(result.trace, start=1):
        print(
            f"{number:>2}  {row.action.name:<18} cost={row.cost:<2} "
            f"state={row.after}"
        )
    print(f"recovered: {result.recovered}")
    print(f"legal: {result.legal}")
    print(f"actions: {len(result.actions)}")
    print(f"cost: {result.total_cost}")
    if result.search_metrics is not None:
        metrics = result.search_metrics
        print(
            "search: "
            f"generated={metrics.generated} expanded={metrics.expanded} "
            f"peak_frontier={metrics.peak_frontier} "
            f"time={metrics.planning_time_seconds:.6f}s"
        )


def _mean_search_metric(
    results: list[EpisodeResult], attribute: str
) -> float | None:
    values = [
        getattr(result.search_metrics, attribute)
        for result in results
        if result.search_metrics is not None
    ]
    return mean(values) if values else None


def _display(value: float | None, digits: int = 2) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def print_summary(results: list[EpisodeResult]) -> None:
    print(
        "| Agent | Success | Mean cost | Mean actions | Mean generated | "
        "Mean expanded | Mean peak | Mean time (ms) |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for agent_name in ("random", "ucs", "astar"):
        rows = [result for result in results if result.agent_name == agent_name]
        if not rows:
            continue
        success = mean(result.recovered for result in rows)
        generated = _mean_search_metric(rows, "generated")
        expanded = _mean_search_metric(rows, "expanded")
        peak = _mean_search_metric(rows, "peak_frontier")
        seconds = _mean_search_metric(rows, "planning_time_seconds")
        milliseconds = None if seconds is None else seconds * 1000
        print(
            f"| {agent_name} | {success:.1%} | "
            f"{mean(result.total_cost for result in rows):.2f} | "
            f"{mean(len(result.actions) for result in rows):.2f} | "
            f"{_display(generated)} | {_display(expanded)} | "
            f"{_display(peak)} | {_display(milliseconds, 3)} |"
        )

    ucs = {
        result.scenario_id: result
        for result in results
        if result.agent_name == "ucs"
    }
    astar = {
        result.scenario_id: result
        for result in results
        if result.agent_name == "astar"
    }
    mismatches = [
        identifier
        for identifier in sorted(ucs.keys() & astar.keys())
        if ucs[identifier].recovered != astar[identifier].recovered
        or ucs[identifier].total_cost != astar[identifier].total_cost
    ]
    if ucs and astar:
        print(f"UCS/A* cost mismatches: {len(mismatches)}")
        if mismatches:
            print("mismatched scenarios: " + ", ".join(map(str, mismatches)))


def _parse_range(value: str) -> tuple[int, int]:
    try:
        start_text, stop_text = value.split(":", maxsplit=1)
        start, stop = int(start_text), int(stop_text)
    except ValueError as error:
        raise argparse.ArgumentTypeError("use START:STOP, for example 0:50") from error
    if not 0 <= start < stop <= 50:
        raise argparse.ArgumentTypeError("range must satisfy 0 <= START < STOP <= 50")
    return start, stop


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=("random", "ucs", "astar"))
    parser.add_argument("--scenario", type=int)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--all", action="store_true", dest="run_all")
    parser.add_argument("--scenarios", type=_parse_range, default=(0, 50))
    args = parser.parse_args()

    if args.run_all:
        start, stop = args.scenarios
        scenarios = evaluation_scenarios(start, stop)
        results = [
            run_agent(agent_name, scenario)
            for agent_name in ("random", "ucs", "astar")
            for scenario in scenarios
        ]
        print_summary(results)
        return

    if args.agent is None or args.scenario is None:
        parser.error("provide --all or both --agent and --scenario")
    result = run_agent(args.agent, get_scenario(args.scenario))
    if args.trace:
        print_trace(result)
    else:
        print_summary([result])


if __name__ == "__main__":
    main()
