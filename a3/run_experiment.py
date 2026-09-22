"""Run offline agents or a genuinely paired live comparison."""

from __future__ import annotations

import argparse
import os
import statistics
from collections.abc import Callable, Sequence

from agents import LLMRepairAgent, MinConflictsAgent, RandomMoveAgent, RepairResult
from constraints import evaluate
from llm_client import OpenAICompatibleClient
from scenarios import Scenario, evaluation_scenarios


ResultRecord = tuple[Scenario, int, RepairResult]


def run_one(agent: object, scenario: Scenario, seed: int) -> RepairResult:
    return agent.repair(  # type: ignore[attr-defined]
        scenario.problem, scenario.initial_schedule, scenario.budget, seed
    )


def run_paired_comparison(
    scenarios: Sequence[Scenario],
    seeds: Sequence[int],
    llm_agent_factory: Callable[[int], LLMRepairAgent],
    emit: Callable[[str], None] = print,
) -> tuple[list[ResultRecord], list[ResultRecord]]:
    """Run both policies consecutively from each exact immutable pair start."""
    minimum_results: list[ResultRecord] = []
    llm_results: list[ResultRecord] = []
    for scenario in scenarios:
        for seed in seeds:
            pair_id = f"{scenario.identifier}/seed-{seed}"
            emit(f"PAIR {pair_id} budget={scenario.budget}")
            minimum = run_one(MinConflictsAgent(), scenario, seed)
            llm = run_one(llm_agent_factory(seed), scenario, seed)
            minimum_results.append((scenario, seed, minimum))
            llm_results.append((scenario, seed, llm))
            emit(
                f"  minconflicts feasible={minimum.feasible} edits={minimum.metrics.edits}"
            )
            emit(
                f"  llm feasible={llm.feasible} edits={llm.metrics.edits} "
                f"client_failures={llm.metrics.client_failures}"
            )
    return minimum_results, llm_results


def summarize(name: str, records: Sequence[ResultRecord]) -> None:
    results = [record[2] for record in records]
    successes = [result for result in results if result.feasible]
    final = [evaluate(scenario.problem, result.schedule) for scenario, _, result in records]
    soft = [evaluation.soft_penalty for evaluation, result in zip(final, results) if result.feasible]
    mean_soft = f"{statistics.fmean(soft):.2f}" if soft else "N/A"
    edits = [result.metrics.edits for result in results]
    runtimes = [result.metrics.runtime_seconds for result in results]
    print(
        f"{name}: feasible={len(successes)}/{len(results)} "
        f"mean_final_hard={statistics.fmean(e.hard_violations for e in final):.2f} "
        f"mean_soft_success={mean_soft} "
        f"mean_edits={statistics.fmean(edits):.2f} edits_range={min(edits)}-{max(edits)} "
        f"mean_runtime={statistics.fmean(runtimes):.6f}s "
        f"runtime_range={min(runtimes):.6f}-{max(runtimes):.6f}s"
    )
    if name == "minconflicts":
        print(
            "  mean_evaluator_calls="
            f"{statistics.fmean(r.metrics.evaluator_calls for r in results):.2f}"
        )
    if name == "llm":
        print(
            f"  mean_model_calls={statistics.fmean(r.metrics.model_calls for r in results):.2f} "
            f"malformed_total={sum(r.metrics.malformed_requests for r in results)} "
            f"invalid_total={sum(r.metrics.invalid_requests for r in results)} "
            f"repeated_total={sum(r.metrics.repeated_requests for r in results)} "
            f"client_failures_total={sum(r.metrics.client_failures for r in results)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--agent", choices=("minconflicts", "random", "llm"))
    mode.add_argument("--all", action="store_true", help="run offline min-conflicts and random agents")
    mode.add_argument("--compare", action="store_true", help="run paired min-conflicts and live LLM")
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--scenario", default=None)
    parser.add_argument("--trace", action="store_true")
    parser.add_argument("--live", action="store_true", help="explicitly enable course model calls")
    parser.add_argument("--model", default="Gemma-4-26B-A4B-it-oQ4e-mtp")
    parser.add_argument("--endpoint", default="http://golem:8000/v1")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",")]
    scenarios = [s for s in evaluation_scenarios() if args.scenario in (None, s.identifier)]
    if not scenarios:
        parser.error("unknown scenario")
    if args.compare:
        if not args.live:
            parser.error("--compare requires --live")
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            parser.error("OPENAI_API_KEY must be set to a non-secret placeholder")
        minimum, llm = run_paired_comparison(
            scenarios,
            seeds,
            lambda seed: LLMRepairAgent(
                OpenAICompatibleClient(
                    args.model, api_key, args.endpoint, args.temperature, seed
                )
            ),
        )
        if args.trace:
            for name, records in (("minconflicts", minimum), ("llm", llm)):
                for scenario, seed, result in records:
                    print(
                        f"TRACE {name} {scenario.identifier}/seed-{seed}",
                        *result.trace,
                        sep="\n  ",
                    )
        summarize("minconflicts", minimum)
        summarize("llm", llm)
        return
    if args.all and args.live:
        parser.error("--all is offline and cannot be combined with --live")
    name = args.agent or "minconflicts"
    if name == "llm" and not args.live:
        parser.error("--agent llm requires --live")
    if args.live and name != "llm":
        parser.error("--live requires --agent llm or --compare")
    api_key = os.environ.get("OPENAI_API_KEY") if args.live else None
    if args.live and not api_key:
        parser.error("OPENAI_API_KEY must be set to a non-secret placeholder")
    names = ("minconflicts", "random") if args.all else (name,)
    for current_name in names:
        records: list[ResultRecord] = []
        for scenario in scenarios:
            for seed in seeds:
                if current_name == "minconflicts":
                    agent = MinConflictsAgent()
                elif current_name == "random":
                    agent = RandomMoveAgent()
                else:
                    agent = LLMRepairAgent(
                        OpenAICompatibleClient(
                            args.model,
                            api_key or "",
                            args.endpoint,
                            args.temperature,
                            seed,
                        )
                    )
                result = run_one(agent, scenario, seed)
                records.append((scenario, seed, result))
                if args.trace:
                    print(f"PAIR {scenario.identifier}/seed-{seed}", *result.trace, sep="\n  ")
        summarize(current_name, records)


if __name__ == "__main__":
    main()
