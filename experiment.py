"""Run one trace or a repeated comparison of Delivery Hallway agents."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from statistics import mean
from typing import Callable

from agents import Agent, ModelBasedReflexAgent, RandomAgent, SimpleReflexAgent
from hallway import Action, HallwayEnvironment, Percept


@dataclass(frozen=True)
class TraceRow:
    step: int
    percept: Percept
    action: Action
    action_succeeded: bool
    reward: int


@dataclass(frozen=True)
class EpisodeResult:
    delivered: bool
    steps: int
    total_reward: int
    invalid_actions: int
    trace: tuple[TraceRow, ...]


AgentFactory = Callable[[int], Agent]


FACTORIES: dict[str, AgentFactory] = {
    "random": lambda seed: RandomAgent(seed),
    "reflex": lambda seed: SimpleReflexAgent(),
    "model": lambda seed: ModelBasedReflexAgent(),
}


def run_episode(factory: AgentFactory, seed: int) -> EpisodeResult:
    environment = HallwayEnvironment(seed)
    agent = factory(seed)
    agent.reset()
    percept = environment.percept()
    rows: list[TraceRow] = []
    invalid_actions = 0

    while not environment.done:
        action = agent.act(percept)
        transition = environment.step(action)
        rows.append(
            TraceRow(
                environment.steps,
                percept,
                action,
                transition.action_succeeded,
                transition.reward,
            )
        )
        invalid_actions += not transition.action_succeeded
        percept = transition.percept

    return EpisodeResult(
        environment.delivered,
        environment.steps,
        environment.total_reward,
        invalid_actions,
        tuple(rows),
    )


def print_trace(result: EpisodeResult) -> None:
    for row in result.trace:
        print(
            f"{row.step:>2}  {row.percept!s:<75}  "
            f"{row.action.value:<10} success={row.action_succeeded!s:<5} reward={row.reward:>3}"
        )
    print(
        f"delivered={result.delivered} steps={result.steps} "
        f"reward={result.total_reward} invalid={result.invalid_actions}"
    )


def print_experiment(factory: AgentFactory, episodes: int) -> None:
    results = [run_episode(factory, seed) for seed in range(episodes)]
    delivered = [result for result in results if result.delivered]
    print(f"episodes: {episodes}")
    print(f"delivery rate: {len(delivered) / episodes:.1%}")
    print(f"mean reward: {mean(result.total_reward for result in results):.2f}")
    print(f"mean actions: {mean(result.steps for result in results):.2f}")
    print(f"mean invalid actions: {mean(result.invalid_actions for result in results):.2f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("agent", choices=FACTORIES)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--trace-seed", type=int)
    args = parser.parse_args()
    factory = FACTORIES[args.agent]

    if args.trace_seed is None:
        print_experiment(factory, args.episodes)
    else:
        print_trace(run_episode(factory, args.trace_seed))


if __name__ == "__main__":
    main()
