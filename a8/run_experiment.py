"""CLI runner for the bounded ReAct Sudoku agent (offline rule-based or live LLM)."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Sequence

from agent import AgentRunResult, ReActSudokuAgent
from controller import ControllerConfig
from llm_client import ChatClient, OpenAICompatibleClient, RuleBasedReActClient
from scenarios import SCENARIOS, SudokuScenario


def format_trace(result: AgentRunResult) -> str:
    lines: list[str] = []
    for entry in result.trace:
        call_str = (
            entry.parsed_call.signature()
            if entry.parsed_call is not None
            else f"RAW({entry.raw_response[:50]!r})"
        )
        thought_str = (
            f" thought={entry.parsed_call.thought!r}"
            if entry.parsed_call and entry.parsed_call.thought
            else ""
        )
        obs = entry.observation
        lines.append(
            f"  [Step {entry.step:02d}] {call_str}{thought_str} -> "
            f"accepted={obs.accepted} budget={obs.remaining_budget} msg={obs.message!r}"
        )
    return "\n".join(lines)


def print_scenario_summary(scenario: SudokuScenario, result: AgentRunResult, trace: bool) -> None:
    m = result.metrics
    print(
        f"Scenario: {scenario.name} ({scenario.difficulty}) | "
        f"status={result.terminal_status.value} | solved={result.solved} | "
        f"steps={m.steps_taken} | placements={m.placements} | undos={m.undos} | "
        f"kb_adds={m.kb_adds} | kb_queries={m.kb_queries} | "
        f"inspections={m.inspections} | rejected={m.rejected_actions} | "
        f"time={m.runtime_seconds:.3f}s"
    )
    if trace:
        print(format_trace(result))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the bounded ReAct Sudoku agent on 9x9 Sudoku scenarios."
    )
    parser.add_argument(
        "--scenario",
        choices=sorted(SCENARIOS.keys()),
        default=None,
        help="Run a single named scenario (default: run all scenarios).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all scenarios in scenarios.py.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=100,
        help="Maximum action budget per puzzle episode (default: 100).",
    )
    parser.add_argument(
        "--history-limit",
        type=int,
        default=12,
        help="Number of recent trace steps included in the model prompt (default: 12).",
    )
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Print step-by-step ReAct tool-call and observation trace.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use OpenAICompatibleClient instead of offline RuleBasedReActClient.",
    )
    parser.add_argument(
        "--model",
        default="Gemma-4-26B-A4B-it-oQ4e-mtp",
        help="Model identifier for --live mode.",
    )
    parser.add_argument(
        "--endpoint",
        default="http://golem:8000/v1",
        help="OpenAI-compatible base URL for --live mode.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for --live mode (default: 0.0).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=4300,
        help="Random seed for --live mode (default: 4300).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.scenario is not None:
        selected = [SCENARIOS[args.scenario]]
    else:
        selected = [SCENARIOS[k] for k in sorted(SCENARIOS.keys())]

    config = ControllerConfig(max_steps=args.max_steps)

    for scenario in selected:
        if args.live:
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                print(
                    "error: OPENAI_API_KEY environment variable must be set for --live runs.",
                    file=sys.stderr,
                )
                return 2
            client: ChatClient = OpenAICompatibleClient(
                model=args.model,
                api_key=api_key,
                base_url=args.endpoint,
                temperature=args.temperature,
                seed=args.seed,
            )
        else:
            client = RuleBasedReActClient()

        agent = ReActSudokuAgent(
            client=client,
            config=config,
            history_limit=args.history_limit,
        )
        result = agent.solve(scenario.make_board())
        print_scenario_summary(scenario, result, trace=args.trace)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

