"""Run a match between registered Jetan agents."""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
from urllib.parse import urlparse

from agents import (
    AgentConfiguration,
    JetanAgent,
    available_agent_types,
    create_configured_agent,
)
from environment import DEFAULT_MAX_THINK_SECONDS, JetanEnvironment, NullViewer
from jetan import Player
from llm_client import OpenAICompatibleClient
from viewers import TerminalViewer, TkViewer

DEFAULT_MODEL = "Gemma-4-26B-A4B-it-oQ4e-mtp"
DEFAULT_ENDPOINT = "http://golem:8000/v1"


def load_environment_file(path: Path) -> None:
    """Load simple KEY=VALUE entries without replacing shell variables."""
    if not path.is_file():
        return
    for line_number, raw_line in enumerate(path.read_text().splitlines(), 1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if not separator or not key.isidentifier():
            raise ValueError(f"invalid .env entry on line {line_number}")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        os.environ.setdefault(key, value)


def endpoint_is_allowed(endpoint: str) -> bool:
    parsed = urlparse(endpoint)
    if parsed.scheme == "https":
        return bool(parsed.hostname)
    return parsed.scheme == "http" and parsed.hostname in {
        "golem",
        "localhost",
        "127.0.0.1",
        "::1",
    }


def format_agent_metrics(label: str, agent: JetanAgent) -> str:
    fields = [f"agent={label}", f"name={agent.name!r}"]
    search = getattr(agent, "total_metrics", None)
    if search is not None:
        fields.extend(
            (
                f"generated={search.generated}",
                f"evaluated={search.evaluated}",
                f"maximum_depth={search.maximum_depth}",
            )
        )
    evaluation = getattr(agent, "llm_evaluation", None)
    if evaluation is not None:
        fields.extend(
            (
                f"model_calls={evaluation.total_model_calls}",
                f"fallback_calls={evaluation.total_fallback_calls}",
                f"cache_hits={evaluation.total_cache_hits}",
            )
        )
    elif hasattr(agent, "model_calls"):
        fields.extend(
            (
                f"model_calls={agent.model_calls}",
                f"fallback_calls={agent.fallback_calls}",
            )
        )
    return " ".join(fields)


def main() -> None:
    load_environment_file(Path(__file__).with_name(".env"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", choices=("none", "text", "tk"), default="text")
    parser.add_argument("--delay", type=float, default=0.1)
    parser.add_argument(
        "--orange-agent", choices=available_agent_types(), default="random"
    )
    parser.add_argument(
        "--black-agent", choices=available_agent_types(), default="random"
    )
    parser.add_argument("--orange-seed", type=int, default=1)
    parser.add_argument("--black-seed", type=int, default=2)
    parser.add_argument("--orange-depth", type=int, default=1)
    parser.add_argument("--black-depth", type=int, default=1)
    parser.add_argument(
        "--max-llm-calls",
        type=int,
        default=os.environ.get("JETAN_MAX_LLM_CALLS", "64"),
    )
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL))
    parser.add_argument(
        "--endpoint", default=os.environ.get("OPENAI_BASE_URL", DEFAULT_ENDPOINT)
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=os.environ.get("OPENAI_TEMPERATURE", "0"),
    )
    parser.add_argument("--max-plies", type=int, default=500)
    parser.add_argument(
        "--max-think-seconds", type=float, default=DEFAULT_MAX_THINK_SECONDS
    )
    args = parser.parse_args()
    if args.delay < 0:
        parser.error("--delay must be nonnegative")
    if args.max_plies < 1:
        parser.error("--max-plies must be at least one")
    if args.orange_depth < 1 or args.black_depth < 1:
        parser.error("both agent depths must be at least one")
    if args.max_llm_calls < 0:
        parser.error("--max-llm-calls cannot be negative")
    if not math.isfinite(args.max_think_seconds) or args.max_think_seconds <= 0:
        parser.error("--max-think-seconds must be positive and finite")

    if args.live:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            parser.error("OPENAI_API_KEY must be set for --live")
        if not endpoint_is_allowed(args.endpoint):
            parser.error(
                "--endpoint must use HTTPS; HTTP is allowed only for golem or "
                "a loopback host"
            )

        def client_for(seed: int) -> OpenAICompatibleClient:
            return OpenAICompatibleClient(
                args.model,
                api_key,
                args.endpoint,
                args.temperature,
                seed,
            )

    else:

        def client_for(seed: int) -> None:
            del seed
            return None

    if args.view == "text":
        viewer = TerminalViewer(delay=args.delay)
    elif args.view == "tk":
        viewer = TkViewer(delay=args.delay)
    else:
        viewer = NullViewer()
    orange = create_configured_agent(
        args.orange_agent,
        AgentConfiguration(
            args.orange_seed,
            "Orange",
            args.orange_depth,
            args.max_llm_calls,
            client_for(args.orange_seed),
        ),
    )
    black = create_configured_agent(
        args.black_agent,
        AgentConfiguration(
            args.black_seed,
            "Black",
            args.black_depth,
            args.max_llm_calls,
            client_for(args.black_seed),
        ),
    )
    environment = JetanEnvironment(
        orange,
        black,
        viewer=viewer,
        max_plies=args.max_plies,
        max_think_seconds=args.max_think_seconds,
    )
    result = environment.run()
    winner = result.winner.name if result.winner is not None else "DRAW"
    print(
        f"result={winner} reason={result.reason} plies={result.plies} "
        f"orange_utility={result.utility(Player.ORANGE)} "
        f"black_utility={result.utility(Player.BLACK)} "
        f"orange_time={result.orange_think_seconds:.6f}s "
        f"black_time={result.black_think_seconds:.6f}s"
    )
    print(format_agent_metrics("orange", orange))
    print(format_agent_metrics("black", black))


if __name__ == "__main__":
    main()
