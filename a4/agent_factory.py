"""Registry for command-line Jetan agents."""

from collections.abc import Callable
from dataclasses import dataclass

from agent_protocol import JetanAgent
from llm_client import ChatClient
from random_agent import RandomJetanAgent
from terminal_agent import TerminalUserAgent

AgentFactory = Callable[[int, str], JetanAgent]
ConfiguredAgentFactory = Callable[["AgentConfiguration"], JetanAgent]
AGENT_FACTORIES: dict[str, ConfiguredAgentFactory] = {}


@dataclass(frozen=True)
class AgentConfiguration:
    seed: int
    player_name: str
    depth: int = 1
    max_model_calls: int = 64
    client: ChatClient | None = None

    def __post_init__(self) -> None:
        if self.depth < 1:
            raise ValueError("depth must be at least one ply")
        if self.max_model_calls < 0:
            raise ValueError("max_model_calls cannot be negative")


def register_agent(agent_type: str, factory: AgentFactory) -> None:
    """Register an agent factory and expose it as a command-line option."""
    register_configured_agent(
        agent_type,
        lambda configuration: factory(
            configuration.seed, configuration.player_name
        ),
    )


def register_configured_agent(
    agent_type: str, factory: ConfiguredAgentFactory
) -> None:
    """Register a factory that can use depth and model configuration."""
    if not agent_type or not agent_type.isidentifier():
        raise ValueError("agent_type must be a nonempty Python identifier")
    if agent_type in AGENT_FACTORIES:
        raise ValueError(f"agent type is already registered: {agent_type}")
    AGENT_FACTORIES[agent_type] = factory


def available_agent_types() -> tuple[str, ...]:
    return tuple(sorted(AGENT_FACTORIES))


def create_agent(agent_type: str, seed: int, player_name: str) -> JetanAgent:
    """Construct a registered agent for one player slot."""
    return create_configured_agent(
        agent_type, AgentConfiguration(seed=seed, player_name=player_name)
    )


def create_configured_agent(
    agent_type: str, configuration: AgentConfiguration
) -> JetanAgent:
    """Construct an agent with experiment and model configuration."""
    try:
        factory = AGENT_FACTORIES[agent_type]
    except KeyError as error:
        available = ", ".join(available_agent_types())
        raise ValueError(
            f"unknown agent type {agent_type!r}; available types: {available}"
        ) from error
    return factory(configuration)


def _random_factory(seed: int, player_name: str) -> JetanAgent:
    return RandomJetanAgent(seed, f"{player_name} Random")


def _terminal_factory(seed: int, player_name: str) -> JetanAgent:
    del seed
    return TerminalUserAgent(f"{player_name} Terminal User")


register_agent("random", _random_factory)
register_agent("terminal", _terminal_factory)
