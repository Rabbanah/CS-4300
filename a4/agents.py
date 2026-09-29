"""Public aggregation point for Jetan agents and student extension interfaces."""

from agent_factory import (
    AGENT_FACTORIES,
    AgentConfiguration,
    AgentFactory,
    ConfiguredAgentFactory,
    available_agent_types,
    create_agent,
    create_configured_agent,
    register_agent,
    register_configured_agent,
)
from agent_builders import (
    register_direct_llm_agent,
    register_llm_minimax_agent,
    register_minimax_agent,
)
from agent_protocol import JetanAgent
from direct_llm_agent import (
    DirectLLMAgent,
    FallbackPolicy,
    MovePromptBuilder,
    MoveResponseParser,
    first_legal_action,
)
from evaluations import EvaluationFunction
from llm_evaluation import (
    EvaluationPromptBuilder,
    EvaluationResponseParser,
    LLMEvaluationFunction,
)
from llm_minimax_agent import LLMMinimaxAgent
from minimax_agent import DepthLimitedMinimaxAgent, SearchMetrics
from random_agent import RandomJetanAgent
from terminal_agent import TerminalUserAgent

__all__ = [
    "AGENT_FACTORIES",
    "AgentConfiguration",
    "AgentFactory",
    "ConfiguredAgentFactory",
    "DepthLimitedMinimaxAgent",
    "DirectLLMAgent",
    "EvaluationFunction",
    "EvaluationPromptBuilder",
    "EvaluationResponseParser",
    "FallbackPolicy",
    "JetanAgent",
    "LLMEvaluationFunction",
    "LLMMinimaxAgent",
    "MovePromptBuilder",
    "MoveResponseParser",
    "RandomJetanAgent",
    "SearchMetrics",
    "TerminalUserAgent",
    "available_agent_types",
    "create_agent",
    "create_configured_agent",
    "first_legal_action",
    "register_agent",
    "register_configured_agent",
    "register_direct_llm_agent",
    "register_llm_minimax_agent",
    "register_minimax_agent",
]

# Importing this module performs student registrations without requiring edits
# to the supplied aggregation or command-line runner.
import student_agents as _student_agents  # noqa: E402,F401
