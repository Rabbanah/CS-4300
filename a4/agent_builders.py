"""Register common agent compositions without writing custom factories."""

from agent_factory import AgentConfiguration, register_configured_agent
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
from minimax_agent import DepthLimitedMinimaxAgent


def register_minimax_agent(
    agent_type: str,
    evaluation_function: EvaluationFunction,
    display_name: str,
) -> None:
    """Register minimax with a student-designed deterministic evaluator."""

    def build(configuration: AgentConfiguration) -> DepthLimitedMinimaxAgent:
        return DepthLimitedMinimaxAgent(
            f"{configuration.player_name} {display_name}",
            configuration.depth,
            evaluation_function,
        )

    register_configured_agent(agent_type, build)


def register_llm_minimax_agent(
    agent_type: str,
    prompt_builder: EvaluationPromptBuilder,
    response_parser: EvaluationResponseParser,
    fallback: EvaluationFunction,
    display_name: str,
) -> None:
    """Register minimax with student-designed LLM evaluation callbacks."""

    def build(configuration: AgentConfiguration) -> LLMMinimaxAgent:
        if configuration.client is None:
            raise RuntimeError(f"{agent_type} requires --live")
        evaluation = LLMEvaluationFunction(
            configuration.client,
            prompt_builder,
            response_parser,
            fallback,
            configuration.max_model_calls,
        )
        return LLMMinimaxAgent(
            f"{configuration.player_name} {display_name}",
            configuration.depth,
            evaluation,
        )

    register_configured_agent(agent_type, build)


def register_direct_llm_agent(
    agent_type: str,
    prompt_builder: MovePromptBuilder,
    response_parser: MoveResponseParser,
    display_name: str,
    fallback: FallbackPolicy = first_legal_action,
    history_limit: int = 6,
) -> None:
    """Register direct move choice with student-designed LLM callbacks."""

    def build(configuration: AgentConfiguration) -> DirectLLMAgent:
        if configuration.client is None:
            raise RuntimeError(f"{agent_type} requires --live")
        return DirectLLMAgent(
            configuration.client,
            prompt_builder,
            response_parser,
            f"{configuration.player_name} {display_name}",
            fallback,
            history_limit,
        )

    register_configured_agent(agent_type, build)
