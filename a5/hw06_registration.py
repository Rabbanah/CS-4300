"""Register Assignment 6 alpha-beta agents with the Assignment 5 framework."""

from agent_factory import AgentConfiguration, register_configured_agent
from alpha_beta_agent import AlphaBetaAgent
from alpha_beta_ordering import order_actions_1, order_actions_2
import hw06_config


def _register(agent_type: str, display_name: str, ordering: object) -> None:
    def build(configuration: AgentConfiguration) -> AlphaBetaAgent:
        return AlphaBetaAgent(
            f"{configuration.player_name} {display_name}",
            configuration.depth,
            hw06_config.SELECTED_EVALUATOR,
            ordering,  # type: ignore[arg-type]
        )

    register_configured_agent(agent_type, build)


_register("alpha_beta_order_1", "Alpha-Beta Order 1", order_actions_1)
_register("alpha_beta_order_2", "Alpha-Beta Order 2", order_actions_2)
