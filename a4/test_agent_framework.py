"""Public tests for the supplied agent framework."""

import unittest
from unittest.mock import patch

from agent_factory import (
    AGENT_FACTORIES,
    AgentConfiguration,
    create_configured_agent,
    register_configured_agent,
)
from agent_builders import (
    register_direct_llm_agent,
    register_llm_minimax_agent,
    register_minimax_agent,
)
from direct_llm_agent import DirectLLMAgent
from jetan import JetanBoard, Move, Piece, PieceType, Player
from llm_client import ScriptedClient
from llm_evaluation import LLMEvaluationFunction
from llm_minimax_agent import LLMMinimaxAgent
from minimax_agent import DepthLimitedMinimaxAgent
from play import endpoint_is_allowed, format_agent_metrics


class AgentFrameworkTests(unittest.TestCase):
    def test_minimax_uses_terminal_utility_before_cutoff_evaluation(self) -> None:
        board = JetanBoard.from_pieces(
            {
                (4, 4): Piece(Player.ORANGE, PieceType.PANTHAN),
                (5, 5): Piece(Player.BLACK, PieceType.PRINCESS),
                (9, 9): Piece(Player.BLACK, PieceType.PANTHAN),
            }
        )

        def reject_cutoff(state: JetanBoard, perspective: Player) -> float:
            del state, perspective
            return -0.5

        agent = DepthLimitedMinimaxAgent("Test Minimax", 1, reject_cutoff)

        self.assertEqual(agent.choose_action(board), Move((4, 4), (5, 5)))
        self.assertEqual(agent.last_metrics.maximum_depth, 1)
        self.assertGreater(agent.last_metrics.evaluated, 0)

    def test_minimax_accumulates_metrics_across_moves(self) -> None:
        board = JetanBoard.from_pieces(
            {
                (1, 1): Piece(Player.ORANGE, PieceType.PANTHAN),
                (8, 8): Piece(Player.BLACK, PieceType.PANTHAN),
                (9, 9): Piece(Player.BLACK, PieceType.PRINCESS),
            }
        )
        agent = DepthLimitedMinimaxAgent(
            "Test Minimax", 1, lambda state, perspective: 0.0
        )

        agent.choose_action(board)
        first = agent.last_metrics
        agent.choose_action(board)

        self.assertEqual(agent.total_metrics.generated, 2 * first.generated)
        self.assertEqual(agent.total_metrics.evaluated, 2 * first.evaluated)
        self.assertEqual(agent.total_metrics.maximum_depth, 1)

    def test_minimax_rejects_nonfinite_evaluation(self) -> None:
        agent = DepthLimitedMinimaxAgent(
            "Test Minimax", 1, lambda state, perspective: float("nan")
        )

        with self.assertRaisesRegex(ValueError, "finite"):
            agent.choose_action(JetanBoard.initial())

    def test_llm_evaluation_uses_student_callbacks_and_caches_result(self) -> None:
        board = JetanBoard.initial()
        client = ScriptedClient(("score response",))
        prompt_calls: list[tuple[JetanBoard, Player]] = []

        def build_prompt(
            state: JetanBoard, perspective: Player
        ) -> tuple[dict[str, str], ...]:
            prompt_calls.append((state, perspective))
            return ({"role": "user", "content": "evaluate"},)

        def parse_response(response: str) -> float:
            self.assertEqual(response, "score response")
            return 0.25

        evaluation = LLMEvaluationFunction(
            client,
            build_prompt,
            parse_response,
            fallback=lambda state, perspective: 0.0,
        )

        self.assertEqual(evaluation(board, Player.ORANGE), 0.25)
        self.assertEqual(evaluation(board, Player.ORANGE), 0.25)
        self.assertEqual(prompt_calls, [(board, Player.ORANGE)])
        self.assertEqual(evaluation.model_calls, 1)
        self.assertEqual(evaluation.cache_hits, 1)

    def test_llm_evaluation_uses_fallback_after_validation_failure(self) -> None:
        board = JetanBoard.initial()

        def reject_response(response: str) -> float:
            raise ValueError(f"invalid response: {response}")

        evaluation = LLMEvaluationFunction(
            ScriptedClient(("invalid",)),
            lambda state, perspective: ({"role": "user", "content": "evaluate"},),
            reject_response,
            fallback=lambda state, perspective: -0.2,
        )

        self.assertEqual(evaluation(board, Player.ORANGE), -0.2)
        self.assertEqual(evaluation.model_calls, 1)
        self.assertEqual(evaluation.fallback_calls, 1)

    def test_llm_evaluation_rejects_nonfinite_values(self) -> None:
        board = JetanBoard.initial()
        evaluation = LLMEvaluationFunction(
            ScriptedClient(("nan",)),
            lambda state, perspective: ({"role": "user", "content": "evaluate"},),
            float,
            fallback=lambda state, perspective: -0.2,
        )

        self.assertEqual(evaluation(board, Player.ORANGE), -0.2)
        self.assertEqual(evaluation.total_fallback_calls, 1)

    def test_llm_evaluation_budget_uses_fallback_without_model_request(self) -> None:
        board = JetanBoard.initial()
        client = ScriptedClient(())
        evaluation = LLMEvaluationFunction(
            client,
            lambda state, perspective: ({"role": "user", "content": "evaluate"},),
            float,
            fallback=lambda state, perspective: 0.1,
            max_model_calls=0,
        )

        self.assertEqual(evaluation(board, Player.ORANGE), 0.1)
        self.assertEqual(evaluation.model_calls, 0)
        self.assertEqual(evaluation.fallback_calls, 1)
        self.assertEqual(client.messages_seen, [])

    def test_llm_evaluation_rejects_nonfinite_fallback(self) -> None:
        evaluation = LLMEvaluationFunction(
            ScriptedClient(()),
            lambda state, perspective: ({"role": "user", "content": "evaluate"},),
            float,
            fallback=lambda state, perspective: float("inf"),
            max_model_calls=0,
        )

        with self.assertRaisesRegex(ValueError, "finite"):
            evaluation(JetanBoard.initial(), Player.ORANGE)

    def test_llm_minimax_resets_per_search_model_metrics(self) -> None:
        board = JetanBoard.from_pieces(
            {
                (1, 1): Piece(Player.ORANGE, PieceType.PANTHAN),
                (8, 8): Piece(Player.BLACK, PieceType.PANTHAN),
                (9, 9): Piece(Player.BLACK, PieceType.PRINCESS),
            }
        )
        responses = ("0",) * len(board.actions())
        evaluation = LLMEvaluationFunction(
            ScriptedClient(responses),
            lambda state, perspective: ({"role": "user", "content": "evaluate"},),
            float,
            fallback=lambda state, perspective: 0.0,
        )
        agent = LLMMinimaxAgent("Test LLM Minimax", 1, evaluation)

        self.assertEqual(agent.choose_action(board), board.actions()[0])
        self.assertEqual(evaluation.model_calls, len(board.actions()))

    def test_direct_llm_agent_delegates_prompt_and_response_strategy(self) -> None:
        board = JetanBoard.initial()
        expected = board.actions()[-1]
        client = ScriptedClient((str(expected),))

        def build_prompt(
            state: JetanBoard,
            actions: tuple[Move, ...],
            history: tuple[Move, ...],
        ) -> tuple[dict[str, str], ...]:
            self.assertIs(state, board)
            self.assertEqual(actions, board.actions())
            self.assertEqual(history, ())
            return ({"role": "user", "content": "choose"},)

        def parse_move(response: str, actions: tuple[Move, ...]) -> Move:
            return {str(action): action for action in actions}[response]

        agent = DirectLLMAgent(client, build_prompt, parse_move)

        self.assertEqual(agent.choose_action(board), expected)
        self.assertEqual(agent.model_calls, 1)
        self.assertEqual(agent.fallback_calls, 0)
        self.assertEqual(agent.recent_moves, (expected,))

    def test_direct_llm_agent_rejects_an_illegal_parsed_move(self) -> None:
        board = JetanBoard.initial()
        agent = DirectLLMAgent(
            ScriptedClient(("invalid",)),
            lambda state, actions, history: (
                {"role": "user", "content": "choose"},
            ),
            lambda response, actions: Move((0, 0), (0, 0)),
        )

        self.assertEqual(agent.choose_action(board), board.actions()[0])
        self.assertEqual(agent.fallback_calls, 1)

    def test_configured_factory_receives_runner_configuration(self) -> None:
        seen: list[AgentConfiguration] = []

        def build(configuration: AgentConfiguration) -> DepthLimitedMinimaxAgent:
            seen.append(configuration)
            return DepthLimitedMinimaxAgent(
                configuration.player_name,
                configuration.depth,
                lambda state, perspective: 0.0,
            )

        configuration = AgentConfiguration(7, "Orange", depth=2)
        with patch.dict(AGENT_FACTORIES, {}, clear=True):
            register_configured_agent("student_agent", build)
            agent = create_configured_agent("student_agent", configuration)

        self.assertEqual(seen, [configuration])
        self.assertEqual(agent.depth, 2)

    def test_supplied_builders_register_all_student_modalities(self) -> None:
        prompt = lambda state, perspective: (
            {"role": "user", "content": "evaluate"},
        )
        move_prompt = lambda state, actions, history: (
            {"role": "user", "content": "choose"},
        )
        with patch.dict(AGENT_FACTORIES, {}, clear=True):
            register_minimax_agent(
                "minimax_student", lambda state, player: 0.0, "M"
            )
            register_llm_minimax_agent(
                "llm_eval_student", prompt, float, lambda state, player: 0.0, "L"
            )
            register_direct_llm_agent(
                "llm_move_student",
                move_prompt,
                lambda response, actions: actions[0],
                "D",
            )
            offline = AgentConfiguration(1, "Orange", depth=2)
            live = AgentConfiguration(
                1, "Orange", depth=2, client=ScriptedClient(("0",))
            )

            minimax = create_configured_agent("minimax_student", offline)
            llm_eval = create_configured_agent("llm_eval_student", live)
            llm_move = create_configured_agent("llm_move_student", live)

        self.assertIsInstance(minimax, DepthLimitedMinimaxAgent)
        self.assertIsInstance(llm_eval, LLMMinimaxAgent)
        self.assertIsInstance(llm_move, DirectLLMAgent)

    def test_runner_exposes_search_and_model_metrics(self) -> None:
        agent = DepthLimitedMinimaxAgent(
            "Measured", 1, lambda state, perspective: 0.0
        )
        agent.choose_action(JetanBoard.initial())

        rendered = format_agent_metrics("orange", agent)

        self.assertIn("generated=", rendered)
        self.assertIn("evaluated=", rendered)
        self.assertIn("maximum_depth=1", rendered)

    def test_live_endpoint_policy(self) -> None:
        self.assertTrue(endpoint_is_allowed("http://golem:8000/v1"))
        self.assertTrue(endpoint_is_allowed("http://localhost:8000/v1"))
        self.assertTrue(endpoint_is_allowed("https://models.example.edu/v1"))
        self.assertFalse(endpoint_is_allowed("http://models.example.edu/v1"))


if __name__ == "__main__":
    unittest.main()
