"""Required student tests for Jetan agents (U0-HW-05)."""

import unittest

from direct_llm_agent import DirectLLMAgent
from jetan import JetanBoard, Move, Piece, PieceType, Player
from llm_client import ScriptedClient
from llm_evaluation import LLMEvaluationFunction
from student_strategies import (
    choose_fallback,
    evaluate_position_1,
    evaluate_position_2,
    evaluate_position_3,
    parse_evaluation_response,
    parse_move_response,
)


class StudentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.initial_board = JetanBoard.initial()

    def test_evaluators_perspective_reversal_and_bounds(self) -> None:
        for eval_fn in (evaluate_position_1, evaluate_position_2, evaluate_position_3):
            orange_val = eval_fn(self.initial_board, Player.ORANGE)
            black_val = eval_fn(self.initial_board, Player.BLACK)
            self.assertAlmostEqual(orange_val, -black_val, places=3)
            self.assertAlmostEqual(orange_val, 0.0, places=3)
            self.assertTrue(-0.99 <= orange_val <= 0.99)
            self.assertTrue(-0.99 <= black_val <= 0.99)

    def test_evaluators_hand_checked_asymmetric_position(self) -> None:
        board = JetanBoard.from_pieces(
            {
                (4, 4): Piece(Player.ORANGE, PieceType.CHIEF),
                (5, 5): Piece(Player.ORANGE, PieceType.PANTHAN),
                (0, 0): Piece(Player.BLACK, PieceType.PANTHAN),
                (9, 9): Piece(Player.BLACK, PieceType.PRINCESS),
                (0, 9): Piece(Player.ORANGE, PieceType.PRINCESS),
            }
        )
        for eval_fn in (evaluate_position_1, evaluate_position_2, evaluate_position_3):
            val_orange = eval_fn(board, Player.ORANGE)
            val_black = eval_fn(board, Player.BLACK)
            self.assertGreater(val_orange, 0.0)
            self.assertLess(val_black, 0.0)
            self.assertAlmostEqual(val_orange, -val_black, places=3)

    def test_llm_evaluator_valid_parsing(self) -> None:
        self.assertEqual(parse_evaluation_response('{"score": 0.45}'), 0.45)
        self.assertEqual(parse_evaluation_response('{"score": -0.8}'), -0.8)
        self.assertEqual(parse_evaluation_response('{"score": 0}'), 0.0)

    def test_llm_evaluator_rejects_malformed_and_duplicate_keys(self) -> None:
        with self.assertRaises(ValueError):
            parse_evaluation_response('{"score": 0.5, "score": 0.2}')
        with self.assertRaises(ValueError):
            parse_evaluation_response('{"score": 0.5, "reason": "winning"}')
        with self.assertRaises(ValueError):
            parse_evaluation_response('```json\n{"score": 0.5}\n```')
        with self.assertRaises(ValueError):
            parse_evaluation_response('{\n"score": 0.5\n}')

    def test_llm_evaluator_rejects_out_of_range_and_nonfinite(self) -> None:
        with self.assertRaises(ValueError):
            parse_evaluation_response('{"score": 1.05}')
        with self.assertRaises(ValueError):
            parse_evaluation_response('{"score": -1.0}')
        with self.assertRaises(ValueError):
            parse_evaluation_response('{"score": "good"}')
        with self.assertRaises(ValueError):
            parse_evaluation_response('{"score": true}')

    def test_direct_move_parser_valid_and_illegal(self) -> None:
        board = self.initial_board
        actions = board.actions()
        first_move_str = str(actions[0])

        parsed = parse_move_response(f'{{"move": "{first_move_str}"}}', actions)
        self.assertEqual(parsed, actions[0])

        with self.assertRaises(ValueError):
            parse_move_response(f'{{"move": "{first_move_str}", "move": "{first_move_str}"}}', actions)

        with self.assertRaises(ValueError):
            parse_move_response(f'{{"move": "{first_move_str}", "comment": "good"}}', actions)

        with self.assertRaises(ValueError):
            parse_move_response('{"move": "a0-a9"}', actions)

    def test_llm_evaluation_fallback_with_scripted_client(self) -> None:
        client = ScriptedClient(('{"score": 2.5}',))  # Out of range, should trigger fallback
        eval_fn = LLMEvaluationFunction(
            client,
            lambda b, p: ({"role": "user", "content": "eval"},),
            parse_evaluation_response,
            fallback=evaluate_position_1,
        )
        val = eval_fn(self.initial_board, Player.ORANGE)
        self.assertEqual(val, evaluate_position_1(self.initial_board, Player.ORANGE))
        self.assertEqual(eval_fn.fallback_calls, 1)

    def test_direct_llm_fallback_on_invalid_response(self) -> None:
        client = ScriptedClient(('{"invalid": "json"}',))
        agent = DirectLLMAgent(
            client,
            lambda b, a, h: ({"role": "user", "content": "move"},),
            parse_move_response,
            fallback=choose_fallback,
        )
        move = agent.choose_action(self.initial_board)
        self.assertIn(move, self.initial_board.actions())
        self.assertEqual(agent.fallback_calls, 1)


if __name__ == "__main__":
    unittest.main()