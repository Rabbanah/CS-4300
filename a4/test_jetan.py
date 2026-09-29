"""Tests for the provided Jetan board and rules."""

import unittest

from jetan import JetanBoard, Move, Piece, PieceType, Player


class JetanBoardTests(unittest.TestCase):
    def test_initial_board_has_two_complete_armies(self) -> None:
        board = JetanBoard.initial()

        self.assertEqual(len(list(board.pieces(Player.ORANGE))), 20)
        self.assertEqual(len(list(board.pieces(Player.BLACK))), 20)
        self.assertEqual(board.player(), Player.ORANGE)
        self.assertFalse(board.is_terminal())

    def test_board_labels_match_move_notation(self) -> None:
        board_text = str(JetanBoard.initial())

        self.assertIn("a b c d e f g h i j", board_text)
        self.assertEqual(str(Move((2, 1), (3, 2))), "c1-d2")

    def test_actions_are_ordered_and_owned_by_current_player(self) -> None:
        board = JetanBoard.initial()

        actions = board.actions()

        self.assertTrue(actions)
        self.assertEqual(actions, tuple(sorted(actions)))
        self.assertTrue(
            all(board.at(action.source).player is Player.ORANGE for action in actions)  # type: ignore[union-attr]
        )

    def test_board_reuses_cached_action_and_attack_results(self) -> None:
        board = JetanBoard.initial()
        equivalent = JetanBoard.initial()
        original_hash = hash(board)

        first_actions = board.actions()
        second_actions = board.actions()
        first_attacks = board.attacked_locations(Player.BLACK)
        second_attacks = board.attacked_locations(Player.BLACK)

        self.assertIs(first_actions, second_actions)
        self.assertIs(first_attacks, second_attacks)
        self.assertEqual(board, equivalent)
        self.assertEqual(hash(board), original_hash)

    def test_legal_action_count_matches_complete_action_tuple(self) -> None:
        board = JetanBoard.initial()

        self.assertEqual(
            board.legal_action_count(Player.ORANGE),
            len(board.legal_actions(Player.ORANGE)),
        )
        self.assertEqual(
            board.legal_action_count(Player.BLACK),
            len(board.legal_actions(Player.BLACK)),
        )
        self.assertTrue(
            all(
                board.at(action.source).player is Player.BLACK  # type: ignore[union-attr]
                for action in board.legal_actions(Player.BLACK)
            )
        )

    def test_result_returns_a_new_board_and_preserves_the_old_board(self) -> None:
        board = JetanBoard.initial()
        action = board.actions()[0]

        successor = board.result(action)

        self.assertIsNot(successor, board)
        self.assertIsNotNone(board.at(action.source))
        self.assertIsNone(successor.at(action.source))
        self.assertEqual(successor.at(action.destination), board.at(action.source))
        self.assertEqual(successor.player(), Player.BLACK)
        self.assertEqual(successor.ply, 1)

    def test_aima_state_argument_forms_match_board_methods(self) -> None:
        board = JetanBoard.initial()
        action = board.actions()[0]

        self.assertEqual(board.player(board), board.player())
        self.assertEqual(board.actions(board), board.actions())
        self.assertEqual(board.result(board, action), board.result(action))
        self.assertEqual(board.terminal_test(board), board.is_terminal())

    def test_capturing_a_princess_ends_the_game(self) -> None:
        board = JetanBoard.from_pieces(
            {
                (4, 4): Piece(Player.ORANGE, PieceType.PANTHAN),
                (5, 5): Piece(Player.BLACK, PieceType.PRINCESS),
            }
        )

        result = board.result(Move((4, 4), (5, 5)))

        self.assertTrue(result.is_terminal())
        self.assertEqual(result.winner, Player.ORANGE)
        self.assertEqual(result.utility(Player.ORANGE), 1)
        self.assertEqual(result.utility(result, Player.BLACK), -1)

    def test_princess_cannot_capture(self) -> None:
        board = JetanBoard.from_pieces(
            {
                (4, 4): Piece(Player.ORANGE, PieceType.PRINCESS),
                (5, 4): Piece(Player.BLACK, PieceType.PANTHAN),
            }
        )

        self.assertNotIn(Move((4, 4), (5, 4)), board.actions())

    def test_attacked_princess_can_use_one_long_escape(self) -> None:
        board = JetanBoard.from_pieces(
            {
                (5, 5): Piece(Player.ORANGE, PieceType.PRINCESS),
                (5, 6): Piece(Player.BLACK, PieceType.PANTHAN),
            }
        )
        escape = Move((5, 5), (0, 0))

        self.assertIn(escape, board.actions())
        result = board.result(escape)
        self.assertTrue(result.used_escape[Player.ORANGE - 1])

    def test_nonterminal_utility_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            JetanBoard.initial().utility(Player.ORANGE)

    def test_no_move_position_resolves_as_an_opponent_win(self) -> None:
        board = JetanBoard.from_pieces({}).resolve_terminal()

        self.assertTrue(board.terminal_test())
        self.assertEqual(board.winner, Player.BLACK)
        self.assertEqual(board.utility(Player.ORANGE), -1)


if __name__ == "__main__":
    unittest.main()
