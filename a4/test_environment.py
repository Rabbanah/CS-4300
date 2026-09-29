"""Tests for agent and viewer integration."""

import unittest
from unittest.mock import patch

from agents import (
    RandomJetanAgent,
    TerminalUserAgent,
    available_agent_types,
    create_agent,
)
from environment import JetanEnvironment
from jetan import JetanBoard, Move, Piece, PieceType, Player


class FirstAgent:
    def __init__(self, name: str) -> None:
        self.name = name
        self.boards: list[JetanBoard] = []

    def choose_action(self, board: JetanBoard) -> Move:
        self.boards.append(board)
        return board.actions()[0]


class ScriptedAgent(FirstAgent):
    def __init__(self, name: str, action: Move) -> None:
        super().__init__(name)
        self.action = action

    def choose_action(self, board: JetanBoard) -> Move:
        self.boards.append(board)
        return self.action


class RecordingViewer:
    def __init__(self) -> None:
        self.events: list[tuple[object, ...]] = []

    def start(self, board: JetanBoard, names: object) -> None:
        self.events.append(("start", board, names))

    def update(self, before: JetanBoard, action: Move, after: JetanBoard) -> None:
        self.events.append(("update", before, action, after))

    def close(self, board: JetanBoard) -> None:
        self.events.append(("close", board))


class EnvironmentTests(unittest.TestCase):
    def test_terminal_user_agent_reprompts_and_accepts_move_text(self) -> None:
        board = JetanBoard.initial()
        expected = board.actions()[0]
        responses = iter(("not-a-move", str(expected)))
        output: list[str] = []
        agent = TerminalUserAgent(
            "Orange Human", input_fn=lambda prompt: next(responses), output_fn=output.append
        )

        selected = agent.choose_action(board)

        self.assertEqual(selected, expected)
        self.assertTrue(any("Invalid selection" in line for line in output))

    def test_terminal_user_agent_accepts_numbered_selection(self) -> None:
        board = JetanBoard.initial()
        agent = TerminalUserAgent(
            "Orange Human", input_fn=lambda prompt: "1", output_fn=lambda text: None
        )

        self.assertEqual(agent.choose_action(board), board.actions()[0])

    def test_registered_agents_can_be_created_by_command_line_name(self) -> None:
        self.assertIn("random", available_agent_types())
        self.assertIn("terminal", available_agent_types())

        agent = create_agent("random", 17, "Orange")

        self.assertEqual(agent.name, "Orange Random")
        self.assertIn(agent.choose_action(JetanBoard.initial()), JetanBoard.initial().actions())

    def test_unknown_agent_type_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown agent type"):
            create_agent("missing", 0, "Orange")

    def test_random_agent_is_seeded_and_returns_a_legal_move(self) -> None:
        board = JetanBoard.initial()
        first = RandomJetanAgent(17).choose_action(board)
        second = RandomJetanAgent(17).choose_action(board)

        self.assertEqual(first, second)
        self.assertIn(first, board.actions())

    def test_environment_passes_board_instances_to_agent_and_viewer(self) -> None:
        board = JetanBoard.from_pieces(
            {
                (4, 4): Piece(Player.ORANGE, PieceType.PANTHAN),
                (5, 5): Piece(Player.BLACK, PieceType.PRINCESS),
            }
        )
        orange = ScriptedAgent("Orange", Move((4, 4), (5, 5)))
        black = FirstAgent("Black")
        viewer = RecordingViewer()
        environment = JetanEnvironment(orange, black, board=board, viewer=viewer)

        result = environment.run()

        self.assertEqual(result.winner, Player.ORANGE)
        self.assertEqual(orange.boards, [board])
        self.assertEqual([event[0] for event in viewer.events], ["start", "update", "close"])
        self.assertIs(viewer.events[1][1], board)
        self.assertIs(viewer.events[1][3], result.final_board)

    def test_move_limit_records_a_draw(self) -> None:
        environment = JetanEnvironment(
            FirstAgent("Orange"), FirstAgent("Black"), max_plies=1
        )

        result = environment.run()

        self.assertIsNone(result.winner)
        self.assertEqual(result.reason, "move_limit")
        self.assertTrue(result.final_board.draw)

    def test_environment_records_agent_think_time(self) -> None:
        board = JetanBoard.from_pieces(
            {
                (4, 4): Piece(Player.ORANGE, PieceType.PANTHAN),
                (5, 5): Piece(Player.BLACK, PieceType.PRINCESS),
            }
        )
        environment = JetanEnvironment(
            ScriptedAgent("Orange", Move((4, 4), (5, 5))),
            FirstAgent("Black"),
            board=board,
        )

        with patch("environment.time.perf_counter", side_effect=(10.0, 10.25)):
            result = environment.run()

        self.assertEqual(result.reason, "win")
        self.assertEqual(result.orange_think_seconds, 0.25)
        self.assertEqual(result.black_think_seconds, 0.0)
        self.assertEqual(result.utility(Player.ORANGE), 1)
        self.assertEqual(result.utility(Player.BLACK), -1)

    def test_cumulative_time_limit_forfeits_unplayed_action(self) -> None:
        viewer = RecordingViewer()
        environment = JetanEnvironment(
            FirstAgent("Orange"),
            FirstAgent("Black"),
            viewer=viewer,
            max_think_seconds=1.0,
        )

        clock = (0.0, 0.6, 0.6, 0.8, 0.8, 1.3)
        with patch("environment.time.perf_counter", side_effect=clock):
            result = environment.run()

        self.assertEqual(result.reason, "time_limit")
        self.assertEqual(result.winner, Player.BLACK)
        self.assertEqual(result.plies, 2)
        self.assertAlmostEqual(result.orange_think_seconds, 1.1)
        self.assertAlmostEqual(result.black_think_seconds, 0.2)
        self.assertEqual(
            [event[0] for event in viewer.events],
            ["start", "update", "update", "close"],
        )

    def test_nonpositive_or_nonfinite_time_limit_is_rejected(self) -> None:
        for limit in (0.0, -1.0, float("inf"), float("nan")):
            with self.subTest(limit=limit):
                with self.assertRaisesRegex(ValueError, "positive and finite"):
                    JetanEnvironment(
                        FirstAgent("Orange"),
                        FirstAgent("Black"),
                        max_think_seconds=limit,
                    )


if __name__ == "__main__":
    unittest.main()
