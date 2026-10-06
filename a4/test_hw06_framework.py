"""Initially passing tests for supplied Assignment 6 framework and support."""

from __future__ import annotations

import unittest
import time
from dataclasses import dataclass
from unittest.mock import patch

import alpha_beta_ordering
from alpha_beta_agent import AlphaBetaAgent, SearchMetrics
from hw06_measured_minimax import MeasuredMinimaxAgent
from hw06_positions import POSITIONS
from hw06_run_experiments import (
    METRIC_NAMES,
    NA,
    RELATIVE_CHANGE_FIELD,
    _run_search,
    main,
    run_matrix,
    summarize,
)
from jetan import JetanBoard, Move, Piece, PieceType, Player


@dataclass(frozen=True)
class ToyBoard:
    node: str
    tree: dict[str, tuple[Player, dict[str, str], float | None]]

    def player(self) -> Player:
        return self.tree[self.node][0]

    def actions(self) -> tuple[str, ...]:
        return tuple(self.tree[self.node][1])

    def result(self, action: str) -> ToyBoard:
        return ToyBoard(self.tree[self.node][1][action], self.tree)

    def is_terminal(self) -> bool:
        return self.tree[self.node][2] is not None

    def utility(self, perspective: Player) -> float:
        value = self.tree[self.node][2]
        if value is None:
            raise ValueError("nonterminal state")
        return value if perspective is Player.ORANGE else -value


def toy_evaluator(board: ToyBoard, perspective: Player) -> float:
    del board, perspective
    return 0.0


class FrameworkTests(unittest.TestCase):
    def test_measured_minimax_and_metrics(self) -> None:
        tree = {
            "root": (Player.ORANGE, {"a": "a", "b": "b"}, None),
            "a": (Player.BLACK, {"a1": "a1", "a2": "a2"}, None),
            "b": (Player.BLACK, {"b1": "b1", "b2": "b2"}, None),
            "a1": (Player.ORANGE, {}, 0.3),
            "a2": (Player.ORANGE, {}, 0.5),
            "b1": (Player.ORANGE, {}, 0.2),
            "b2": (Player.ORANGE, {}, 0.4),
        }
        agent = MeasuredMinimaxAgent("measured", 2, toy_evaluator)  # type: ignore[arg-type]
        self.assertEqual(agent.choose_action(ToyBoard("root", tree)), "a")  # type: ignore[arg-type]
        self.assertAlmostEqual(agent.last_value, 0.3)
        metrics = agent.last_metrics
        self.assertEqual(
            (metrics.visited_states, metrics.expanded_nodes,
             metrics.generated_actions, metrics.evaluated_states,
             metrics.pruned_actions, metrics.maximum_depth),
            (7, 3, 6, 4, 0, 2),
        )

    def test_metric_fields_aliases_and_depth_guards(self) -> None:
        self.assertEqual(
            METRIC_NAMES,
            ("visited_states", "expanded_nodes", "generated_actions",
             "evaluated_states", "pruned_actions", "maximum_depth",
             "thinking_time"),
        )
        metrics = SearchMetrics(1, 2, 3, 4, 5, 6, 0.1)
        self.assertEqual(metrics.generated, 3)
        self.assertEqual(metrics.evaluated, 4)
        with self.assertRaises(ValueError):
            MeasuredMinimaxAgent("bad", 0, toy_evaluator)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            AlphaBetaAgent("bad", 0, toy_evaluator, lambda b, a, p: a)  # type: ignore[arg-type]

    def test_six_positions_use_existing_engine(self) -> None:
        self.assertEqual(len(POSITIONS), 6)
        self.assertEqual(len({position.identifier for position in POSITIONS}), 6)
        for position in POSITIONS:
            self.assertIsInstance(position.board, JetanBoard)
            self.assertFalse(position.board.is_terminal(), position.identifier)
            self.assertTrue(position.board.actions(), position.identifier)
        capture = JetanBoard.from_pieces({
            (4, 4): Piece(Player.ORANGE, PieceType.PANTHAN),
            (5, 5): Piece(Player.BLACK, PieceType.PRINCESS),
        })
        terminal = capture.result(Move((4, 4), (5, 5)))
        self.assertEqual(terminal.utility(Player.ORANGE), 1)

    def test_ordering_support_does_not_regenerate_and_preserves_ties(self) -> None:
        board = POSITIONS[0].board
        canonical = board.actions()
        with patch.object(JetanBoard, "actions", side_effect=AssertionError("must not call")):
            with patch.object(alpha_beta_ordering, "ordering_key_1", return_value=0):
                ordered = alpha_beta_ordering.order_actions_1(
                    board, canonical, board.player()
                )
        self.assertEqual(ordered, canonical)

    def test_matrix_shape_and_status_summary(self) -> None:
        def fake_run(stage, position, depth, agent, repetition, timeout):
            del timeout
            return {
                "stage": stage, "position": position.identifier,
                "category": position.category, "depth": depth, "agent": agent,
                "repetition": repetition, "status": "error",
                "selected_move": NA, "root_value": NA,
                **{name: NA for name in METRIC_NAMES},
                "error_type": "Fixture", "error_message": "expected",
            }

        with patch("hw06_run_experiments._run_search", side_effect=fake_run):
            rows = run_matrix()
        stage_a = [row for row in rows if row["stage"] == "A"]
        stage_b = [row for row in rows if row["stage"] == "B"]
        self.assertEqual((len(stage_a), len(stage_b)), (18, 36))
        self.assertTrue(all(row["depth"] == 2 for row in stage_a))
        self.assertTrue(all(row["depth"] == 3 for row in stage_b))
        summary = summarize(rows)
        self.assertEqual(len(summary), 30)
        self.assertTrue(all(row["errors"] for row in summary))

    def test_filtered_matrix_shapes_and_incompatible_filter(self) -> None:
        with patch("hw06_run_experiments._run_search", return_value={}):
            one = run_matrix(
                repeats=1, stage="B", position_identifier="reduced",
                agent_name="alpha_beta_order_1",
            )
            trial = run_matrix(repeats=1, stage="B")
        self.assertEqual(len(one), 1)
        self.assertEqual(len(trial), 12)
        with self.assertRaisesRegex(ValueError, "only in Stage A"):
            run_matrix(stage="B", agent_name="measured_minimax")

    def test_summary_stable_results_and_relative_change_signs(self) -> None:
        rows = []
        counts = {
            "initial": (10, 8),
            "early-a": (10, 10),
            "early-b": (10, 12),
        }
        for position, (order_1, order_2) in counts.items():
            for agent, expanded in (
                ("alpha_beta_order_1", order_1),
                ("alpha_beta_order_2", order_2),
            ):
                for repetition in (1, 2, 3):
                    rows.append({
                        "stage": "B", "position": position,
                        "category": "fixture", "depth": 3, "agent": agent,
                        "repetition": repetition, "status": "ok",
                        "selected_move": "fixture-move", "root_value": 0.25,
                        **{
                            name: (0.01 * repetition if name == "thinking_time"
                                   else expanded if name == "expanded_nodes" else 1)
                            for name in METRIC_NAMES
                        },
                        "error_type": NA, "error_message": NA,
                    })
        summary = summarize(rows)
        self.assertEqual(len(summary), 6)
        self.assertTrue(all(row["selected_move"] == "fixture-move" for row in summary))
        self.assertTrue(all(row["root_value"] == 0.25 for row in summary))
        self.assertTrue(all(row["thinking_time"] == 0.02 for row in summary))
        changes = {
            row["position"]: row[RELATIVE_CHANGE_FIELD]
            for row in summary
        }
        self.assertGreater(changes["initial"], 0)
        self.assertEqual(changes["early-a"], 0)
        self.assertLess(changes["early-b"], 0)

    def test_search_timeout_terminates_child_process(self) -> None:
        started = time.perf_counter()
        row = _run_search(
            "A", POSITIONS[0], 2, "measured_minimax", 1, 0.001
        )
        self.assertEqual(row["status"], "timeout")
        self.assertEqual(row["error_type"], "SearchTimeout")
        self.assertLess(time.perf_counter() - started, 5.0)

    def test_cli_rejects_zero_timeout(self) -> None:
        with patch("sys.argv", ["hw06_run_experiments.py", "--timeout", "0"]):
            with self.assertRaises(SystemExit):
                main()

    def test_registration_uses_exact_public_names(self) -> None:
        import hw06_registration  # noqa: F401
        from agent_factory import available_agent_types

        registered = available_agent_types()
        self.assertIn("alpha_beta_order_1", registered)
        self.assertIn("alpha_beta_order_2", registered)


if __name__ == "__main__":
    unittest.main()
