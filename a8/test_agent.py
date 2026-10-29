"""Unit tests for `agent.py` derived from SPECIFICATIONS.md (SPEC-REACT-1..2)."""

from __future__ import annotations

import unittest

from agent import ReActSudokuAgent, TerminalStatus
from controller import ControllerConfig
from llm_client import RuleBasedReActClient, ScriptedClient
from scenarios import SCENARIOS


class TestReActSudokuAgentSpec(unittest.TestCase):
    """Black-box and supplementary white-box tests for ReActSudokuAgent."""

    def test_scripted_agent_solves_near_complete_puzzle(self) -> None:
        board = SCENARIOS["near-complete"].make_board()
        script = ScriptedClient(
            [
                '{"thought": "Inspect row 7", "tool": "inspect", "target": "row", "row": 7}',
                '{"thought": "Record fact for (7,8)=5", "tool": "kb_add", "kind": "fact", "formula": "c_r7_c8_d5"}',
                '{"thought": "Query (7,8)=5", "tool": "kb_query", "formula": "c_r7_c8_d5"}',
                '{"thought": "Place 5 at (7,8)", "tool": "place", "row": 7, "col": 8, "digit": 5}',
                '{"thought": "Place 1 at (8,6)", "tool": "place", "row": 8, "col": 6, "digit": 1}',
                '{"thought": "Place 7 at (8,7)", "tool": "place", "row": 8, "col": 7, "digit": 7}',
                '{"thought": "Submit solved board", "tool": "submit"}',
            ]
        )
        agent = ReActSudokuAgent(script, config=ControllerConfig(max_steps=15))
        result = agent.solve(board)

        self.assertTrue(result.solved)
        self.assertEqual(result.terminal_status, TerminalStatus.SOLVED)
        self.assertEqual(result.metrics.steps_taken, 7)
        self.assertEqual(result.metrics.placements, 3)
        self.assertEqual(result.metrics.kb_adds, 1)
        self.assertEqual(result.metrics.kb_queries, 1)
        self.assertEqual(result.metrics.inspections, 1)
        self.assertEqual(result.metrics.submissions, 1)

    def test_model_text_cannot_establish_success_without_valid_submit(self) -> None:
        board = SCENARIOS["near-complete"].make_board()
        script = ScriptedClient(
            [
                "SOLVED! The puzzle is complete.",
                '{"tool": "submit"}',
            ]
        )
        agent = ReActSudokuAgent(
            script,
            config=ControllerConfig(max_steps=2, stop_on_invalid_submit=True),
        )
        result = agent.solve(board)
        self.assertFalse(result.solved)
        self.assertEqual(result.terminal_status, TerminalStatus.INVALID_SUBMISSION)

    def test_undo_recovery_in_react_loop(self) -> None:
        board = SCENARIOS["branching-1"].make_board()
        # (0, 3) is empty in branching-1 with candidates [6, 7].
        script = ScriptedClient(
            [
                '{"thought": "Try 6 at (0,3)", "tool": "place", "row": 0, "col": 3, "digit": 6}',
                '{"thought": "Undo placement at (0,3)", "tool": "undo", "steps": 1}',
                '{"thought": "Inspect cell (0,3)", "tool": "inspect", "target": "cell", "row": 0, "col": 3}',
            ]
        )
        agent = ReActSudokuAgent(script, config=ControllerConfig(max_steps=3))
        result = agent.solve(board)
        self.assertFalse(result.solved)
        self.assertEqual(result.terminal_status, TerminalStatus.ACTION_LIMIT_REACHED)
        self.assertEqual(result.metrics.placements, 1)
        self.assertEqual(result.metrics.undos, 1)
        self.assertEqual(result.final_board[0][3], 0)

    def test_ordered_terminal_check_rejection_limit(self) -> None:
        board = SCENARIOS["near-complete"].make_board()
        script = ScriptedClient(["invalid json"] * 5)
        agent = ReActSudokuAgent(
            script,
            config=ControllerConfig(max_steps=10, max_consecutive_rejections=3),
        )
        result = agent.solve(board)
        self.assertFalse(result.solved)
        self.assertEqual(result.terminal_status, TerminalStatus.REJECTION_LIMIT_REACHED)
        self.assertEqual(result.metrics.steps_taken, 3)

    def test_client_exception_does_not_leak_sensitive_text(self) -> None:
        board = SCENARIOS["near-complete"].make_board()
        secret = " Bearer sk-secret-credential-999 "
        script = ScriptedClient(
            [
                RuntimeError(f"Connection failed with header {secret}"),
                RuntimeError(f"Connection failed with header {secret}"),
            ]
        )
        agent = ReActSudokuAgent(
            script,
            config=ControllerConfig(max_steps=5, max_consecutive_rejections=2),
        )
        result = agent.solve(board)
        self.assertFalse(result.solved)
        self.assertEqual(result.terminal_status, TerminalStatus.CLIENT_FAILURE_LIMIT)
        self.assertEqual(result.metrics.client_failures, 2)
        for step in result.trace:
            self.assertNotIn("sk-secret-credential-999", step.observation.message)
            self.assertNotIn("sk-secret-credential-999", step.observation.rejection_reason or "")

    def test_rule_based_react_client_solves_all_scenarios(self) -> None:
        for name, scenario in SCENARIOS.items():
            with self.subTest(scenario=name):
                client = RuleBasedReActClient()
                agent = ReActSudokuAgent(
                    client,
                    config=ControllerConfig(max_steps=120),
                )
                result = agent.solve(scenario.make_board())
                self.assertTrue(result.solved, f"Failed to solve {name}: {result.terminal_reason}")
                self.assertEqual(result.terminal_status, TerminalStatus.SOLVED)
                self.assertGreaterEqual(result.metrics.kb_adds, 1)
                self.assertGreaterEqual(result.metrics.kb_queries, 1)
                if name == "branching-1":
                    self.assertGreaterEqual(result.metrics.undos, 1)


if __name__ == "__main__":
    unittest.main()
