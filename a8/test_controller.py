"""Unit tests for `controller.py` derived from SPECIFICATIONS.md (SPEC-CTRL-1..6)."""

from __future__ import annotations

import unittest

from controller import ControllerConfig, SudokuController
from scenarios import SCENARIOS


class TestSudokuControllerSpec(unittest.TestCase):
    """Black-box and supplementary white-box tests for SudokuController policies."""

    def setUp(self) -> None:
        self.board = SCENARIOS["near-complete"].make_board()

    # --- SPEC-CTRL-1: Action Budget Accounting ---

    def test_action_budget_decrements_on_every_request(self) -> None:
        ctrl = SudokuController(self.board, config=ControllerConfig(max_steps=3))
        self.assertEqual(ctrl.remaining_budget, 3)

        _, obs1 = ctrl.step_raw('{"tool": "inspect", "target": "cell", "row": 7, "col": 8}')
        self.assertTrue(obs1.accepted)
        self.assertEqual(obs1.remaining_budget, 2)

        # Malformed request still consumes 1 action from budget
        _, obs2 = ctrl.step_raw("not json")
        self.assertFalse(obs2.accepted)
        self.assertEqual(obs2.remaining_budget, 1)

        _, obs3 = ctrl.step_raw('{"tool": "place", "row": 7, "col": 8, "digit": 5}')
        self.assertTrue(obs3.accepted)
        self.assertEqual(obs3.remaining_budget, 0)

        # Further calls after budget is 0 are rejected as terminal
        _, obs4 = ctrl.step_raw('{"tool": "submit"}')
        self.assertFalse(obs4.accepted)
        self.assertTrue(obs4.terminal)
        self.assertIn("action limit", (obs4.terminal_reason or "").lower())

    # --- SPEC-CTRL-2: Immutable Clue Protection ---

    def test_controller_rejects_modifying_given_clue(self) -> None:
        ctrl = SudokuController(self.board)
        _, obs = ctrl.step_raw('{"tool": "place", "row": 0, "col": 0, "digit": 9}')
        self.assertFalse(obs.accepted)
        self.assertIn("given", (obs.rejection_reason or "").lower())
        self.assertEqual(self.board.get_cell(0, 0), 5)

    # --- SPEC-CTRL-3: Immediate Conflict Policy ---

    def test_controller_rejects_conflicting_placement(self) -> None:
        ctrl = SudokuController(self.board)
        _, obs = ctrl.step_raw('{"tool": "place", "row": 7, "col": 8, "digit": 2}')
        self.assertFalse(obs.accepted)
        self.assertIsNotNone(obs.rejection_reason)
        self.assertEqual(self.board.get_cell(7, 8), 0)

    # --- SPEC-CTRL-4: Empty Undo Protection ---

    def test_controller_rejects_undo_when_history_empty(self) -> None:
        ctrl = SudokuController(self.board)
        _, obs = ctrl.step_raw('{"tool": "undo", "steps": 1}')
        self.assertFalse(obs.accepted)
        self.assertIn("undo", (obs.rejection_reason or "").lower())

    # --- SPEC-CTRL-5 & SPEC-CTRL-6: Repeated Rejection & Consecutive Rejection Limit ---

    def test_repeated_rejected_request_and_consecutive_limit(self) -> None:
        ctrl = SudokuController(
            self.board,
            config=ControllerConfig(
                max_steps=20,
                reject_repeated_failures=True,
                max_consecutive_rejections=3,
            ),
        )
        bad_req = '{"tool": "place", "row": 7, "col": 8, "digit": 2}'
        _, obs1 = ctrl.step_raw(bad_req)
        self.assertFalse(obs1.accepted)
        self.assertFalse(obs1.terminal)

        _, obs2 = ctrl.step_raw(bad_req)
        self.assertFalse(obs2.accepted)
        self.assertIn("repeated", (obs2.rejection_reason or "").lower())
        self.assertFalse(obs2.terminal)

        _, obs3 = ctrl.step_raw('{"tool": "undo"}')
        self.assertFalse(obs3.accepted)
        self.assertTrue(obs3.terminal)
        self.assertIn("rejection", (obs3.terminal_reason or "").lower())

    # --- KB & Submit Controller Mediation ---

    def test_controller_kb_and_submit_validation(self) -> None:
        ctrl = SudokuController(self.board)

        # Submitting an incomplete board is rejected
        _, sub_early = ctrl.step_raw('{"tool": "submit"}')
        self.assertFalse(sub_early.accepted)
        self.assertFalse(sub_early.terminal)

        # Add KB rule & fact, query entailment
        _, kb1 = ctrl.step_raw(
            '{"tool": "kb_add", "kind": "rule", "formula": "c_r7_c8_d5 | c_r7_c8_d2"}'
        )
        self.assertTrue(kb1.accepted)
        self.assertTrue(kb1.kb_consistent)

        _, kb2 = ctrl.step_raw('{"tool": "kb_add", "kind": "fact", "formula": "~c_r7_c8_d2"}')
        self.assertTrue(kb2.accepted)

        _, kbq = ctrl.step_raw('{"tool": "kb_query", "formula": "c_r7_c8_d5"}')
        self.assertTrue(kbq.accepted)
        self.assertEqual(kbq.query_status, "entailed")

        # Complete all 3 empty cells and submit
        ctrl.step_raw('{"tool": "place", "row": 7, "col": 8, "digit": 5}')
        ctrl.step_raw('{"tool": "place", "row": 8, "col": 6, "digit": 1}')
        ctrl.step_raw('{"tool": "place", "row": 8, "col": 7, "digit": 7}')

        _, sub_ok = ctrl.step_raw('{"tool": "submit"}')
        self.assertTrue(sub_ok.accepted)
        self.assertTrue(sub_ok.terminal)
        self.assertTrue(ctrl.submission_accepted)


if __name__ == "__main__":
    unittest.main()

