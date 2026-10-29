"""Unit tests for `sudoku.py` derived from SPECIFICATIONS.md (SPEC-SUDOKU-1..5, SPEC-UNDO-1..2)."""

from __future__ import annotations

import unittest

from scenarios import SCENARIOS
from sudoku import CellPos, PlacementRecord, SudokuBoard, box_index


NEAR_COMPLETE_PUZZLE = SCENARIOS["near-complete"].puzzle
SOLVED_PUZZLE = SCENARIOS["near-complete"].solution


class TestSudokuBoardSpec(unittest.TestCase):
    """Black-box and supplementary white-box tests for SudokuBoard."""

    # --- SPEC-SUDOKU-1: Grid Coordinate System and Representation ---

    def test_box_index_mapping(self) -> None:
        self.assertEqual(box_index(0, 0), 0)
        self.assertEqual(box_index(2, 2), 0)
        self.assertEqual(box_index(0, 3), 1)
        self.assertEqual(box_index(1, 8), 2)
        self.assertEqual(box_index(3, 0), 3)
        self.assertEqual(box_index(4, 4), 4)
        self.assertEqual(box_index(8, 8), 8)
        with self.assertRaises(ValueError):
            box_index(-1, 0)
        with self.assertRaises(ValueError):
            box_index(0, 9)

    def test_from_string_and_grid_dimensions(self) -> None:
        board = SudokuBoard.from_string(NEAR_COMPLETE_PUZZLE)
        grid = board.to_grid()
        self.assertEqual(len(grid), 9)
        self.assertTrue(all(len(row) == 9 for row in grid))
        self.assertEqual(board.get_cell(0, 0), 5)
        self.assertEqual(board.get_cell(7, 8), 0)
        self.assertEqual(board.get_cell(8, 6), 0)
        self.assertEqual(board.get_cell(8, 7), 0)

    def test_from_string_accepts_dots_and_whitespace(self) -> None:
        dotted = NEAR_COMPLETE_PUZZLE.replace("0", ".")
        spaced = "\n".join(dotted[i : i + 9] for i in range(0, 81, 9))
        board = SudokuBoard.from_string(spaced)
        self.assertEqual(board.get_cell(7, 8), 0)
        self.assertEqual(board.get_cell(0, 0), 5)

    def test_invalid_initialization_rejected(self) -> None:
        with self.assertRaises(ValueError):
            SudokuBoard.from_string("123")
        with self.assertRaises(ValueError):
            SudokuBoard([[0] * 8 for _ in range(9)])
        bad_grid = [[0] * 9 for _ in range(9)]
        bad_grid[0][0] = 10
        with self.assertRaises(ValueError):
            SudokuBoard(bad_grid)

        bool_grid = [[0] * 9 for _ in range(9)]
        bool_grid[0][0] = True  # type: ignore[assignment]
        with self.assertRaises(TypeError):
            SudokuBoard(bool_grid)

        # Conflicting initial given clues in row 0
        conflict_grid = [[0] * 9 for _ in range(9)]
        conflict_grid[0][0] = 4
        conflict_grid[0][5] = 4
        with self.assertRaises(ValueError):
            SudokuBoard(conflict_grid)

    # --- SPEC-SUDOKU-2: Immutable Given Clues vs. Mutable Cells ---

    def test_given_clues_are_immutable(self) -> None:
        board = SudokuBoard.from_string(NEAR_COMPLETE_PUZZLE)
        self.assertTrue(board.is_given(0, 0))
        self.assertFalse(board.is_given(7, 8))
        with self.assertRaises(ValueError):
            board.place(0, 0, 9)
        self.assertEqual(board.get_cell(0, 0), 5)
        self.assertTrue(board.givens_preserved())

    # --- SPEC-SUDOKU-3: Inspection Operations ---

    def test_inspect_cell_row_col_box_and_candidates(self) -> None:
        board = SudokuBoard.from_string(NEAR_COMPLETE_PUZZLE)
        self.assertEqual(board.get_row(7), (2, 8, 7, 4, 1, 9, 6, 3, 0))
        self.assertEqual(board.get_col(8), (2, 8, 7, 3, 1, 6, 4, 0, 9))
        self.assertEqual(board.get_box(8), (2, 8, 4, 6, 3, 0, 0, 0, 9))
        self.assertEqual(board.candidates(7, 8), (5,))
        self.assertEqual(board.candidates(8, 6), (1,))
        self.assertEqual(board.candidates(8, 7), (7,))

        with self.assertRaises(ValueError):
            board.get_row(9)
        with self.assertRaises(ValueError):
            board.get_col(-1)
        with self.assertRaises(ValueError):
            board.get_box(9)

    # --- SPEC-SUDOKU-4: Placement and Conflict Detection ---

    def test_place_and_conflict_detection(self) -> None:
        board = SudokuBoard.from_string(NEAR_COMPLETE_PUZZLE)
        conflict, reason = board.would_conflict(7, 8, 2)
        self.assertTrue(conflict)
        self.assertIsNotNone(reason)

        ok, reason_ok = board.would_conflict(7, 8, 5)
        self.assertFalse(ok)
        self.assertIsNone(reason_ok)

        with self.assertRaises(ValueError):
            board.place(7, 8, 2, allow_conflict=False)
        self.assertEqual(board.get_cell(7, 8), 0)

        rec = board.place(7, 8, 5)
        self.assertEqual(rec, PlacementRecord(row=7, col=8, previous_digit=0, new_digit=5))
        self.assertEqual(board.get_cell(7, 8), 5)

        # Redundant placement of same digit rejected
        with self.assertRaises(ValueError):
            board.place(7, 8, 5)

        # Bool digit or coordinate rejected with TypeError
        with self.assertRaises(TypeError):
            board.place(8, 6, True)  # type: ignore[arg-type]

    # --- SPEC-SUDOKU-5: Completion and Solution Validation ---

    def test_completion_and_solution_validation(self) -> None:
        board = SudokuBoard.from_string(NEAR_COMPLETE_PUZZLE)
        self.assertFalse(board.is_complete())
        self.assertFalse(board.is_solved())

        board.place(7, 8, 5)
        board.place(8, 6, 1)
        board.place(8, 7, 7)

        self.assertTrue(board.is_complete())
        self.assertEqual(board.find_conflicts(), ())
        self.assertTrue(board.givens_preserved())
        self.assertTrue(board.is_solved())

    def test_complete_board_with_conflict_is_not_solved(self) -> None:
        board = SudokuBoard.from_string(NEAR_COMPLETE_PUZZLE)
        board.place(7, 8, 5)
        board.place(8, 6, 7, allow_conflict=True)
        board.place(8, 7, 1, allow_conflict=True)
        self.assertTrue(board.is_complete())
        self.assertGreater(len(board.find_conflicts()), 0)
        self.assertFalse(board.is_solved())

    # --- SPEC-UNDO-1 & SPEC-UNDO-2: LIFO Undo Stack ---

    def test_lifo_undo_single_and_multi_step(self) -> None:
        board = SudokuBoard.from_string(NEAR_COMPLETE_PUZZLE)
        with self.assertRaises(ValueError):
            board.undo(1)

        board.place(7, 8, 5)
        board.place(8, 6, 1)
        board.place(8, 6, 7, allow_conflict=True)
        self.assertEqual(len(board.history), 3)

        undone_one = board.undo(1)
        self.assertEqual(
            undone_one,
            (PlacementRecord(row=8, col=6, previous_digit=1, new_digit=7),),
        )
        self.assertEqual(board.get_cell(8, 6), 1)

        undone_two = board.undo(2)
        self.assertEqual(len(undone_two), 2)
        self.assertEqual(board.get_cell(8, 6), 0)
        self.assertEqual(board.get_cell(7, 8), 0)
        self.assertEqual(len(board.history), 0)

        with self.assertRaises(ValueError):
            board.undo(1)


if __name__ == "__main__":
    unittest.main()

