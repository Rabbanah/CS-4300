"""9-by-9 Sudoku puzzle representation, inspection, placement, and undo operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True, order=True)
class CellPos:
    row: int
    col: int


@dataclass(frozen=True)
class PlacementRecord:
    row: int
    col: int
    previous_digit: int
    new_digit: int


def _validate_coord(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
    if value < 0 or value > 8:
        raise ValueError(f"{name} must be in 0..8, got {value}")


def _validate_digit(digit: int, allow_zero: bool = False) -> None:
    if isinstance(digit, bool) or not isinstance(digit, int):
        raise TypeError(f"digit must be an integer, got {type(digit).__name__}")
    low = 0 if allow_zero else 1
    if digit < low or digit > 9:
        raise ValueError(f"digit must be in {low}..9, got {digit}")


def box_index(row: int, col: int) -> int:
    """Return the 0-based 3x3 box index (0..8) for (row, col)."""
    _validate_coord(row, "row")
    _validate_coord(col, "col")
    return 3 * (row // 3) + (col // 3)


class SudokuBoard:
    """9-by-9 Sudoku puzzle with immutable given clues and LIFO mutable undo stack."""

    def __init__(self, grid: Sequence[Sequence[int]]) -> None:
        if not isinstance(grid, Sequence) or len(grid) != 9:
            raise ValueError("SudokuBoard requires a 9x9 sequence of integers")

        parsed_rows: list[list[int]] = []
        givens: set[CellPos] = set()
        for r, row in enumerate(grid):
            if not isinstance(row, Sequence) or isinstance(row, (str, bytes)) or len(row) != 9:
                raise ValueError(f"Row {r} must contain exactly 9 integers")
            parsed_row: list[int] = []
            for c, val in enumerate(row):
                _validate_digit(val, allow_zero=True)
                parsed_row.append(val)
                if val != 0:
                    givens.add(CellPos(r, c))
            parsed_rows.append(parsed_row)

        self._initial_grid: tuple[tuple[int, ...], ...] = tuple(
            tuple(row) for row in parsed_rows
        )
        self._grid: list[list[int]] = [list(row) for row in parsed_rows]
        self._givens: frozenset[CellPos] = frozenset(givens)
        self._history: list[PlacementRecord] = []

        conflicts = self.find_conflicts()
        if conflicts:
            raise ValueError(
                f"Initial Sudoku board contains conflicting given clues: {'; '.join(conflicts)}"
            )

    @classmethod
    def from_string(cls, puzzle_str: str) -> SudokuBoard:
        """Create a SudokuBoard from an 81-character puzzle string (0 or . for empty)."""
        if not isinstance(puzzle_str, str):
            raise TypeError("puzzle_str must be a string")
        cells: list[int] = []
        for ch in puzzle_str:
            if ch in "123456789":
                cells.append(int(ch))
            elif ch in ("0", "."):
                cells.append(0)
            elif ch.isspace() or ch in ("|", "-", "+"):
                continue
            else:
                raise ValueError(f"Invalid character {ch!r} in puzzle string")
        if len(cells) != 81:
            raise ValueError(f"Expected 81 cells in puzzle string, found {len(cells)}")
        grid = [cells[i : i + 9] for i in range(0, 81, 9)]
        return cls(grid)

    @property
    def history(self) -> tuple[PlacementRecord, ...]:
        """Return the current LIFO stack of accepted mutable placements."""
        return tuple(self._history)

    def is_given(self, row: int, col: int) -> bool:
        """Return whether (row, col) is an immutable initial clue."""
        _validate_coord(row, "row")
        _validate_coord(col, "col")
        return CellPos(row, col) in self._givens

    def get_cell(self, row: int, col: int) -> int:
        """Return the digit (0..9) at (row, col)."""
        _validate_coord(row, "row")
        _validate_coord(col, "col")
        return self._grid[row][col]

    def get_row(self, row: int) -> tuple[int, ...]:
        """Return the 9 digits in row (0..8)."""
        _validate_coord(row, "row")
        return tuple(self._grid[row])

    def get_col(self, col: int) -> tuple[int, ...]:
        """Return the 9 digits in col (0..8)."""
        _validate_coord(col, "col")
        return tuple(self._grid[r][col] for r in range(9))

    def get_box(self, box: int) -> tuple[int, ...]:
        """Return the 9 digits in 3x3 box (0..8) in row-major order."""
        _validate_coord(box, "box")
        r0 = 3 * (box // 3)
        c0 = 3 * (box % 3)
        return tuple(
            self._grid[r][c]
            for r in range(r0, r0 + 3)
            for c in range(c0, c0 + 3)
        )

    def candidates(self, row: int, col: int) -> tuple[int, ...]:
        """Return sorted legal digits (1..9) for (row, col) given current peers."""
        _validate_coord(row, "row")
        _validate_coord(col, "col")
        if self.is_given(row, col):
            return (self._grid[row][col],)
        return tuple(
            d for d in range(1, 10) if not self.would_conflict(row, col, d)[0]
        )

    def would_conflict(self, row: int, col: int, digit: int) -> tuple[bool, str | None]:
        """Return (True, reason) if placing digit at (row, col) conflicts with any peer."""
        _validate_coord(row, "row")
        _validate_coord(col, "col")
        _validate_digit(digit, allow_zero=False)

        for c in range(9):
            if c != col and self._grid[row][c] == digit:
                return True, f"digit {digit} already in row {row} at ({row}, {c})"
        for r in range(9):
            if r != row and self._grid[r][col] == digit:
                return True, f"digit {digit} already in col {col} at ({r}, {col})"
        b = box_index(row, col)
        r0 = 3 * (b // 3)
        c0 = 3 * (b % 3)
        for r in range(r0, r0 + 3):
            for c in range(c0, c0 + 3):
                if (r != row or c != col) and self._grid[r][c] == digit:
                    return True, f"digit {digit} already in box {b} at ({r}, {c})"
        return False, None

    def place(
        self, row: int, col: int, digit: int, allow_conflict: bool = False
    ) -> PlacementRecord:
        """Place digit (1..9) into mutable cell (row, col) and record undo history."""
        _validate_coord(row, "row")
        _validate_coord(col, "col")
        _validate_digit(digit, allow_zero=False)

        if self.is_given(row, col):
            raise ValueError(f"cannot modify given clue at ({row}, {col})")

        previous = self._grid[row][col]
        if previous == digit:
            raise ValueError(f"cell ({row}, {col}) already holds digit {digit}")

        if not allow_conflict:
            conflict, reason = self.would_conflict(row, col, digit)
            if conflict:
                raise ValueError(f"conflict placing {digit} at ({row}, {col}): {reason}")

        self._grid[row][col] = digit
        record = PlacementRecord(
            row=row, col=col, previous_digit=previous, new_digit=digit
        )
        self._history.append(record)
        return record

    def undo(self, steps: int = 1) -> tuple[PlacementRecord, ...]:
        """Undo the most recent `steps` mutable placements in LIFO order."""
        if isinstance(steps, bool) or not isinstance(steps, int):
            raise TypeError("steps must be an integer")
        if steps < 1:
            raise ValueError("steps must be >= 1")
        if len(self._history) < steps:
            raise ValueError(
                f"cannot undo {steps} step(s): placement history has {len(self._history)} entry/entries"
            )

        undone: list[PlacementRecord] = []
        for _ in range(steps):
            rec = self._history.pop()
            self._grid[rec.row][rec.col] = rec.previous_digit
            undone.append(rec)
        return tuple(undone)

    def find_conflicts(self) -> tuple[str, ...]:
        """Return conflict descriptions for any row, column, or box duplicates."""
        issues: list[str] = []
        for r in range(9):
            seen: dict[int, int] = {}
            for c in range(9):
                val = self._grid[r][c]
                if val != 0:
                    if val in seen:
                        issues.append(
                            f"row {r} duplicate digit {val} at cols {seen[val]} and {c}"
                        )
                    else:
                        seen[val] = c
        for c in range(9):
            seen = {}
            for r in range(9):
                val = self._grid[r][c]
                if val != 0:
                    if val in seen:
                        issues.append(
                            f"col {c} duplicate digit {val} at rows {seen[val]} and {r}"
                        )
                    else:
                        seen[val] = r
        for b in range(9):
            r0 = 3 * (b // 3)
            c0 = 3 * (b % 3)
            seen_pos: dict[int, tuple[int, int]] = {}
            for r in range(r0, r0 + 3):
                for c in range(c0, c0 + 3):
                    val = self._grid[r][c]
                    if val != 0:
                        if val in seen_pos:
                            pr, pc = seen_pos[val]
                            issues.append(
                                f"box {b} duplicate digit {val} at ({pr}, {pc}) and ({r}, {c})"
                            )
                        else:
                            seen_pos[val] = (r, c)
        return tuple(issues)

    def is_complete(self) -> bool:
        """Return True iff every cell holds a non-zero digit (1..9)."""
        return all(self._grid[r][c] != 0 for r in range(9) for c in range(9))

    def givens_preserved(self) -> bool:
        """Return True iff all initial given clues remain unchanged."""
        for pos in self._givens:
            if self._grid[pos.row][pos.col] != self._initial_grid[pos.row][pos.col]:
                return False
        return True

    def is_solved(self) -> bool:
        """Return True iff givens are preserved, board is complete, and no conflicts exist."""
        if not self.givens_preserved() or not self.is_complete():
            return False
        target = set(range(1, 10))
        for i in range(9):
            if set(self.get_row(i)) != target:
                return False
            if set(self.get_col(i)) != target:
                return False
            if set(self.get_box(i)) != target:
                return False
        return len(self.find_conflicts()) == 0

    def to_grid(self) -> tuple[tuple[int, ...], ...]:
        """Return an immutable 9x9 copy of the current grid."""
        return tuple(tuple(row) for row in self._grid)

    def format_board(self) -> str:
        """Return a human- and LLM-readable ASCII representation of the board."""
        lines = ["    c0 c1 c2   c3 c4 c5   c6 c7 c8"]
        for r in range(9):
            if r in (3, 6):
                lines.append("   ----------+----------+----------")
            row_cells: list[str] = []
            for c in range(9):
                if c in (3, 6):
                    row_cells.append("|")
                val = self._grid[r][c]
                ch = "." if val == 0 else str(val)
                row_cells.append(f" {ch}")
            lines.append(f"r{r} " + " ".join(row_cells))
        return "\n".join(lines)
