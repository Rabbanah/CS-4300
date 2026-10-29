"""Sudoku puzzle scenarios with unique solutions for testing and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from sudoku import SudokuBoard


@dataclass(frozen=True)
class SudokuScenario:
    name: str
    difficulty: str
    description: str
    puzzle: str
    solution: str

    def make_board(self) -> SudokuBoard:
        return SudokuBoard.from_string(self.puzzle)


SCENARIOS: dict[str, SudokuScenario] = {
    "near-complete": SudokuScenario(
        name="near-complete",
        difficulty="trivial",
        description="9x9 puzzle with 3 empty cells solvable by direct deduction.",
        puzzle=(
            "534678912"
            "672195348"
            "198342567"
            "859761423"
            "426853791"
            "713924856"
            "961537284"
            "287419630"
            "345286009"
        ),
        solution=(
            "534678912"
            "672195348"
            "198342567"
            "859761423"
            "426853791"
            "713924856"
            "961537284"
            "287419635"
            "345286179"
        ),
    ),
    "easy-1": SudokuScenario(
        name="easy-1",
        difficulty="easy",
        description="Classic 9x9 puzzle with 12 empty cells solvable via naked singles and KB queries.",
        puzzle=(
            "530678912"
            "672195348"
            "198342567"
            "859061423"
            "426803791"
            "713920856"
            "960537284"
            "287419035"
            "345280170"
        ),
        solution=(
            "534678912"
            "672195348"
            "198342567"
            "859761423"
            "426853791"
            "713924856"
            "961537284"
            "287419635"
            "345286179"
        ),
    ),
    "branching-1": SudokuScenario(
        name="branching-1",
        difficulty="medium",
        description="9x9 puzzle with multi-candidate branching that triggers hypothesis testing and undo recovery.",
        puzzle=(
            "534008912"
            "002195308"
            "198302500"
            "859001023"
            "020853091"
            "013920850"
            "901530280"
            "280019035"
            "305280109"
        ),
        solution=(
            "534678912"
            "672195348"
            "198342567"
            "859761423"
            "426853791"
            "713924856"
            "961537284"
            "287419635"
            "345286179"
        ),
    ),
}
