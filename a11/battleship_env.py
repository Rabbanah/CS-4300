"""Battleship environment, configuration validation, and explicit terminal statuses.

Implements the deterministic, partially observable Battleship simulator specified in
U0-HW-10. The environment owns ground-truth ship placements, enforces coordinate bounds
and duplicate-probe rules, tracks shot budgets, and emits structured ShotResult percepts.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum

Coord = tuple[int, int]

STANDARD_FLEET: dict[str, int] = {
    "Carrier": 5,
    "Battleship": 4,
    "Cruiser": 3,
    "Submarine": 3,
    "Destroyer": 2,
}

POC_FLEET: dict[str, int] = {
    "Cruiser": 3,
    "Destroyer": 2,
}


class EpisodeStatus(str, Enum):
    """Explicit lifecycle and terminal statuses for a Battleship episode."""

    RUNNING = "RUNNING"
    VICTORY = "VICTORY"
    TIMEOUT = "TIMEOUT"
    BELIEF_COLLAPSE = "BELIEF_COLLAPSE"
    INVALID_CONFIG = "INVALID_CONFIG"
    INVALID_ACTION = "INVALID_ACTION"


class InvalidConfigError(ValueError):
    """Raised when board dimensions, fleet manifest, or budget are unsupported."""


class InvalidActionError(ValueError):
    """Raised when an agent attempts an out-of-bounds or duplicate probe."""


@dataclass(frozen=True)
class ShotResult:
    """Structured sensor percept returned by BattleshipEnv.probe()."""

    coord: Coord
    hit: bool
    sunk_ship: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "coord": [self.coord[0], self.coord[1]],
            "hit": self.hit,
            "sunk_ship": self.sunk_ship,
        }


@dataclass(frozen=True)
class EnvConfig:
    """Configuration for a Battleship episode."""

    rows: int = 10
    cols: int = 10
    fleet_spec: dict[str, int] = field(default_factory=lambda: dict(STANDARD_FLEET))
    shot_budget: int = 75
    seed: int = 4300
    fixed_ships: dict[str, list[Coord]] | None = None


def validate_config(config: EnvConfig) -> None:
    """Validate grid dimensions, fleet manifest, shot budget, and optional fixed layout.

    Raises InvalidConfigError with an actionable message for any unsupported input.
    """
    if not isinstance(config.rows, int) or not isinstance(config.cols, int):
        raise InvalidConfigError("Grid dimensions rows and cols must be integers.")
    if config.rows <= 0 or config.cols <= 0:
        raise InvalidConfigError(
            f"Grid dimensions must be positive; got {config.rows}x{config.cols}."
        )
    if config.rows > 20 or config.cols > 20:
        raise InvalidConfigError(
            f"Grid dimensions {config.rows}x{config.cols} exceed supported maximum (20x20)."
        )
    if not config.fleet_spec:
        raise InvalidConfigError("Fleet manifest must contain at least one ship.")
    if not isinstance(config.shot_budget, int) or config.shot_budget <= 0:
        raise InvalidConfigError(
            f"Shot budget must be a positive integer; got {config.shot_budget}."
        )

    total_ship_cells = 0
    max_dim = max(config.rows, config.cols)
    for name, length in config.fleet_spec.items():
        if not name or not isinstance(name, str):
            raise InvalidConfigError("Each ship must have a non-empty string name.")
        if not isinstance(length, int) or length <= 0:
            raise InvalidConfigError(
                f"Ship '{name}' must have a positive integer length; got {length}."
            )
        if length > max_dim:
            raise InvalidConfigError(
                f"Ship '{name}' (length {length}) cannot fit on a {config.rows}x{config.cols} grid."
            )
        total_ship_cells += length

    board_cells = config.rows * config.cols
    if total_ship_cells > board_cells:
        raise InvalidConfigError(
            f"Total fleet cells ({total_ship_cells}) exceed board capacity ({board_cells} cells on {config.rows}x{config.cols})."
        )

    if config.fixed_ships is not None:
        if set(config.fixed_ships.keys()) != set(config.fleet_spec.keys()):
            raise InvalidConfigError(
                "fixed_ships keys must match fleet_spec keys exactly."
            )
        occupied: set[Coord] = set()
        for name, coords in config.fixed_ships.items():
            expected_len = config.fleet_spec[name]
            if len(coords) != expected_len:
                raise InvalidConfigError(
                    f"Fixed ship '{name}' has {len(coords)} cells, expected {expected_len}."
                )
            coord_set = set(coords)
            if len(coord_set) != expected_len:
                raise InvalidConfigError(
                    f"Fixed ship '{name}' contains duplicate coordinates."
                )
            for r, c in coords:
                if not (0 <= r < config.rows and 0 <= c < config.cols):
                    raise InvalidConfigError(
                        f"Fixed ship '{name}' has out-of-bounds coordinate {(r, c)}."
                    )
            rows_used = sorted({r for r, _ in coords})
            cols_used = sorted({c for _, c in coords})
            horizontal = len(rows_used) == 1 and cols_used == list(
                range(cols_used[0], cols_used[0] + expected_len)
            )
            vertical = len(cols_used) == 1 and rows_used == list(
                range(rows_used[0], rows_used[0] + expected_len)
            )
            if not (horizontal or vertical):
                raise InvalidConfigError(
                    f"Fixed ship '{name}' must occupy contiguous horizontal or vertical cells."
                )
            if not occupied.isdisjoint(coord_set):
                raise InvalidConfigError(
                    f"Fixed ship '{name}' overlaps another ship."
                )
            occupied |= coord_set


class BattleshipEnv:
    """Deterministic Battleship environment owning hidden ship coordinates and rules."""

    def __init__(self, config: EnvConfig | None = None) -> None:
        self.config = config if config is not None else EnvConfig()
        validate_config(self.config)

        self.rows = self.config.rows
        self.cols = self.config.cols
        self.fleet_spec = dict(self.config.fleet_spec)
        self.shot_budget = self.config.shot_budget
        self.seed = self.config.seed
        self._rng = random.Random(self.seed)

        if self.config.fixed_ships is not None:
            self._ships: dict[str, set[Coord]] = {
                name: set(coords) for name, coords in self.config.fixed_ships.items()
            }
        else:
            self._ships = self._generate_fleet()

        self._all_ship_cells: set[Coord] = set().union(*self._ships.values())
        self.probed: set[Coord] = set()
        self.hits: set[Coord] = set()
        self.misses: set[Coord] = set()
        self.sunk_ships: list[str] = []
        self.shots_taken: list[ShotResult] = []

    def _all_placements(self, length: int) -> list[set[Coord]]:
        placements: list[set[Coord]] = []
        if length == 1:
            for r in range(self.rows):
                for c in range(self.cols):
                    placements.append({(r, c)})
            return placements
        for r in range(self.rows):
            for c in range(self.cols - length + 1):
                placements.append({(r, c + i) for i in range(length)})
        for r in range(self.rows - length + 1):
            for c in range(self.cols):
                placements.append({(r + i, c) for i in range(length)})
        return placements

    def _generate_fleet(self, max_attempts: int = 500) -> dict[str, set[Coord]]:
        # Place longer ships first to reduce packing rejections on tight grids.
        ordered = sorted(self.fleet_spec.items(), key=lambda kv: (-kv[1], kv[0]))
        for _ in range(max_attempts):
            occupied: set[Coord] = set()
            placed: dict[str, set[Coord]] = {}
            ok = True
            for name, length in ordered:
                candidates = self._all_placements(length)
                self._rng.shuffle(candidates)
                chosen = next((c for c in candidates if c.isdisjoint(occupied)), None)
                if chosen is None:
                    ok = False
                    break
                placed[name] = chosen
                occupied |= chosen
            if ok:
                return {name: placed[name] for name in self.fleet_spec}
        raise InvalidConfigError(
            f"Unable to pack fleet {self.fleet_spec} onto {self.rows}x{self.cols} grid "
            f"within {max_attempts} placement attempts."
        )

    @property
    def total_ship_cells(self) -> int:
        return len(self._all_ship_cells)

    @property
    def remaining_ship_cells(self) -> int:
        return len(self._all_ship_cells - self.hits)

    @property
    def shots_count(self) -> int:
        return len(self.shots_taken)

    @property
    def budget_remaining(self) -> int:
        return max(0, self.shot_budget - self.shots_count)

    @property
    def is_won(self) -> bool:
        return self._all_ship_cells.issubset(self.probed)

    @property
    def is_timeout(self) -> bool:
        return (not self.is_won) and (self.shots_count >= self.shot_budget)

    @property
    def status(self) -> EpisodeStatus:
        if self.is_won:
            return EpisodeStatus.VICTORY
        if self.is_timeout:
            return EpisodeStatus.TIMEOUT
        return EpisodeStatus.RUNNING

    def available_actions(self) -> list[Coord]:
        """Return all valid unprobed coordinates in deterministic row-major order."""
        return [
            (r, c)
            for r in range(self.rows)
            for c in range(self.cols)
            if (r, c) not in self.probed
        ]

    def probe(self, coord: Coord) -> ShotResult:
        """Apply a probe action, enforcing bounds, uniqueness, and terminal state."""
        if self.status != EpisodeStatus.RUNNING:
            raise InvalidActionError(
                f"Cannot probe {coord}: episode already in terminal status {self.status.value}."
            )
        if not (isinstance(coord, tuple) and len(coord) == 2):
            raise InvalidActionError(f"Invalid coordinate format: {coord!r}.")
        r, c = coord
        if not (isinstance(r, int) and isinstance(c, int)):
            raise InvalidActionError(f"Coordinate indices must be integers: {coord!r}.")
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            raise InvalidActionError(
                f"Coordinate {coord} is out of bounds for {self.rows}x{self.cols} grid."
            )
        if coord in self.probed:
            raise InvalidActionError(f"Coordinate {coord} has already been probed.")

        self.probed.add(coord)
        for name, cells in self._ships.items():
            if coord in cells:
                self.hits.add(coord)
                sunk = name if cells.issubset(self.probed) else None
                if sunk is not None:
                    self.sunk_ships.append(sunk)
                res = ShotResult(coord=coord, hit=True, sunk_ship=sunk)
                self.shots_taken.append(res)
                return res

        self.misses.add(coord)
        res = ShotResult(coord=coord, hit=False, sunk_ship=None)
        self.shots_taken.append(res)
        return res

    def inspect_state(self, reveal_ships: bool = False) -> dict[str, object]:
        """Return an inspectable snapshot of environment effects and status."""
        snapshot: dict[str, object] = {
            "rows": self.rows,
            "cols": self.cols,
            "shots_taken": self.shots_count,
            "shot_budget": self.shot_budget,
            "budget_remaining": self.budget_remaining,
            "total_hits": len(self.hits),
            "total_misses": len(self.misses),
            "sunk_ships": list(self.sunk_ships),
            "remaining_ship_cells": self.remaining_ship_cells,
            "status": self.status.value,
        }
        if reveal_ships:
            snapshot["ground_truth_ships"] = {
                name: [list(coord) for coord in sorted(cells)]
                for name, cells in self._ships.items()
            }
        return snapshot

    def render_board(self, reveal_ships: bool = False) -> str:
        """Render an ASCII grid showing hits (X), misses (o), and optionally hidden ships (S)."""
        header = "   " + " ".join(f"{c}" for c in range(self.cols))
        lines = [header]
        for r in range(self.rows):
            row_chars: list[str] = []
            for c in range(self.cols):
                coord = (r, c)
                if coord in self.hits:
                    row_chars.append("X")
                elif coord in self.misses:
                    row_chars.append("o")
                elif reveal_ships and coord in self._all_ship_cells:
                    row_chars.append("S")
                else:
                    row_chars.append(".")
            lines.append(f"{r:2d} " + " ".join(row_chars))
        return "\n".join(lines)
