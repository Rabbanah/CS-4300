"""Minimum Proof of Concept: Bayesian Belief-State Battleship Agent (6x6 Grid)."""

from __future__ import annotations

import itertools
import math
import random
from dataclasses import dataclass

Coord = tuple[int, int]


@dataclass(frozen=True)
class ShotResult:
    coord: Coord
    hit: bool
    sunk_ship: str | None = None


class BattleshipEnv:
    """Deterministic 6x6 Battleship environment with hidden fleet."""

    def __init__(self, rows: int = 6, cols: int = 6, seed: int = 4300) -> None:
        self.rows = rows
        self.cols = cols
        self.fleet_spec: dict[str, int] = {"Cruiser": 3, "Destroyer": 2}
        self._rng = random.Random(seed)
        self._ships: dict[str, set[Coord]] = self._generate_fleet()
        self.shots_taken: list[ShotResult] = []
        self.probed: set[Coord] = set()

    def _generate_fleet(self) -> dict[str, set[Coord]]:
        while True:
            occupied: set[Coord] = set()
            placed: dict[str, set[Coord]] = {}
            ok = True
            for name, length in self.fleet_spec.items():
                candidates = self._all_placements(length)
                self._rng.shuffle(candidates)
                chosen = next((c for c in candidates if c.isdisjoint(occupied)), None)
                if chosen is None:
                    ok = False
                    break
                placed[name] = chosen
                occupied |= chosen
            if ok:
                return placed

    def _all_placements(self, length: int) -> list[set[Coord]]:
        placements: list[set[Coord]] = []
        for r in range(self.rows):
            for c in range(self.cols - length + 1):
                placements.append({(r, c + i) for i in range(length)})
        for r in range(self.rows - length + 1):
            for c in range(self.cols):
                placements.append({(r + i, c) for i in range(length)})
        return placements

    @property
    def is_won(self) -> bool:
        all_ship_cells = set().union(*self._ships.values())
        return all_ship_cells.issubset(self.probed)

    def probe(self, coord: Coord) -> ShotResult:
        if coord in self.probed:
            raise ValueError(f"Coordinate {coord} already probed")
        self.probed.add(coord)
        for name, cells in self._ships.items():
            if coord in cells:
                sunk = name if cells.issubset(self.probed) else None
                res = ShotResult(coord=coord, hit=True, sunk_ship=sunk)
                self.shots_taken.append(res)
                return res
        res = ShotResult(coord=coord, hit=False, sunk_ship=None)
        self.shots_taken.append(res)
        return res


class BayesianBeliefAgent:
    """Computes exact posterior cell occupancy probabilities over valid fleet states."""

    def __init__(self, rows: int, cols: int, fleet_spec: dict[str, int]) -> None:
        self.rows = rows
        self.cols = cols
        self.fleet_spec = fleet_spec
        self.hits: set[Coord] = set()
        self.misses: set[Coord] = set()
        self.sunk_ships: set[str] = set()

        # Precompute all valid non-overlapping joint fleet configurations
        ship_names = list(fleet_spec.keys())
        per_ship = [self._valid_single_placements(fleet_spec[name]) for name in ship_names]
        self.valid_fleets: list[dict[str, frozenset[Coord]]] = []
        for combo in itertools.product(*per_ship):
            union_cells = set().union(*combo)
            if len(union_cells) == sum(fleet_spec.values()):
                self.valid_fleets.append(
                    {name: frozenset(cells) for name, cells in zip(ship_names, combo)}
                )

    def _valid_single_placements(self, length: int) -> list[set[Coord]]:
        out: list[set[Coord]] = []
        for r in range(self.rows):
            for c in range(self.cols - length + 1):
                out.append({(r, c + i) for i in range(length)})
        for r in range(self.rows - length + 1):
            for c in range(self.cols):
                out.append({(r + i, c) for i in range(length)})
        return out

    def observe(self, result: ShotResult) -> None:
        if result.hit:
            self.hits.add(result.coord)
            if result.sunk_ship:
                self.sunk_ships.add(result.sunk_ship)
        else:
            self.misses.add(result.coord)

        # Filter hypothesis space to fleets consistent with all observations
        self.valid_fleets = [
            fleet
            for fleet in self.valid_fleets
            if self._is_consistent(fleet)
        ]

    def _is_consistent(self, fleet: dict[str, frozenset[Coord]]) -> bool:
        occupied = set().union(*fleet.values())
        if not self.hits.issubset(occupied):
            return False
        if not self.misses.isdisjoint(occupied):
            return False
        for name, cells in fleet.items():
            is_sunk_in_obs = name in self.sunk_ships
            all_cells_hit = cells.issubset(self.hits)
            if is_sunk_in_obs != all_cells_hit:
                return False
        return True

    def posterior_probabilities(self) -> dict[Coord, float]:
        total = len(self.valid_fleets)
        if total == 0:
            raise RuntimeError("Belief state collapsed: zero consistent fleets remain.")
        probs: dict[Coord, float] = {}
        probed = self.hits | self.misses
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) in probed:
                    continue
                count = sum(
                    1 for fleet in self.valid_fleets if any((r, c) in s for s in fleet.values())
                )
                probs[(r, c)] = count / total
        return probs

    def select_action(self) -> tuple[Coord, float, float]:
        """Select unprobed cell with highest posterior occupancy probability (and compute entropy)."""
        probs = self.posterior_probabilities()
        best_coord = max(sorted(probs.keys()), key=lambda k: probs[k])
        p = probs[best_coord]
        entropy = 0.0 if p in (0.0, 1.0) else -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))
        return best_coord, p, entropy


def run_poc(seed: int = 4300) -> None:
    env = BattleshipEnv(rows=6, cols=6, seed=seed)
    agent = BayesianBeliefAgent(rows=6, cols=6, fleet_spec=env.fleet_spec)

    print(f"=== Battleship Bayesian Belief PoC (6x6, seed={seed}) ===")
    print(f"Initial consistent fleet hypotheses: {len(agent.valid_fleets)}\n")

    step = 0
    while not env.is_won:
        step += 1
        target, prob, entropy = agent.select_action()
        outcome = env.probe(target)
        agent.observe(outcome)
        status = f"HIT (SUNK {outcome.sunk_ship}!)" if outcome.sunk_ship else ("HIT" if outcome.hit else "MISS")
        print(
            f"Step {step:02d} | Probe {target} | P(occ)={prob:.3f} | H={entropy:.3f} bits | "
            f"Result={status:<22} | Remaining Hypotheses={len(agent.valid_fleets)}"
        )

    print(f"\nFleet destroyed in {step} shots (out of 36 cells).")


if __name__ == "__main__":
    run_poc(seed=4300)