"""Bayesian belief-tracking agents for partially observable Battleship.

Implements the U0-HW-10 approved strategy:
- ParticleFilterAgent: Sequential Monte Carlo (SMC) belief tracking with exact bitmask
  likelihood filtering, MRV hit-covering backtracking repair on particle deprivation,
  exact single-ship Gibbs + coupled two-ship MCMC rejuvenation, and automatic exact
  enumeration fallback in the endgame.
- HuntAndTargetAgent: Parity-based checkerboard search with orthogonal target queue
  (reference baseline from U0-HW-10).
- RandomAgent: Uniform random unprobed selection (lower-bound baseline from U0-HW-10).
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from typing import Protocol

from battleship_env import Coord, ShotResult


class BeliefCollapseError(RuntimeError):
    """Raised when zero fleet hypotheses remain consistent with observed percepts."""


def binary_entropy(p: float) -> float:
    """Compute binary Shannon entropy H(p) in bits."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


@dataclass(frozen=True)
class CandidateAction:
    coord: Coord
    probability: float
    cell_entropy_bits: float

    def to_dict(self) -> dict[str, object]:
        return {
            "coord": [self.coord[0], self.coord[1]],
            "probability": round(self.probability, 6),
            "cell_entropy_bits": round(self.cell_entropy_bits, 6),
        }


@dataclass(frozen=True)
class ActionDecision:
    coord: Coord
    probability: float
    cell_entropy_bits: float
    decision_time_ms: float
    top_candidates: list[CandidateAction]
    belief_snapshot: dict[str, object]


@dataclass(frozen=True)
class BeliefUpdateMetrics:
    mode: str
    survived_particles: int
    particle_count: int
    unique_particles: int
    rejuvenated: bool
    repair_triggered: bool
    mcmc_moves_attempted: int
    mcmc_moves_accepted: int
    board_entropy_bits: float
    entropy_reduction_bits: float
    update_time_ms: float

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "survived_particles": self.survived_particles,
            "particle_count": self.particle_count,
            "unique_particles": self.unique_particles,
            "rejuvenated": self.rejuvenated,
            "repair_triggered": self.repair_triggered,
            "mcmc_moves_attempted": self.mcmc_moves_attempted,
            "mcmc_moves_accepted": self.mcmc_moves_accepted,
            "board_entropy_bits": round(self.board_entropy_bits, 6),
            "entropy_reduction_bits": round(self.entropy_reduction_bits, 6),
            "update_time_ms": round(self.update_time_ms, 3),
        }


class BattleshipAgent(Protocol):
    """Common interface for Battleship agents driven by the episode runner."""

    name: str

    def select_action(self) -> ActionDecision: ...

    def observe(self, result: ShotResult) -> BeliefUpdateMetrics: ...

    def inspect_belief(self) -> dict[str, object]: ...


class ParticleFilterAgent:
    """Sequential Bayesian belief-tracking agent using SMC + MCMC + exact fallback."""

    def __init__(
        self,
        rows: int,
        cols: int,
        fleet_spec: dict[str, int],
        num_particles: int = 250,
        mcmc_sweeps: int = 3,
        exact_threshold: int = 4000,
        exact_product_limit: int = 60000,
        seed: int = 4300,
    ) -> None:
        if num_particles <= 0:
            raise ValueError(f"num_particles must be positive; got {num_particles}.")
        self.name = "smc_max_prob"
        self.rows = rows
        self.cols = cols
        self.fleet_spec = dict(fleet_spec)
        self.ship_names: list[str] = list(fleet_spec.keys())
        self.num_ships = len(self.ship_names)
        self._ship_index: dict[str, int] = {
            name: idx for idx, name in enumerate(self.ship_names)
        }
        self.num_particles = num_particles
        self.mcmc_sweeps = mcmc_sweeps
        self.exact_threshold = exact_threshold
        self.exact_product_limit = exact_product_limit
        self.seed = seed
        self._rng = random.Random(seed)

        # Observation state stored as both sets (for inspection) and bitmasks (for speed)
        self.hits: set[Coord] = set()
        self.misses: set[Coord] = set()
        self.sunk_ships: set[str] = set()
        self.hits_mask: int = 0
        self.misses_mask: int = 0
        self.probed_mask: int = 0
        # ship_idx -> (sunk_trigger_bit, cumulative_hits_mask_when_sunk)
        self._sunk_records: dict[int, tuple[int, int]] = {}

        # Precompute bitmasks for every valid single-ship placement
        self._all_placements: list[list[int]] = [
            self._generate_single_placements(fleet_spec[name])
            for name in self.ship_names
        ]

        # Active belief representation
        self.mode: str = "SMC"
        self.particles: list[tuple[int, ...]] = []
        self._cached_probs: dict[Coord, float] | None = None
        self._last_board_entropy: float = 0.0

        self._initialize_belief()

    def _coord_to_bit(self, coord: Coord) -> int:
        return 1 << (coord[0] * self.cols + coord[1])

    def _generate_single_placements(self, length: int) -> list[int]:
        placements: list[int] = []
        if length == 1:
            for r in range(self.rows):
                for c in range(self.cols):
                    placements.append(self._coord_to_bit((r, c)))
            return placements
        for r in range(self.rows):
            for c in range(self.cols - length + 1):
                mask = 0
                for i in range(length):
                    mask |= self._coord_to_bit((r, c + i))
                placements.append(mask)
        for r in range(self.rows - length + 1):
            for c in range(self.cols):
                mask = 0
                for i in range(length):
                    mask |= self._coord_to_bit((r + i, c))
                placements.append(mask)
        return placements

    def _valid_domains(self) -> list[list[int]] | None:
        """Filter each ship's single-ship placements against current observations."""
        domains: list[list[int]] = []
        for idx in range(self.num_ships):
            if idx in self._sunk_records:
                sunk_bit, sink_hits_mask = self._sunk_records[idx]
                valid = [
                    p
                    for p in self._all_placements[idx]
                    if (p & sunk_bit)
                    and (p & sink_hits_mask) == p
                    and (p & self.misses_mask) == 0
                ]
            else:
                valid = [
                    p
                    for p in self._all_placements[idx]
                    if (p & self.misses_mask) == 0
                    and (p & self.hits_mask) != p
                ]
            if not valid:
                return None
            domains.append(valid)
        return domains

    def _is_particle_consistent(
        self,
        particle: tuple[int, ...],
        domain_sets: list[set[int]],
    ) -> bool:
        occupied = 0
        for idx, p in enumerate(particle):
            if p not in domain_sets[idx]:
                return False
            if occupied & p:
                return False
            occupied |= p
        return (occupied & self.hits_mask) == self.hits_mask

    def _try_exact_enumeration(
        self, domains: list[list[int]]
    ) -> list[tuple[int, ...]] | None:
        """Attempt exact enumeration when the pruned hypothesis space is small.

        Returns:
            list of all consistent fleets if count <= exact_threshold (including empty list
            if proven unsatisfiable); None if the search exceeds exact_threshold or the
            domain product is above the attempt threshold.
        """
        unsunk_count = self.num_ships - len(self._sunk_records)
        domain_product = math.prod(len(d) for d in domains)
        limit = self.exact_product_limit * 2 if unsunk_count <= 2 else self.exact_product_limit
        if domain_product > limit:
            return None

        solutions: list[tuple[int, ...]] = []
        placed = [0] * self.num_ships
        all_indices = tuple(range(self.num_ships))
        aborted = False

        def backtrack(remaining: tuple[int, ...], occupied: int) -> None:
            nonlocal aborted
            if aborted:
                return
            if not remaining:
                if (occupied & self.hits_mask) == self.hits_mask:
                    solutions.append(tuple(placed))
                    if len(solutions) > self.exact_threshold:
                        aborted = True
                return

            uncovered_hits = self.hits_mask & ~occupied
            cands: dict[int, list[int]] = {}
            coverable = 0
            for idx in remaining:
                valid_p = [p for p in domains[idx] if (p & occupied) == 0]
                if not valid_p:
                    return
                cands[idx] = valid_p
                if uncovered_hits:
                    for p in valid_p:
                        coverable |= p

            if uncovered_hits and (coverable & uncovered_hits) != uncovered_hits:
                return

            if uncovered_hits:
                # Branch on the most-constrained uncovered hit bit.
                best_bit = 0
                best_options: list[tuple[int, int]] | None = None
                bits = uncovered_hits
                while bits:
                    lsb = bits & -bits
                    bits ^= lsb
                    options = [
                        (idx, p)
                        for idx in remaining
                        for p in cands[idx]
                        if p & lsb
                    ]
                    if not options:
                        return
                    if best_options is None or len(options) < len(best_options):
                        best_bit = lsb
                        best_options = options
                        if len(best_options) == 1:
                            break
                assert best_bit and best_options is not None
                for idx, p in best_options:
                    placed[idx] = p
                    next_rem = tuple(i for i in remaining if i != idx)
                    backtrack(next_rem, occupied | p)
                    if aborted:
                        return
                return

            # No uncovered hits remain: pick ship with minimum remaining values (MRV).
            next_idx = min(remaining, key=lambda i: len(cands[i]))
            next_rem = tuple(i for i in remaining if i != next_idx)
            for p in cands[next_idx]:
                placed[next_idx] = p
                backtrack(next_rem, occupied | p)
                if aborted:
                    return

        backtrack(all_indices, 0)
        if aborted:
            return None
        return solutions

    def _sample_consistent_particle(
        self, domains: list[list[int]], max_nodes: int = 25000
    ) -> tuple[tuple[int, ...] | None, bool]:
        """Sample one consistent fleet via randomized MRV hit-covering backtracking.

        Returns (particle_or_None, exhausted_entire_tree).
        """
        placed = [0] * self.num_ships
        nodes = 0
        hit_limit = False

        def backtrack(remaining: tuple[int, ...], occupied: int) -> bool:
            nonlocal nodes, hit_limit
            nodes += 1
            if nodes > max_nodes:
                hit_limit = True
                return False
            if not remaining:
                return (occupied & self.hits_mask) == self.hits_mask

            uncovered_hits = self.hits_mask & ~occupied
            cands: dict[int, list[int]] = {}
            coverable = 0
            for idx in remaining:
                valid_p = [p for p in domains[idx] if (p & occupied) == 0]
                if not valid_p:
                    return False
                cands[idx] = valid_p
                if uncovered_hits:
                    for p in valid_p:
                        coverable |= p

            if uncovered_hits and (coverable & uncovered_hits) != uncovered_hits:
                return False

            if uncovered_hits:
                best_options: list[tuple[int, int]] | None = None
                bits = uncovered_hits
                while bits:
                    lsb = bits & -bits
                    bits ^= lsb
                    options = [
                        (idx, p)
                        for idx in remaining
                        for p in cands[idx]
                        if p & lsb
                    ]
                    if not options:
                        return False
                    if best_options is None or len(options) < len(best_options):
                        best_options = options
                        if len(best_options) == 1:
                            break
                assert best_options is not None
                self._rng.shuffle(best_options)
                for idx, p in best_options:
                    placed[idx] = p
                    next_rem = tuple(i for i in remaining if i != idx)
                    if backtrack(next_rem, occupied | p):
                        return True
                    if hit_limit:
                        return False
                return False

            next_idx = min(remaining, key=lambda i: len(cands[i]))
            next_rem = tuple(i for i in remaining if i != next_idx)
            choices = list(cands[next_idx])
            self._rng.shuffle(choices)
            for p in choices:
                placed[next_idx] = p
                if backtrack(next_rem, occupied | p):
                    return True
                if hit_limit:
                    return False
            return False

        found = backtrack(tuple(range(self.num_ships)), 0)
        if found:
            return tuple(placed), False
        return None, (not hit_limit)

    def _rejuvenate_mcmc(
        self,
        particles: list[tuple[int, ...]],
        domains: list[list[int]],
        sweeps: int,
    ) -> tuple[list[tuple[int, ...]], int, int]:
        """Apply exact single-ship Gibbs sweeps and coupled two-ship block moves."""
        attempted = 0
        accepted = 0
        rejuvenated: list[tuple[int, ...]] = []
        ship_order = list(range(self.num_ships))
        unsunk_indices = [i for i in range(self.num_ships) if i not in self._sunk_records]

        for particle in particles:
            state = list(particle)
            total_occ = 0
            for p in state:
                total_occ |= p

            for _ in range(sweeps):
                self._rng.shuffle(ship_order)
                for idx in ship_order:
                    old_p = state[idx]
                    other_mask = total_occ ^ old_p
                    must_cover = self.hits_mask & ~other_mask
                    valid = [
                        p
                        for p in domains[idx]
                        if (p & other_mask) == 0 and (p & must_cover) == must_cover
                    ]
                    attempted += 1
                    if valid:
                        new_p = self._rng.choice(valid)
                        if new_p != old_p:
                            accepted += 1
                        state[idx] = new_p
                        total_occ = other_mask | new_p

                # Occasional coupled two-ship move when active hits constrain unsunk ships
                if self.hits_mask and len(unsunk_indices) >= 2:
                    i, j = self._rng.sample(unsunk_indices, 2)
                    other_mask = total_occ ^ state[i] ^ state[j]
                    must_cover = self.hits_mask & ~other_mask
                    if must_cover:
                        attempted += 1
                        cands_i = [
                            p for p in domains[i] if (p & other_mask) == 0
                        ]
                        cands_j = [
                            p for p in domains[j] if (p & other_mask) == 0
                        ]
                        # Focus on placements interacting with must_cover to keep block Gibbs fast
                        hitting_i = [p for p in cands_i if p & must_cover]
                        hitting_j = [p for p in cands_j if p & must_cover]
                        pairs: list[tuple[int, int]] = []
                        if len(hitting_i) * len(cands_j) <= 400:
                            for pi in hitting_i:
                                rem = must_cover & ~pi
                                for pj in cands_j:
                                    if (pi & pj) == 0 and (pj & rem) == rem:
                                        pairs.append((pi, pj))
                        if len(hitting_j) * len(cands_i) <= 400:
                            seen = set(pairs)
                            for pj in hitting_j:
                                rem = must_cover & ~pj
                                for pi in cands_i:
                                    if (
                                        (pi & pj) == 0
                                        and (pi & rem) == rem
                                        and (pi, pj) not in seen
                                    ):
                                        pairs.append((pi, pj))
                                        seen.add((pi, pj))
                        if pairs:
                            new_i, new_j = self._rng.choice(pairs)
                            if (new_i, new_j) != (state[i], state[j]):
                                accepted += 1
                            state[i] = new_i
                            state[j] = new_j
                            total_occ = other_mask | new_i | new_j

            rejuvenated.append(tuple(state))
        return rejuvenated, attempted, accepted

    def _initialize_belief(self) -> None:
        domains = self._valid_domains()
        if domains is None:
            raise BeliefCollapseError(
                "Initial fleet manifest has no valid placements on the configured grid."
            )

        exact = self._try_exact_enumeration(domains)
        if exact is not None:
            if not exact:
                raise BeliefCollapseError(
                    "Initial fleet manifest has zero non-overlapping configurations."
                )
            self.mode = "EXACT_FALLBACK"
            self.particles = exact
        else:
            self.mode = "SMC"
            sampled: list[tuple[int, ...]] = []
            for _ in range(self.num_particles):
                p, exhausted = self._sample_consistent_particle(domains)
                if p is None:
                    if exhausted:
                        raise BeliefCollapseError(
                            "Zero consistent fleet configurations exist for initial board."
                        )
                    break
                sampled.append(p)
            if not sampled:
                raise BeliefCollapseError(
                    "Failed to initialize any valid fleet particles."
                )
            while len(sampled) < self.num_particles:
                sampled.append(self._rng.choice(sampled))
            self.particles = sampled

        self._cached_probs = None
        probs = self.posterior_probabilities()
        self._last_board_entropy = sum(binary_entropy(p) for p in probs.values())

    def posterior_probabilities(self) -> dict[Coord, float]:
        """Compute marginal posterior occupancy P(occ_{r,c} | history) for unprobed cells."""
        if self._cached_probs is not None:
            return dict(self._cached_probs)
        total = len(self.particles)
        if total == 0:
            raise BeliefCollapseError(
                "Belief state collapsed: zero consistent fleet hypotheses remain."
            )

        counts = [0] * (self.rows * self.cols)
        for particle in self.particles:
            occ = 0
            for mask in particle:
                occ |= mask
            bits = occ & ~self.probed_mask
            while bits:
                lsb = bits & -bits
                counts[lsb.bit_length() - 1] += 1
                bits ^= lsb

        probs: dict[Coord, float] = {}
        for r in range(self.rows):
            row_offset = r * self.cols
            for c in range(self.cols):
                idx = row_offset + c
                if not (self.probed_mask & (1 << idx)):
                    probs[(r, c)] = counts[idx] / total
        self._cached_probs = probs
        return dict(probs)

    def inspect_belief(self) -> dict[str, object]:
        """Return an inspectable summary of current agent belief state."""
        remaining_ships = [
            name for name in self.ship_names if name not in self.sunk_ships
        ]
        return {
            "agent": self.name,
            "mode": self.mode,
            "particle_count": len(self.particles),
            "unique_particles": len(set(self.particles)),
            "hits_count": len(self.hits),
            "misses_count": len(self.misses),
            "sunk_ships": sorted(self.sunk_ships),
            "remaining_ships": remaining_ships,
            "board_entropy_bits": round(self._last_board_entropy, 6),
        }

    def select_action(self) -> ActionDecision:
        """Select unprobed cell maximizing posterior marginal occupancy probability."""
        t0 = time.perf_counter()
        probs = self.posterior_probabilities()
        if not probs:
            raise BeliefCollapseError("No unprobed coordinates remain to select.")

        ranked = sorted(
            probs.items(),
            key=lambda kv: (-kv[1], kv[0][0], kv[0][1]),
        )
        best_coord, best_prob = ranked[0]
        top_candidates = [
            CandidateAction(
                coord=coord,
                probability=prob,
                cell_entropy_bits=binary_entropy(prob),
            )
            for coord, prob in ranked[:5]
        ]
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return ActionDecision(
            coord=best_coord,
            probability=best_prob,
            cell_entropy_bits=binary_entropy(best_prob),
            decision_time_ms=elapsed_ms,
            top_candidates=top_candidates,
            belief_snapshot=self.inspect_belief(),
        )

    def observe(self, result: ShotResult) -> BeliefUpdateMetrics:
        """Condition the belief distribution on a new ShotResult percept."""
        t0 = time.perf_counter()
        prev_entropy = self._last_board_entropy
        bit = self._coord_to_bit(result.coord)

        if result.sunk_ship is not None:
            if not result.hit:
                raise BeliefCollapseError(
                    f"Contradictory percept: ship '{result.sunk_ship}' reported sunk on a MISS at {result.coord}."
                )
            if result.sunk_ship not in self._ship_index:
                raise BeliefCollapseError(
                    f"Unknown ship '{result.sunk_ship}' reported sunk at {result.coord}."
                )
            if result.sunk_ship in self.sunk_ships:
                raise BeliefCollapseError(
                    f"Contradictory percept: ship '{result.sunk_ship}' reported sunk more than once."
                )

        self.probed_mask |= bit
        if result.hit:
            self.hits.add(result.coord)
            self.hits_mask |= bit
            if result.sunk_ship is not None:
                self.sunk_ships.add(result.sunk_ship)
                ship_idx = self._ship_index[result.sunk_ship]
                self._sunk_records[ship_idx] = (bit, self.hits_mask)
        else:
            self.misses.add(result.coord)
            self.misses_mask |= bit

        domains = self._valid_domains()
        if domains is None:
            self.particles = []
            self._cached_probs = None
            raise BeliefCollapseError(
                f"Belief state collapsed after {result}: at least one ship has zero consistent placements."
            )

        domain_sets = [set(d) for d in domains]
        survivors = [
            p for p in self.particles if self._is_particle_consistent(p, domain_sets)
        ]
        survived_count = len(survivors)

        rejuvenated = False
        repair_triggered = False
        moves_attempted = 0
        moves_accepted = 0

        if self.mode == "EXACT_FALLBACK":
            if survived_count == 0:
                self.particles = []
                self._cached_probs = None
                raise BeliefCollapseError(
                    f"Exact belief state collapsed after {result}: zero consistent fleet hypotheses remain."
                )
            self.particles = survivors
        else:
            # Check whether the pruned hypothesis space is now small enough for exact endgame inference
            exact = self._try_exact_enumeration(domains)
            if exact is not None:
                if len(exact) == 0:
                    self.particles = []
                    self._cached_probs = None
                    raise BeliefCollapseError(
                        f"Belief state collapsed after {result}: exact solver proved zero consistent fleets exist."
                    )
                self.mode = "EXACT_FALLBACK"
                self.particles = exact
            else:
                # Stay in SMC mode; repair if all particles were eliminated, rejuvenate if depleted
                if survived_count == 0:
                    repair_triggered = True
                    seeds: list[tuple[int, ...]] = []
                    target_seeds = min(self.num_particles, 25)
                    for _ in range(target_seeds):
                        sample, exhausted = self._sample_consistent_particle(domains)
                        if sample is None:
                            if exhausted:
                                self.particles = []
                                self._cached_probs = None
                                raise BeliefCollapseError(
                                    f"Belief state collapsed after {result}: constraint repair proved zero consistent fleets exist."
                                )
                            break
                        seeds.append(sample)
                    if not seeds:
                        self.particles = []
                        self._cached_probs = None
                        raise BeliefCollapseError(
                            f"Belief state collapsed after {result}: particle repair failed to recover a consistent fleet."
                        )
                    rejuvenated = True
                    resampled = [
                        self._rng.choice(seeds) for _ in range(self.num_particles)
                    ]
                    self.particles, moves_attempted, moves_accepted = (
                        self._rejuvenate_mcmc(
                            resampled, domains, sweeps=self.mcmc_sweeps + 1
                        )
                    )
                elif survived_count < self.num_particles:
                    rejuvenated = True
                    deficit = self.num_particles - survived_count
                    clones = [self._rng.choice(survivors) for _ in range(deficit)]
                    mixed_clones, moves_attempted, moves_accepted = (
                        self._rejuvenate_mcmc(
                            clones, domains, sweeps=self.mcmc_sweeps
                        )
                    )
                    self.particles = survivors + mixed_clones
                else:
                    self.particles = survivors

        self._cached_probs = None
        probs = self.posterior_probabilities() if len(self.sunk_ships) < self.num_ships else {}
        new_entropy = sum(binary_entropy(p) for p in probs.values())
        self._last_board_entropy = new_entropy
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        return BeliefUpdateMetrics(
            mode=self.mode,
            survived_particles=survived_count,
            particle_count=len(self.particles),
            unique_particles=len(set(self.particles)),
            rejuvenated=rejuvenated,
            repair_triggered=repair_triggered,
            mcmc_moves_attempted=moves_attempted,
            mcmc_moves_accepted=moves_accepted,
            board_entropy_bits=new_entropy,
            entropy_reduction_bits=prev_entropy - new_entropy,
            update_time_ms=elapsed_ms,
        )

    def render_probability_heatmap(self) -> str:
        """Render an ASCII grid of posterior cell probabilities (0-99%) and probed marks."""
        probs = self.posterior_probabilities()
        header = "    " + "  ".join(f"{c:2d}" for c in range(self.cols))
        lines = [header]
        for r in range(self.rows):
            cells: list[str] = []
            for c in range(self.cols):
                coord = (r, c)
                if coord in self.hits:
                    cells.append("XX")
                elif coord in self.misses:
                    cells.append("--")
                else:
                    pct = int(round(probs.get(coord, 0.0) * 99))
                    cells.append(f"{pct:2d}")
            lines.append(f"{r:2d}  " + "  ".join(cells))
        return "\n".join(lines)


class HuntAndTargetAgent:
    """Parity checkerboard hunt + orthogonal neighbor target baseline (from U0-HW-10)."""

    def __init__(
        self,
        rows: int,
        cols: int,
        fleet_spec: dict[str, int],
        seed: int = 4300,
    ) -> None:
        self.name = "hunt_target"
        self.rows = rows
        self.cols = cols
        self.fleet_spec = dict(fleet_spec)
        self._rng = random.Random(seed)
        self.probed: set[Coord] = set()
        self.hits: set[Coord] = set()
        self.sunk_ships: set[str] = set()
        self.target_queue: list[Coord] = []
        self._parity_cells: list[Coord] = [
            (r, c)
            for r in range(rows)
            for c in range(cols)
            if (r + c) % 2 == 0
        ]
        self._off_parity_cells: list[Coord] = [
            (r, c)
            for r in range(rows)
            for c in range(cols)
            if (r + c) % 2 == 1
        ]
        self._rng.shuffle(self._parity_cells)
        self._rng.shuffle(self._off_parity_cells)

    def inspect_belief(self) -> dict[str, object]:
        return {
            "agent": self.name,
            "mode": "TARGET" if self.target_queue else "HUNT",
            "target_queue_len": len(self.target_queue),
            "hits_count": len(self.hits),
            "sunk_ships": sorted(self.sunk_ships),
        }

    def select_action(self) -> ActionDecision:
        t0 = time.perf_counter()
        chosen: Coord | None = None
        while self.target_queue:
            cand = self.target_queue.pop(0)
            if cand not in self.probed:
                chosen = cand
                break
        if chosen is None:
            for pool in (self._parity_cells, self._off_parity_cells):
                for coord in pool:
                    if coord not in self.probed:
                        chosen = coord
                        break
                if chosen is not None:
                    break
        if chosen is None:
            raise BeliefCollapseError("No unprobed coordinates remain.")

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return ActionDecision(
            coord=chosen,
            probability=0.0,
            cell_entropy_bits=0.0,
            decision_time_ms=elapsed_ms,
            top_candidates=[CandidateAction(chosen, 0.0, 0.0)],
            belief_snapshot=self.inspect_belief(),
        )

    def observe(self, result: ShotResult) -> BeliefUpdateMetrics:
        t0 = time.perf_counter()
        self.probed.add(result.coord)
        if result.hit:
            self.hits.add(result.coord)
            if result.sunk_ship:
                self.sunk_ships.add(result.sunk_ship)
            r, c = result.coord
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (
                    0 <= nr < self.rows
                    and 0 <= nc < self.cols
                    and (nr, nc) not in self.probed
                    and (nr, nc) not in self.target_queue
                ):
                    self.target_queue.append((nr, nc))
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return BeliefUpdateMetrics(
            mode="TARGET" if self.target_queue else "HUNT",
            survived_particles=0,
            particle_count=0,
            unique_particles=0,
            rejuvenated=False,
            repair_triggered=False,
            mcmc_moves_attempted=0,
            mcmc_moves_accepted=0,
            board_entropy_bits=0.0,
            entropy_reduction_bits=0.0,
            update_time_ms=elapsed_ms,
        )


class RandomAgent:
    """Uniform random unprobed selection baseline (from U0-HW-10)."""

    def __init__(
        self,
        rows: int,
        cols: int,
        fleet_spec: dict[str, int],
        seed: int = 4300,
    ) -> None:
        self.name = "random"
        self.rows = rows
        self.cols = cols
        self.fleet_spec = dict(fleet_spec)
        self._rng = random.Random(seed)
        self._order: list[Coord] = [
            (r, c) for r in range(rows) for c in range(cols)
        ]
        self._rng.shuffle(self._order)
        self._cursor = 0
        self.probed: set[Coord] = set()

    def inspect_belief(self) -> dict[str, object]:
        return {
            "agent": self.name,
            "mode": "RANDOM",
            "remaining_cells": len(self._order) - self._cursor,
        }

    def select_action(self) -> ActionDecision:
        t0 = time.perf_counter()
        while self._cursor < len(self._order) and self._order[self._cursor] in self.probed:
            self._cursor += 1
        if self._cursor >= len(self._order):
            raise BeliefCollapseError("No unprobed coordinates remain.")
        chosen = self._order[self._cursor]
        self._cursor += 1
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return ActionDecision(
            coord=chosen,
            probability=0.0,
            cell_entropy_bits=0.0,
            decision_time_ms=elapsed_ms,
            top_candidates=[CandidateAction(chosen, 0.0, 0.0)],
            belief_snapshot=self.inspect_belief(),
        )

    def observe(self, result: ShotResult) -> BeliefUpdateMetrics:
        t0 = time.perf_counter()
        self.probed.add(result.coord)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return BeliefUpdateMetrics(
            mode="RANDOM",
            survived_particles=0,
            particle_count=0,
            unique_particles=0,
            rejuvenated=False,
            repair_triggered=False,
            mcmc_moves_attempted=0,
            mcmc_moves_accepted=0,
            board_entropy_bits=0.0,
            entropy_reduction_bits=0.0,
            update_time_ms=elapsed_ms,
        )
