"""Agent protocol, provided baselines, and knowledge-based LogicAgent."""

from __future__ import annotations

import heapq
import random
from typing import Iterable, Mapping, Protocol, Sequence

from proplogic import KnowledgeBase, atom
from rover import Action, Percept, Position, ScanObservation, Scanner


class RoverAgent(Protocol):
    def reset(self) -> None:
        """Clear episode-specific state."""
        ...

    def act(self, percept: Percept) -> Action:
        """Choose one action from the current percept and observation history."""
        ...


class StopAgent:
    """Minimal interface example that immediately ends the mission."""

    def reset(self) -> None:
        pass

    def act(self, percept: Percept) -> Action:
        del percept
        return Action.stop()


class RandomAgent:
    """Seeded baseline that does not reason about scanner evidence."""

    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def reset(self) -> None:
        pass

    def act(self, percept: Percept) -> Action:
        actions = self._candidate_actions(percept)
        return self._rng.choice(actions)

    def _candidate_actions(self, percept: Percept) -> list[Action]:
        actions = [Action.retrieve(), Action.stop()]
        actions.append(
            Action.disable_traction()
            if percept.traction_enabled
            else Action.enable_traction()
        )

        row = percept.position.row
        column = percept.position.column
        for candidate in (
            Position(row - 1, column),
            Position(row + 1, column),
            Position(row, column - 1),
            Position(row, column + 1),
        ):
            if 0 <= candidate.row < percept.rows and 0 <= candidate.column < percept.columns:
                actions.append(Action.move(candidate))

        for scan_row in range(percept.rows):
            for scan_column in range(percept.columns - 2):
                actions.append(
                    Action.structural_scan(
                        *(Position(scan_row, scan_column + offset) for offset in range(3))
                    )
                )
        for scan_row in range(percept.rows - 2):
            for scan_column in range(percept.columns):
                actions.append(
                    Action.structural_scan(
                        *(Position(scan_row + offset, scan_column) for offset in range(3))
                    )
                )
        for scan_row in range(percept.rows - 1):
            for scan_column in range(percept.columns - 1):
                actions.append(
                    Action.surface_scan(
                        Position(scan_row, scan_column),
                        Position(scan_row, scan_column + 1),
                        Position(scan_row + 1, scan_column),
                        Position(scan_row + 1, scan_column + 1),
                    )
                )
        return actions


def stable_proposition(position: Position) -> str:
    """Return the proposition name claiming that position is stable."""
    return f"stable_{position.row}_{position.column}"


def slippery_proposition(position: Position) -> str:
    """Return the proposition name claiming that position is slippery."""
    return f"slippery_{position.row}_{position.column}"


def fragile_proposition(position: Position) -> str:
    """Return the proposition name claiming that position is fragile."""
    return f"fragile_{position.row}_{position.column}"


def add_terrain_rules(
    kb: KnowledgeBase,
    rows: int,
    columns: int,
    start: Position | None = None,
) -> None:
    """Add mutually exclusive and exhaustive terrain rules for every mapped tile."""
    if rows <= 0 or columns <= 0:
        raise ValueError("rows and columns must be positive")

    for row in range(rows):
        for column in range(columns):
            position = Position(row, column)
            stable = atom(stable_proposition(position))
            slippery = atom(slippery_proposition(position))
            fragile = atom(fragile_proposition(position))

            # 1. At least one of stable, slippery, and fragile is true.
            kb.add_clause(stable, slippery, fragile)
            # 2. stable and slippery cannot both be true.
            kb.add_clause(~stable, ~slippery)
            # 3. stable and fragile cannot both be true.
            kb.add_clause(~stable, ~fragile)
            # 4. slippery and fragile cannot both be true.
            kb.add_clause(~slippery, ~fragile)

    if start is not None:
        if not (0 <= start.row < rows and 0 <= start.column < columns):
            raise ValueError("start position must be within the mapped grid")
        kb.add_fact(atom(stable_proposition(start)))


def scan_observation_to_formulas(observation: ScanObservation) -> tuple[str, ...]:
    """Translate a ScanObservation into propositional logic formula strings."""
    if not observation.region:
        raise ValueError("scan observation region cannot be empty")

    if observation.scanner is Scanner.STRUCTURAL:
        prop_fn = fragile_proposition
    elif observation.scanner is Scanner.SURFACE:
        prop_fn = slippery_proposition
    else:
        raise ValueError(f"unsupported scanner type: {observation.scanner!r}")

    if observation.positive:
        return (" OR ".join(prop_fn(position) for position in observation.region),)
    return tuple(f"NOT {prop_fn(position)}" for position in observation.region)


def record_scan_observation(
    kb: KnowledgeBase, observation: ScanObservation
) -> None:
    """Translate a scan observation, add it to the KB, and verify KB consistency."""
    for formula in scan_observation_to_formulas(observation):
        kb.add_formula(formula)
    if not kb.is_consistent():
        raise RuntimeError(
            f"Knowledge base became inconsistent after observation: {observation!r}"
        )


def is_non_fragile(kb: KnowledgeBase, position: Position) -> bool:
    """Return whether the knowledge base entails that position is not fragile."""
    return kb.entails(~atom(fragile_proposition(position)))


def is_fragile(kb: KnowledgeBase, position: Position) -> bool:
    """Return whether the knowledge base entails that position is fragile."""
    return kb.entails(atom(fragile_proposition(position)))


def is_stable(kb: KnowledgeBase, position: Position) -> bool:
    """Return whether the knowledge base entails that position is stable."""
    return kb.entails(atom(stable_proposition(position)))


def is_slippery(kb: KnowledgeBase, position: Position) -> bool:
    """Return whether the knowledge base entails that position is slippery."""
    return kb.entails(atom(slippery_proposition(position)))


def valid_structural_scan_regions(
    rows: int, columns: int
) -> tuple[tuple[Position, ...], ...]:
    """Return all valid 3-tile horizontal and vertical structural scan regions."""
    regions: list[tuple[Position, ...]] = []
    for row in range(rows):
        for column in range(columns - 2):
            regions.append(
                tuple(Position(row, column + offset) for offset in range(3))
            )
    for row in range(rows - 2):
        for column in range(columns):
            regions.append(
                tuple(Position(row + offset, column) for offset in range(3))
            )
    return tuple(regions)


def valid_surface_scan_regions(
    rows: int, columns: int
) -> tuple[tuple[Position, ...], ...]:
    """Return all valid 2x2 square surface scan regions."""
    regions: list[tuple[Position, ...]] = []
    for row in range(rows - 1):
        for column in range(columns - 1):
            regions.append(
                (
                    Position(row, column),
                    Position(row, column + 1),
                    Position(row + 1, column),
                    Position(row + 1, column + 1),
                )
            )
    return tuple(regions)


def _neighbors(position: Position, rows: int | None = None, columns: int | None = None) -> tuple[Position, ...]:
    candidates = (
        Position(position.row - 1, position.column),
        Position(position.row + 1, position.column),
        Position(position.row, position.column - 1),
        Position(position.row, position.column + 1),
    )
    if rows is None or columns is None:
        return candidates
    return tuple(
        candidate
        for candidate in candidates
        if 0 <= candidate.row < rows and 0 <= candidate.column < columns
    )


def astar_search(
    start: Position,
    goal: Position,
    non_fragile: Iterable[Position],
    stable: Iterable[Position] = (),
    slippery: Iterable[Position] = (),
) -> tuple[Position, ...] | None:
    """Find an optimal path from start to goal over inferred non-fragile tiles using A*."""
    allowed = set(non_fragile)
    if start not in allowed or goal not in allowed:
        return None
    if start == goal:
        return (start,)

    stable_set = set(stable)
    slippery_set = set(slippery)
    use_surface_weights = bool(stable_set or slippery_set)

    def step_cost(target: Position) -> int:
        if not use_surface_weights or target in stable_set:
            return 3
        if target in slippery_set:
            return 4
        return 5

    def heuristic(position: Position) -> int:
        return 3 * (
            abs(position.row - goal.row) + abs(position.column - goal.column)
        )

    frontier: list[tuple[int, int, Position]] = [(heuristic(start), 0, start)]
    came_from: dict[Position, Position] = {}
    g_score: dict[Position, int] = {start: 0}

    while frontier:
        _, current_g, current = heapq.heappop(frontier)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return tuple(path)

        if current_g > g_score.get(current, current_g):
            continue

        for neighbor in _neighbors(current):
            if neighbor not in allowed:
                continue
            tentative_g = current_g + step_cost(neighbor)
            if tentative_g < g_score.get(neighbor, tentative_g + 1):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor)
                heapq.heappush(frontier, (f_score, tentative_g, neighbor))

    return None


def route_to_actions(
    start_to_module: Sequence[Position],
    module_to_exit: Sequence[Position] | None = None,
    *,
    stable_tiles: Iterable[Position] = (),
    initial_traction: bool = False,
    carrying_module: bool = False,
) -> tuple[Action, ...]:
    """Convert planned route legs into movement, traction, retrieval, and stop actions."""
    if not start_to_module:
        raise ValueError("start_to_module route cannot be empty")
    if module_to_exit is not None:
        if not module_to_exit:
            raise ValueError("module_to_exit route cannot be empty")
        if start_to_module[-1] != module_to_exit[0]:
            raise ValueError(
                "start_to_module must end at the start of module_to_exit"
            )

    stable_set = set(stable_tiles)
    traction = initial_traction
    actions: list[Action] = []

    def append_leg_moves(leg: Sequence[Position]) -> None:
        nonlocal traction
        for index in range(1, len(leg)):
            current = leg[index - 1]
            target = leg[index]
            distance = abs(target.row - current.row) + abs(
                target.column - current.column
            )
            if distance != 1:
                raise ValueError(
                    f"Route positions must be adjacent: {current!r} -> {target!r}"
                )
            needs_traction = target not in stable_set
            if needs_traction and not traction:
                actions.append(Action.enable_traction())
                traction = True
            elif not needs_traction and traction:
                actions.append(Action.disable_traction())
                traction = False
            actions.append(Action.move(target))

    append_leg_moves(start_to_module)
    if not carrying_module:
        actions.append(Action.retrieve())
    if module_to_exit is not None:
        append_leg_moves(module_to_exit)
    actions.append(Action.stop())
    return tuple(actions)


def _optimistic_leg_path(
    start: Position,
    goal: Position,
    rows: int,
    columns: int,
    entailed_non_fragile: set[Position],
    certifiable_unknowns: set[Position],
    clean_window_counts: Mapping[Position, int],
    positive_structural_counts: Mapping[Position, int],
) -> tuple[Position, ...] | None:
    """Find a path from start to goal preferring known non-fragile and clean unknown tiles."""
    allowed = entailed_non_fragile | certifiable_unknowns
    if start not in allowed or goal not in allowed:
        return None
    if start == goal:
        return (start,)

    def step_cost(target: Position) -> int:
        if target in entailed_non_fragile:
            return 1
        pos_hits = positive_structural_counts.get(target, 0)
        no_clean_penalty = 0 if clean_window_counts.get(target, 0) > 0 else 25
        return 10 + 40 * pos_hits + no_clean_penalty

    def heuristic(position: Position) -> int:
        return abs(position.row - goal.row) + abs(position.column - goal.column)

    frontier: list[tuple[int, int, Position]] = [(heuristic(start), 0, start)]
    came_from: dict[Position, Position] = {}
    g_score: dict[Position, int] = {start: 0}

    while frontier:
        _, current_g, current = heapq.heappop(frontier)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return tuple(path)

        if current_g > g_score.get(current, current_g):
            continue

        for neighbor in _neighbors(current, rows, columns):
            if neighbor not in allowed:
                continue
            tentative_g = current_g + step_cost(neighbor)
            if tentative_g < g_score.get(neighbor, tentative_g + 1):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic(neighbor)
                heapq.heappush(frontier, (f_score, tentative_g, neighbor))

    return None


def select_structural_scan_region(
    rows: int,
    columns: int,
    start: Position,
    module: Position,
    exit_pos: Position,
    entailed_non_fragile: set[Position],
    entailed_fragile: set[Position],
    scanned_structural: set[tuple[Position, ...]],
    positive_structural_counts: Mapping[Position, int],
    need_start_to_module: bool,
    need_module_to_exit: bool,
) -> tuple[Position, ...] | None:
    """Choose the next structural scan region to certify missing mission route legs."""
    all_regions = valid_structural_scan_regions(rows, columns)
    unscanned = [
        region for region in all_regions if region not in scanned_structural
    ]
    if not unscanned:
        return None

    viable = [
        region
        for region in unscanned
        if all(pos not in entailed_fragile for pos in region)
        and any(pos not in entailed_non_fragile for pos in region)
    ]
    if not viable:
        return unscanned[0]

    certifiable_unknowns: set[Position] = set()
    clean_window_counts: dict[Position, int] = {}
    for region in viable:
        is_clean = all(
            positive_structural_counts.get(pos, 0) == 0 for pos in region
        )
        for pos in region:
            if pos not in entailed_non_fragile:
                certifiable_unknowns.add(pos)
                if is_clean:
                    clean_window_counts[pos] = clean_window_counts.get(pos, 0) + 1

    target_unknowns: set[Position] = set()
    if need_start_to_module:
        leg1 = _optimistic_leg_path(
            start,
            module,
            rows,
            columns,
            entailed_non_fragile,
            certifiable_unknowns,
            clean_window_counts,
            positive_structural_counts,
        )
        if leg1 is not None:
            target_unknowns.update(
                pos for pos in leg1 if pos not in entailed_non_fragile
            )

    if need_module_to_exit:
        leg2 = _optimistic_leg_path(
            module,
            exit_pos,
            rows,
            columns,
            entailed_non_fragile,
            certifiable_unknowns,
            clean_window_counts,
            positive_structural_counts,
        )
        if leg2 is not None:
            target_unknowns.update(
                pos for pos in leg2 if pos not in entailed_non_fragile
            )

    def region_priority(region: tuple[Position, ...]) -> tuple[int, ...]:
        target_hits = sum(1 for pos in region if pos in target_unknowns)
        unknown_hits = sum(1 for pos in region if pos not in entailed_non_fragile)
        non_target_unknowns = unknown_hits - target_hits
        positive_penalty = sum(
            positive_structural_counts.get(pos, 0) for pos in region
        )
        endpoint_hits = sum(
            1
            for pos in region
            if pos in (module, exit_pos) and pos not in entailed_non_fragile
        )
        return (
            1 if target_hits > 0 else 0,
            -positive_penalty,
            target_hits,
            endpoint_hits,
            -non_target_unknowns,
            unknown_hits,
        )

    return max(viable, key=region_priority)


def select_surface_scan_region(
    rows: int,
    columns: int,
    route_moves: Sequence[Position],
    entailed_non_fragile: set[Position],
    entailed_stable: set[Position],
    entailed_slippery: set[Position],
    scanned_surface: set[tuple[Position, ...]],
    positive_surface_regions: Sequence[tuple[Position, ...]],
) -> tuple[Position, ...] | None:
    """Choose a 2x2 surface scan region when it covers at least 3 unknown-surface route moves."""
    unknown_route_counts: dict[Position, int] = {}
    for pos in route_moves:
        if pos not in entailed_stable and pos not in entailed_slippery:
            unknown_route_counts[pos] = unknown_route_counts.get(pos, 0) + 1

    if sum(unknown_route_counts.values()) < 3:
        return None

    positive_surface_tiles = {
        pos for region in positive_surface_regions for pos in region
    }
    all_regions = valid_surface_scan_regions(rows, columns)
    candidates: list[tuple[tuple[int, int, int], tuple[Position, ...]]] = []

    for region in all_regions:
        if region in scanned_surface:
            continue
        if any(pos in entailed_slippery for pos in region):
            continue
        if any(pos in positive_surface_tiles for pos in region):
            continue

        covered_moves = sum(unknown_route_counts.get(pos, 0) for pos in region)
        if covered_moves < 3:
            continue

        known_stable_count = sum(1 for pos in region if pos in entailed_stable)
        non_fragile_count = sum(1 for pos in region if pos in entailed_non_fragile)
        if non_fragile_count < 3:
            continue
        priority = (covered_moves, known_stable_count, non_fragile_count)
        candidates.append((priority, region))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


class LogicAgent:
    """Required knowledge-based rover agent implemented by students."""

    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.reset()

    def reset(self) -> None:
        self.knowledge = KnowledgeBase()
        self.initialized = False
        self.last_processed_action = -1
        self.scanned_structural: set[tuple[Position, ...]] = set()
        self.scanned_surface: set[tuple[Position, ...]] = set()
        self.positive_structural_regions: list[tuple[Position, ...]] = []
        self.positive_surface_regions: list[tuple[Position, ...]] = []
        self.negative_surface_tiles: set[Position] = set()
        self.positive_structural_counts: dict[Position, int] = {}
        self.entailed_non_fragile: set[Position] = set()
        self.entailed_fragile: set[Position] = set()
        self.entailed_stable: set[Position] = set()
        self.entailed_slippery: set[Position] = set()
        self.planned_actions: tuple[Action, ...] = ()
        self.plan_start_action_count: int | None = None

    def act(self, percept: Percept) -> Action:
        """Update propositional knowledge and choose one justified action."""
        if not self.initialized:
            add_terrain_rules(
                self.knowledge,
                percept.rows,
                percept.columns,
                start=percept.position,
            )
            if not self.knowledge.is_consistent():
                raise RuntimeError("Initial knowledge base is inconsistent")
            self.initialized = True
            self.negative_surface_tiles.add(percept.position)
            self._query_tile(percept.position, check_surface=True)

        if percept.action_count > self.last_processed_action:
            if (
                percept.last_outcome is not None
                and percept.last_outcome.scan is not None
            ):
                self._process_scan_observation(percept.last_outcome.scan)
            self.last_processed_action = percept.action_count

        if self.planned_actions and self.plan_start_action_count is not None:
            index = percept.action_count - self.plan_start_action_count
            if 0 <= index < len(self.planned_actions):
                return self.planned_actions[index]
            return Action.stop()

        start_to_module = astar_search(
            percept.position,
            percept.module_position,
            self.entailed_non_fragile,
            stable=self.entailed_stable,
            slippery=self.entailed_slippery,
        )
        module_to_exit = astar_search(
            percept.module_position,
            percept.exit_position,
            self.entailed_non_fragile,
            stable=self.entailed_stable,
            slippery=self.entailed_slippery,
        )

        if start_to_module is None or module_to_exit is None:
            scan_region = select_structural_scan_region(
                rows=percept.rows,
                columns=percept.columns,
                start=percept.position,
                module=percept.module_position,
                exit_pos=percept.exit_position,
                entailed_non_fragile=self.entailed_non_fragile,
                entailed_fragile=self.entailed_fragile,
                scanned_structural=self.scanned_structural,
                positive_structural_counts=self.positive_structural_counts,
                need_start_to_module=start_to_module is None,
                need_module_to_exit=module_to_exit is None,
            )
            if scan_region is not None:
                return Action.structural_scan(*scan_region)
            raise RuntimeError("No structural scan regions remain to certify a route")

        route_moves = (*start_to_module[1:], *module_to_exit[1:])
        # Only consider surface scans if plenty of action budget remains
        estimated_exec_actions = 2 * len(route_moves) + 3
        if percept.action_count + 1 + estimated_exec_actions < 95:
            surface_region = select_surface_scan_region(
                rows=percept.rows,
                columns=percept.columns,
                route_moves=route_moves,
                entailed_non_fragile=self.entailed_non_fragile,
                entailed_stable=self.entailed_stable,
                entailed_slippery=self.entailed_slippery,
                scanned_surface=self.scanned_surface,
                positive_surface_regions=self.positive_surface_regions,
            )
            if surface_region is not None:
                return Action.surface_scan(*surface_region)

        self.planned_actions = route_to_actions(
            start_to_module,
            module_to_exit,
            stable_tiles=self.entailed_stable,
            initial_traction=percept.traction_enabled,
            carrying_module=percept.carrying_module,
        )
        self.plan_start_action_count = percept.action_count
        return self.planned_actions[0]

    def _process_scan_observation(self, observation: ScanObservation) -> None:
        record_scan_observation(self.knowledge, observation)
        region = tuple(sorted(observation.region))

        if observation.scanner is Scanner.STRUCTURAL:
            self.scanned_structural.add(region)
            if observation.positive:
                self.positive_structural_regions.append(region)
                for pos in region:
                    self.positive_structural_counts[pos] = (
                        self.positive_structural_counts.get(pos, 0) + 1
                    )
            else:
                for pos in region:
                    self._query_tile(
                        pos,
                        check_surface=(pos in self.negative_surface_tiles),
                    )
        elif observation.scanner is Scanner.SURFACE:
            self.scanned_surface.add(region)
            if observation.positive:
                self.positive_surface_regions.append(region)
            else:
                self.negative_surface_tiles.update(region)
                for pos in region:
                    if pos in self.entailed_non_fragile:
                        self._query_tile(pos, check_surface=True)

        # Propagate overlapping positive structural and surface scans until fixed point
        changed = True
        while changed:
            changed = False
            for pos_region in self.positive_structural_regions:
                non_fragile_count = sum(
                    1 for pos in pos_region if pos in self.entailed_non_fragile
                )
                if non_fragile_count >= 2:
                    for pos in pos_region:
                        if (
                            pos not in self.entailed_non_fragile
                            and pos not in self.entailed_fragile
                        ):
                            before = len(self.entailed_fragile)
                            self._query_tile(pos, check_surface=False)
                            if len(self.entailed_fragile) > before:
                                changed = True

            for pos_region in self.positive_surface_regions:
                known_non_slippery = sum(
                    1
                    for pos in pos_region
                    if (
                        pos in self.negative_surface_tiles
                        or pos in self.entailed_stable
                        or pos in self.entailed_fragile
                    )
                )
                if known_non_slippery >= 3:
                    for pos in pos_region:
                        if (
                            pos not in self.entailed_slippery
                            and pos not in self.entailed_stable
                            and pos not in self.entailed_fragile
                        ):
                            before = len(self.entailed_slippery)
                            self._query_tile(pos, check_surface=True)
                            if len(self.entailed_slippery) > before:
                                changed = True

    def _query_tile(self, position: Position, check_surface: bool = False) -> None:
        if (
            position not in self.entailed_non_fragile
            and position not in self.entailed_fragile
        ):
            if is_non_fragile(self.knowledge, position):
                self.entailed_non_fragile.add(position)
            elif is_fragile(self.knowledge, position):
                self.entailed_fragile.add(position)

        if (
            check_surface
            and position not in self.entailed_fragile
            and position not in self.entailed_stable
            and position not in self.entailed_slippery
        ):
            if is_stable(self.knowledge, position):
                self.entailed_stable.add(position)
                self.entailed_non_fragile.add(position)
            elif (
                any(position in reg for reg in self.positive_surface_regions)
                and is_slippery(self.knowledge, position)
            ):
                self.entailed_slippery.add(position)
                self.entailed_non_fragile.add(position)
