"""Provided deterministic Rescue Rover scenarios for U0-HW-07."""

from __future__ import annotations

import random

from rover import Position, RoverScenario, Terrain


_SYMBOLS = {
    ".": Terrain.STABLE,
    "~": Terrain.SLIPPERY,
    "#": Terrain.FRAGILE,
}


def _scenario(
    identifier: str,
    rows: tuple[str, ...],
    start: tuple[int, int],
    module: tuple[int, int],
    exit_: tuple[int, int],
) -> RoverScenario:
    terrain = tuple(tuple(_SYMBOLS[symbol] for symbol in row) for row in rows)
    return RoverScenario(
        identifier=identifier,
        terrain=terrain,
        start=Position(*start),
        module=Position(*module),
        exit=Position(*exit_),
    )


_SCENARIOS = (
    _scenario(
        "training-0",
        (".....", ".~#..", "..~..", ".#..."),
        start=(0, 0),
        module=(2, 2),
        exit_=(3, 4),
    ),
    _scenario(
        "training-1",
        ("..#...", ".~..#.", "...~..", "#.....", "..#..."),
        start=(0, 0),
        module=(2, 4),
        exit_=(4, 5),
    ),
    _scenario(
        "training-2",
        ("......", ".##~..", "...~#.", ".~....", "...#.."),
        start=(4, 0),
        module=(0, 4),
        exit_=(4, 5),
    ),
)


def get_scenario(identifier: int) -> RoverScenario:
    if not 0 <= identifier < len(_SCENARIOS):
        raise ValueError(f"scenario identifier must be between 0 and {len(_SCENARIOS) - 1}")
    return _SCENARIOS[identifier]


def training_scenarios() -> tuple[RoverScenario, ...]:
    return _SCENARIOS


def generated_scenarios(
    seed: int,
    count: int,
    rows: int = 6,
    columns: int = 8,
) -> tuple[RoverScenario, ...]:
    """Build a reproducible suite with a guaranteed non-fragile mission route."""
    if count <= 0:
        raise ValueError("count must be positive")
    if rows < 4 or columns < 5:
        raise ValueError("generated scenarios require at least a 4x5 map")

    rng = random.Random(seed)
    scenarios: list[RoverScenario] = []
    seen: set[tuple[object, ...]] = set()
    attempts = 0

    while len(scenarios) < count:
        attempts += 1
        if attempts > count * 1000:
            raise RuntimeError("could not generate enough unique scenarios")

        locations = [Position(row, column) for row in range(rows) for column in range(columns)]
        start, module, exit_ = rng.sample(locations, 3)
        route = _random_shortest_path(rng, start, module)
        route.update(_random_shortest_path(rng, module, exit_))
        protected = _structural_scan_support(rng, route, rows, columns)
        unprotected = [position for position in locations if position not in protected]
        if not unprotected:
            continue

        terrain = [
            [
                rng.choices(
                    (Terrain.STABLE, Terrain.SLIPPERY, Terrain.FRAGILE),
                    weights=(55, 25, 20),
                )[0]
                for _ in range(columns)
            ]
            for _ in range(rows)
        ]
        for position in protected:
            terrain[position.row][position.column] = rng.choices(
                (Terrain.STABLE, Terrain.SLIPPERY), weights=(70, 30)
            )[0]
        terrain[start.row][start.column] = Terrain.STABLE

        if not any(
            terrain[position.row][position.column] is Terrain.FRAGILE
            for position in unprotected
        ):
            position = rng.choice(unprotected)
            terrain[position.row][position.column] = Terrain.FRAGILE
        if not any(
            terrain[position.row][position.column] is Terrain.SLIPPERY
            for position in locations
        ):
            position = rng.choice(tuple(protected - {start}))
            terrain[position.row][position.column] = Terrain.SLIPPERY

        terrain_tuple = tuple(tuple(row) for row in terrain)
        signature: tuple[object, ...] = (terrain_tuple, start, module, exit_)
        if signature in seen:
            continue
        seen.add(signature)
        index = len(scenarios)
        scenarios.append(
            RoverScenario(
                identifier=f"generated-{seed}-{index:03d}",
                terrain=terrain_tuple,
                start=start,
                module=module,
                exit=exit_,
            )
        )

    return tuple(scenarios)


def _random_shortest_path(
    rng: random.Random, start: Position, destination: Position
) -> set[Position]:
    path = {start}
    current = start
    while current != destination:
        candidates: list[Position] = []
        if current.row != destination.row:
            step = 1 if destination.row > current.row else -1
            candidates.append(Position(current.row + step, current.column))
        if current.column != destination.column:
            step = 1 if destination.column > current.column else -1
            candidates.append(Position(current.row, current.column + step))
        current = rng.choice(candidates)
        path.add(current)
    return path


def _structural_scan_support(
    rng: random.Random,
    route: set[Position],
    rows: int,
    columns: int,
) -> set[Position]:
    protected: set[Position] = set()
    for position in route:
        windows: list[tuple[Position, ...]] = []
        for start_column in range(
            max(0, position.column - 2),
            min(position.column, columns - 3) + 1,
        ):
            windows.append(
                tuple(
                    Position(position.row, start_column + offset)
                    for offset in range(3)
                )
            )
        for start_row in range(
            max(0, position.row - 2),
            min(position.row, rows - 3) + 1,
        ):
            windows.append(
                tuple(
                    Position(start_row + offset, position.column)
                    for offset in range(3)
                )
            )
        protected.update(rng.choice(windows))
    return protected
