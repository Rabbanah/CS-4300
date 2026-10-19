"""Provided Rescue Rover environment for U0-HW-07.

Students should not modify this file.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Terrain(Enum):
    STABLE = "stable"
    SLIPPERY = "slippery"
    FRAGILE = "fragile"


class Scanner(Enum):
    STRUCTURAL = "structural"
    SURFACE = "surface"


class ActionKind(Enum):
    MOVE = "move"
    ENABLE_TRACTION = "enable_traction"
    DISABLE_TRACTION = "disable_traction"
    STRUCTURAL_SCAN = "structural_scan"
    SURFACE_SCAN = "surface_scan"
    RETRIEVE = "retrieve"
    STOP = "stop"


class MissionStatus(Enum):
    RUNNING = "running"
    SUCCESS = "success"
    STOPPED = "stopped"
    DESTROYED = "destroyed"
    TIMEOUT = "timeout"


class Event(Enum):
    MOVED = "moved"
    TRACTION_ENABLED = "traction_enabled"
    TRACTION_DISABLED = "traction_disabled"
    SCANNED = "scanned"
    RETRIEVED = "retrieved"
    MISSION_COMPLETED = "mission_completed"
    STOPPED_EARLY = "stopped_early"
    FRAGILE_COLLAPSE = "fragile_collapse"
    SLIPPED = "slipped"
    INVALID_ACTION = "invalid_action"
    TIME_LIMIT = "time_limit"


@dataclass(frozen=True, order=True)
class Position:
    row: int
    column: int


@dataclass(frozen=True)
class Action:
    kind: ActionKind
    target: Position | None = None
    region: tuple[Position, ...] = ()

    @classmethod
    def move(cls, target: Position) -> Action:
        return cls(ActionKind.MOVE, target=target)

    @classmethod
    def structural_scan(cls, *region: Position) -> Action:
        return cls(ActionKind.STRUCTURAL_SCAN, region=tuple(region))

    @classmethod
    def surface_scan(cls, *region: Position) -> Action:
        return cls(ActionKind.SURFACE_SCAN, region=tuple(region))

    @classmethod
    def enable_traction(cls) -> Action:
        return cls(ActionKind.ENABLE_TRACTION)

    @classmethod
    def disable_traction(cls) -> Action:
        return cls(ActionKind.DISABLE_TRACTION)

    @classmethod
    def retrieve(cls) -> Action:
        return cls(ActionKind.RETRIEVE)

    @classmethod
    def stop(cls) -> Action:
        return cls(ActionKind.STOP)


@dataclass(frozen=True)
class ScanObservation:
    scanner: Scanner
    region: tuple[Position, ...]
    positive: bool


@dataclass(frozen=True)
class ActionOutcome:
    succeeded: bool
    event: Event
    score_delta: int
    scan: ScanObservation | None = None


@dataclass(frozen=True)
class Percept:
    rows: int
    columns: int
    position: Position
    module_position: Position
    exit_position: Position
    traction_enabled: bool
    carrying_module: bool
    action_count: int
    last_outcome: ActionOutcome | None


@dataclass(frozen=True)
class Transition:
    action: Action
    percept: Percept
    outcome: ActionOutcome
    status: MissionStatus
    total_score: int


@dataclass(frozen=True)
class ScoreRules:
    move: int = -3
    stable_traction_surcharge: int = -3
    structural_scan: int = -2
    surface_scan: int = -2
    traction_toggle: int = -1
    retrieve: int = -1
    stop: int = 0
    invalid_action: int = -5
    fragile_collapse: int = -1000
    slippery_without_traction: int = -100
    stopped_early: int = -100
    timeout: int = -100
    mission_success: int = 10000


SCORE_RULES = ScoreRules()


@dataclass(frozen=True)
class RoverScenario:
    identifier: str
    terrain: tuple[tuple[Terrain, ...], ...]
    start: Position
    module: Position
    exit: Position

    def __post_init__(self) -> None:
        if not self.terrain or not self.terrain[0]:
            raise ValueError("terrain must contain at least one tile")
        columns = len(self.terrain[0])
        if any(len(row) != columns for row in self.terrain):
            raise ValueError("terrain must be rectangular")
        positions = (self.start, self.module, self.exit)
        if any(not self.contains(position) for position in positions):
            raise ValueError("start, module, and exit must be mapped positions")
        if self.tile(self.start) is not Terrain.STABLE:
            raise ValueError("the start tile must be stable")
        if self.tile(self.exit) is Terrain.FRAGILE:
            raise ValueError("the exit tile cannot be fragile")
        if self.tile(self.module) is Terrain.FRAGILE:
            raise ValueError("the module tile cannot be fragile")

    @property
    def rows(self) -> int:
        return len(self.terrain)

    @property
    def columns(self) -> int:
        return len(self.terrain[0])

    def contains(self, position: Position) -> bool:
        return 0 <= position.row < self.rows and 0 <= position.column < self.columns

    def tile(self, position: Position) -> Terrain:
        return self.terrain[position.row][position.column]


class RescueRoverEnvironment:
    """Deterministic, partially observable Rescue Rover simulator."""

    def __init__(
        self,
        scenario: RoverScenario,
        max_actions: int = 100,
        score_rules: ScoreRules = SCORE_RULES,
    ) -> None:
        if max_actions <= 0:
            raise ValueError("max_actions must be positive")
        self._scenario = scenario
        self.max_actions = max_actions
        self.score_rules = score_rules
        self.reset()

    def reset(self) -> Percept:
        self.position = self._scenario.start
        self.traction_enabled = False
        self.carrying_module = False
        self.action_count = 0
        self.move_count = 0
        self.scan_count = 0
        self.traction_moves = 0
        self.traction_toggles = 0
        self.unprotected_slips = 0
        self.invalid_actions = 0
        self.total_score = 0
        self.status = MissionStatus.RUNNING
        self.trace: list[Transition] = []
        self._last_outcome: ActionOutcome | None = None
        return self.percept()

    @property
    def done(self) -> bool:
        return self.status is not MissionStatus.RUNNING

    @property
    def successful(self) -> bool:
        return self.status is MissionStatus.SUCCESS

    def percept(self) -> Percept:
        return Percept(
            rows=self._scenario.rows,
            columns=self._scenario.columns,
            position=self.position,
            module_position=self._scenario.module,
            exit_position=self._scenario.exit,
            traction_enabled=self.traction_enabled,
            carrying_module=self.carrying_module,
            action_count=self.action_count,
            last_outcome=self._last_outcome,
        )

    def step(self, action: Action) -> Transition:
        if self.done:
            raise RuntimeError("Cannot act after an episode has ended")
        if not isinstance(action, Action):
            raise TypeError("action must be an Action value")

        outcome = self._apply(action)
        self.action_count += 1
        self.total_score += outcome.score_delta
        self._last_outcome = outcome

        if self.status is MissionStatus.RUNNING and self.action_count >= self.max_actions:
            self.status = MissionStatus.TIMEOUT
            timeout_delta = self.score_rules.timeout
            self.total_score += timeout_delta
            outcome = ActionOutcome(
                outcome.succeeded,
                Event.TIME_LIMIT,
                outcome.score_delta + timeout_delta,
                outcome.scan,
            )
            self._last_outcome = outcome

        transition = Transition(
            action=action,
            percept=self.percept(),
            outcome=outcome,
            status=self.status,
            total_score=self.total_score,
        )
        self.trace.append(transition)
        return transition

    def _apply(self, action: Action) -> ActionOutcome:
        if action.kind is ActionKind.MOVE:
            return self._move(action)
        if action.kind is ActionKind.ENABLE_TRACTION:
            if action.target is not None or action.region or self.traction_enabled:
                return self._invalid()
            self.traction_enabled = True
            self.traction_toggles += 1
            return ActionOutcome(True, Event.TRACTION_ENABLED, self.score_rules.traction_toggle)
        if action.kind is ActionKind.DISABLE_TRACTION:
            if action.target is not None or action.region or not self.traction_enabled:
                return self._invalid()
            self.traction_enabled = False
            self.traction_toggles += 1
            return ActionOutcome(True, Event.TRACTION_DISABLED, self.score_rules.traction_toggle)
        if action.kind is ActionKind.STRUCTURAL_SCAN:
            return self._scan(action, Scanner.STRUCTURAL)
        if action.kind is ActionKind.SURFACE_SCAN:
            return self._scan(action, Scanner.SURFACE)
        if action.kind is ActionKind.RETRIEVE:
            if action.target is not None or action.region:
                return self._invalid()
            if self.position != self._scenario.module or self.carrying_module:
                return self._invalid()
            self.carrying_module = True
            return ActionOutcome(True, Event.RETRIEVED, self.score_rules.retrieve)
        if action.kind is ActionKind.STOP:
            if action.target is not None or action.region:
                return self._invalid()
            if self.position == self._scenario.exit and self.carrying_module:
                self.status = MissionStatus.SUCCESS
                return ActionOutcome(
                    True,
                    Event.MISSION_COMPLETED,
                    self.score_rules.stop + self.score_rules.mission_success,
                )
            self.status = MissionStatus.STOPPED
            return ActionOutcome(
                True,
                Event.STOPPED_EARLY,
                self.score_rules.stop + self.score_rules.stopped_early,
            )
        return self._invalid()

    def _move(self, action: Action) -> ActionOutcome:
        target = action.target
        if action.region or target is None or not self._scenario.contains(target):
            return self._invalid()
        distance = abs(target.row - self.position.row) + abs(
            target.column - self.position.column
        )
        if distance != 1:
            return self._invalid()

        terrain = self._scenario.tile(target)
        score = self.score_rules.move
        self.move_count += 1
        if self.traction_enabled:
            if terrain is Terrain.STABLE:
                score += self.score_rules.stable_traction_surcharge
            self.traction_moves += 1

        self.position = target
        if terrain is Terrain.FRAGILE:
            self.status = MissionStatus.DESTROYED
            return ActionOutcome(
                False,
                Event.FRAGILE_COLLAPSE,
                score + self.score_rules.fragile_collapse,
            )
        if terrain is Terrain.SLIPPERY and not self.traction_enabled:
            self.unprotected_slips += 1
            return ActionOutcome(
                True,
                Event.SLIPPED,
                score + self.score_rules.slippery_without_traction,
            )
        return ActionOutcome(True, Event.MOVED, score)

    def _scan(self, action: Action, scanner: Scanner) -> ActionOutcome:
        if action.target is not None or not self._valid_region(action.region, scanner):
            return self._invalid()
        region = tuple(sorted(action.region))
        terrain = (self._scenario.tile(position) for position in region)
        sought = Terrain.FRAGILE if scanner is Scanner.STRUCTURAL else Terrain.SLIPPERY
        observation = ScanObservation(
            scanner=scanner,
            region=region,
            positive=any(tile is sought for tile in terrain),
        )
        self.scan_count += 1
        score = (
            self.score_rules.structural_scan
            if scanner is Scanner.STRUCTURAL
            else self.score_rules.surface_scan
        )
        return ActionOutcome(True, Event.SCANNED, score, observation)

    def _valid_region(
        self, region: tuple[Position, ...], scanner: Scanner
    ) -> bool:
        expected_size = 3 if scanner is Scanner.STRUCTURAL else 4
        if len(region) != expected_size or len(set(region)) != expected_size:
            return False
        if any(not self._scenario.contains(position) for position in region):
            return False

        cells = set(region)
        rows = sorted({position.row for position in cells})
        columns = sorted({position.column for position in cells})
        if scanner is Scanner.STRUCTURAL:
            horizontal = (
                len(rows) == 1
                and len(columns) == 3
                and columns[-1] - columns[0] == 2
            )
            vertical = (
                len(columns) == 1
                and len(rows) == 3
                and rows[-1] - rows[0] == 2
            )
            return horizontal or vertical
        return (
            len(rows) == 2
            and rows[1] - rows[0] == 1
            and len(columns) == 2
            and columns[1] - columns[0] == 1
        )

    def _invalid(self) -> ActionOutcome:
        self.invalid_actions += 1
        return ActionOutcome(False, Event.INVALID_ACTION, self.score_rules.invalid_action)
