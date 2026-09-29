"""Immutable Jetan board and AIMA-style adversarial-search interface."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum, IntEnum
from itertools import product
from typing import Iterator


BOARD_SIZE = 10
Location = tuple[int, int]
Step = tuple[int, int]
Path = tuple[Step, ...]

ORTHOGONAL: tuple[Step, ...] = ((0, 1), (1, 0), (0, -1), (-1, 0))
DIAGONAL: tuple[Step, ...] = ((1, 1), (1, -1), (-1, -1), (-1, 1))
KING: tuple[Step, ...] = ORTHOGONAL + DIAGONAL


class Player(IntEnum):
    ORANGE = 1
    BLACK = 2

    @property
    def opponent(self) -> "Player":
        return Player.BLACK if self is Player.ORANGE else Player.ORANGE


class PieceType(Enum):
    WARRIOR = "W"
    PADWAR = "A"
    DWAR = "D"
    FLIER = "F"
    CHIEF = "C"
    PRINCESS = "R"
    THOAT = "T"
    PANTHAN = "P"


@dataclass(frozen=True)
class Piece:
    player: Player
    kind: PieceType


@dataclass(frozen=True, order=True)
class Move:
    source: Location
    destination: Location

    def __str__(self) -> str:
        return f"{_location_name(self.source)}-{_location_name(self.destination)}"


def _walk_paths(steps: tuple[Step, ...], length: int) -> tuple[Path, ...]:
    """Generate fixed-length paths without immediately retracing a step."""
    paths = []
    for path in product(steps, repeat=length):
        if any(a == (-b[0], -b[1]) for a, b in zip(path, path[1:])):
            continue
        if sum(step[0] for step in path) == sum(step[1] for step in path) == 0:
            continue
        paths.append(path)
    return tuple(paths)


WARRIOR_PATHS = _walk_paths(ORTHOGONAL, 2)
PADWAR_PATHS = _walk_paths(DIAGONAL, 2)
DWAR_PATHS = _walk_paths(ORTHOGONAL, 3)
CHIEF_PATHS = _walk_paths(KING, 3)
FLIER_PATHS: tuple[Path, ...] = tuple(
    ((dx, dy),) for dx in (-3, -1, 1, 3) for dy in (-3, -1, 1, 3)
)
THOAT_PATHS: tuple[Path, ...] = tuple(
    ((dx, dy),)
    for dx, dy in (
        (-1, 2),
        (1, 2),
        (-2, 1),
        (0, 1),
        (2, 1),
        (-1, 0),
        (1, 0),
        (-2, -1),
        (0, -1),
        (2, -1),
        (-1, -2),
        (1, -2),
    )
)
PRINCESS_PATHS: tuple[Path, ...] = tuple(
    ((dx, dy),)
    for dx in range(-3, 4)
    for dy in range(-3, 4)
    if (dx, dy) != (0, 0)
)
ORANGE_PANTHAN_PATHS: tuple[Path, ...] = tuple(
    ((dx, dy),) for dx, dy in ((-1, 1), (0, 1), (1, 1), (-1, 0), (1, 0))
)
BLACK_PANTHAN_PATHS: tuple[Path, ...] = tuple(
    ((dx, dy),) for dx, dy in ((-1, -1), (0, -1), (1, -1), (-1, 0), (1, 0))
)


@dataclass(frozen=True)
class JetanBoard:
    """A complete immutable game state and AIMA-style game problem.

    Agents may call ``player``, ``actions``, ``result``, ``is_terminal``, and
    ``utility`` directly on any board instance while exploring a game tree.
    """

    cells: tuple[Piece | None, ...]
    turn: Player = Player.ORANGE
    winner: Player | None = None
    draw: bool = False
    used_escape: tuple[bool, bool] = (False, False)
    ply: int = 0
    # These memoized derived values do not participate in board identity.
    _attacked_cache: dict[Player, frozenset[Location]] = field(
        default_factory=dict, init=False, repr=False, compare=False, hash=False
    )
    _legal_actions_cache: dict[Player, tuple[Move, ...]] = field(
        default_factory=dict, init=False, repr=False, compare=False, hash=False
    )
    _has_legal_action_cache: dict[Player, bool] = field(
        default_factory=dict, init=False, repr=False, compare=False, hash=False
    )
    _legal_action_count_cache: dict[Player, int] = field(
        default_factory=dict, init=False, repr=False, compare=False, hash=False
    )

    def __post_init__(self) -> None:
        if len(self.cells) != BOARD_SIZE * BOARD_SIZE:
            raise ValueError("a Jetan board must contain exactly 100 cells")
        if not isinstance(self.turn, Player):
            raise TypeError("turn must be a Player")
        if len(self.used_escape) != 2 or not all(
            isinstance(value, bool) for value in self.used_escape
        ):
            raise ValueError("used_escape must contain two boolean values")
        if self.ply < 0:
            raise ValueError("ply cannot be negative")
        if self.winner is not None and self.draw:
            raise ValueError("a board cannot have both a winner and a draw")

    @classmethod
    def initial(cls) -> "JetanBoard":
        cells: list[Piece | None] = [None] * (BOARD_SIZE * BOARD_SIZE)
        back_rank = (
            PieceType.WARRIOR,
            PieceType.PADWAR,
            PieceType.DWAR,
            PieceType.FLIER,
            PieceType.CHIEF,
            PieceType.PRINCESS,
            PieceType.FLIER,
            PieceType.DWAR,
            PieceType.PADWAR,
            PieceType.WARRIOR,
        )
        front_rank = (PieceType.THOAT,) + (PieceType.PANTHAN,) * 8 + (
            PieceType.THOAT,
        )
        for player, back_y, front_y in (
            (Player.ORANGE, 0, 1),
            (Player.BLACK, 9, 8),
        ):
            for x, kind in enumerate(back_rank):
                cells[_index((x, back_y))] = Piece(player, kind)
            for x, kind in enumerate(front_rank):
                cells[_index((x, front_y))] = Piece(player, kind)
        return cls(tuple(cells))

    @classmethod
    def from_pieces(
        cls,
        pieces: dict[Location, Piece],
        turn: Player = Player.ORANGE,
        used_escape: tuple[bool, bool] = (False, False),
    ) -> "JetanBoard":
        """Construct a position for examples and tests."""
        cells: list[Piece | None] = [None] * (BOARD_SIZE * BOARD_SIZE)
        for location, piece in pieces.items():
            if not _on_board(location):
                raise ValueError(f"location is outside the board: {location}")
            if cells[_index(location)] is not None:
                raise ValueError(f"multiple pieces at {location}")
            cells[_index(location)] = piece
        return cls(tuple(cells), turn=turn, used_escape=used_escape)

    def at(self, location: Location) -> Piece | None:
        if not _on_board(location):
            raise IndexError(f"location is outside the board: {location}")
        return self.cells[_index(location)]

    def pieces(self, player: Player | None = None) -> Iterator[tuple[Location, Piece]]:
        for index, piece in enumerate(self.cells):
            if piece is not None and (player is None or piece.player is player):
                yield (index % BOARD_SIZE, index // BOARD_SIZE), piece

    # AIMA game interface -------------------------------------------------
    def player(self, state: "JetanBoard | None" = None) -> Player:
        board = self if state is None else state
        return board.turn

    def actions(self, state: "JetanBoard | None" = None) -> tuple[Move, ...]:
        board = self if state is None else state
        if board.winner is not None or board.draw:
            return ()
        return board._legal_actions(board.turn)

    def legal_actions(self, player: Player) -> tuple[Move, ...]:
        """Return legal moves for either player without changing the turn."""
        if self.winner is not None or self.draw:
            return ()
        return self._legal_actions(player)

    def legal_action_count(self, player: Player) -> int:
        """Count legal moves without sorting or retaining the complete tuple."""
        if self.winner is not None or self.draw:
            return 0
        if player in self._legal_actions_cache:
            return len(self._legal_actions_cache[player])
        if player not in self._legal_action_count_cache:
            self._legal_action_count_cache[player] = sum(
                1 for _ in self._iter_legal_action_locations(player)
            )
            self._has_legal_action_cache[player] = bool(
                self._legal_action_count_cache[player]
            )
        return self._legal_action_count_cache[player]

    def result(
        self, state_or_action: "JetanBoard | Move", action: Move | None = None
    ) -> "JetanBoard":
        """Return a successor using ``result(action)`` or ``result(state, action)``."""
        if action is None:
            board = self
            move = state_or_action
        else:
            if not isinstance(state_or_action, JetanBoard):
                raise TypeError("the first argument must be a JetanBoard")
            board = state_or_action
            move = action
        if not isinstance(move, Move):
            raise TypeError("action must be a Move")
        if move not in board.actions():
            raise ValueError(f"illegal move: {move}")

        moving_piece = board.at(move.source)
        captured_piece = board.at(move.destination)
        assert moving_piece is not None
        cells = list(board.cells)
        cells[_index(move.source)] = None
        cells[_index(move.destination)] = moving_piece
        winner = None
        if captured_piece is not None:
            if captured_piece.kind is PieceType.PRINCESS:
                winner = moving_piece.player
            elif (
                captured_piece.kind is PieceType.CHIEF
                and moving_piece.kind is PieceType.CHIEF
            ):
                winner = moving_piece.player

        escapes = list(board.used_escape)
        dx = abs(move.destination[0] - move.source[0])
        dy = abs(move.destination[1] - move.source[1])
        if moving_piece.kind is PieceType.PRINCESS and max(dx, dy) > 3:
            escapes[moving_piece.player - 1] = True
        successor = JetanBoard(
            tuple(cells),
            turn=board.turn if winner else board.turn.opponent,
            winner=winner,
            used_escape=tuple(escapes),  # type: ignore[arg-type]
            ply=board.ply + 1,
        )
        if winner is None and not successor._has_legal_action(successor.turn):
            successor = replace(successor, winner=moving_piece.player)
        return successor

    def is_terminal(self, state: "JetanBoard | None" = None) -> bool:
        board = self if state is None else state
        return (
            board.winner is not None
            or board.draw
            or not board._has_legal_action(board.turn)
        )

    def terminal_test(self, state: "JetanBoard | None" = None) -> bool:
        """AIMA-compatible alias for :meth:`is_terminal`."""
        return self.is_terminal(state)

    def utility(
        self,
        state_or_player: "JetanBoard | Player",
        player: Player | None = None,
    ) -> int:
        """Return terminal utility from ``player``'s fixed perspective."""
        if player is None:
            board = self
            perspective = state_or_player
        else:
            if not isinstance(state_or_player, JetanBoard):
                raise TypeError("the first argument must be a JetanBoard")
            board = state_or_player
            perspective = player
        if not isinstance(perspective, Player):
            raise TypeError("player must be a Player")
        if not board.is_terminal():
            raise ValueError("utility is defined only for terminal states")
        if board.draw:
            return 0
        winner = board.winner or board.turn.opponent
        return 1 if winner is perspective else -1

    def as_draw(self) -> "JetanBoard":
        return replace(self, winner=None, draw=True)

    def resolve_terminal(self) -> "JetanBoard":
        """Record the opponent's win when the current player cannot move."""
        if (
            self.winner is None
            and not self.draw
            and not self._has_legal_action(self.turn)
        ):
            return replace(self, winner=self.turn.opponent)
        return self

    # Rule implementation ------------------------------------------------
    def attacked_locations(self, player: Player) -> frozenset[Location]:
        if player in self._attacked_cache:
            return self._attacked_cache[player]
        attacked: set[Location] = set()
        for source, piece in self.pieces(player):
            if piece.kind is PieceType.PRINCESS:
                continue
            for path in self._paths(piece):
                destination = self._path_destination(source, path)
                if destination is not None:
                    attacked.add(destination)
        result = frozenset(attacked)
        self._attacked_cache[player] = result
        return result

    def _legal_actions(self, player: Player) -> tuple[Move, ...]:
        if player in self._legal_actions_cache:
            return self._legal_actions_cache[player]
        if self._has_legal_action_cache.get(player) is False:
            return ()
        result = tuple(sorted(self._iter_legal_actions(player)))
        self._legal_actions_cache[player] = result
        self._has_legal_action_cache[player] = bool(result)
        self._legal_action_count_cache[player] = len(result)
        return result

    def _has_legal_action(self, player: Player) -> bool:
        if player in self._legal_actions_cache:
            return bool(self._legal_actions_cache[player])
        if player not in self._has_legal_action_cache:
            self._has_legal_action_cache[player] = next(
                self._iter_legal_action_locations(player), None
            ) is not None
        return self._has_legal_action_cache[player]

    def _iter_legal_actions(self, player: Player) -> Iterator[Move]:
        for source, destination in self._iter_legal_action_locations(player):
            yield Move(source, destination)

    def _iter_legal_action_locations(
        self, player: Player
    ) -> Iterator[tuple[Location, Location]]:
        attacked = self.attacked_locations(player.opponent)
        for source, piece in self.pieces(player):
            destinations_seen: set[Location] = set()
            paths = self._paths(piece)
            if (
                piece.kind is PieceType.PRINCESS
                and source in attacked
                and not self.used_escape[player - 1]
            ):
                paths = tuple(
                    ((x - source[0], y - source[1]),)
                    for y in range(BOARD_SIZE)
                    for x in range(BOARD_SIZE)
                    if (x, y) != source
                )
            for path in paths:
                destination = self._path_destination(source, path)
                if destination is None:
                    continue
                occupant = self.cells[_index(destination)]
                if occupant is not None and occupant.player is player:
                    continue
                if piece.kind is PieceType.PRINCESS:
                    if occupant is not None or destination in attacked:
                        continue
                if destination not in destinations_seen:
                    destinations_seen.add(destination)
                    yield source, destination

    def _path_destination(self, source: Location, path: Path) -> Location | None:
        x, y = source
        cells = self.cells
        final_index = len(path) - 1
        for index, (dx, dy) in enumerate(path):
            x += dx
            y += dy
            if x < 0 or x >= BOARD_SIZE or y < 0 or y >= BOARD_SIZE:
                return None
            if index < final_index and cells[y * BOARD_SIZE + x] is not None:
                return None
        return x, y

    @staticmethod
    def _paths(piece: Piece) -> tuple[Path, ...]:
        if piece.kind is PieceType.WARRIOR:
            return WARRIOR_PATHS
        if piece.kind is PieceType.PADWAR:
            return PADWAR_PATHS
        if piece.kind is PieceType.DWAR:
            return DWAR_PATHS
        if piece.kind is PieceType.FLIER:
            return FLIER_PATHS
        if piece.kind is PieceType.CHIEF:
            return CHIEF_PATHS
        if piece.kind is PieceType.PRINCESS:
            return PRINCESS_PATHS
        if piece.kind is PieceType.THOAT:
            return THOAT_PATHS
        if piece.kind is PieceType.PANTHAN:
            return (
                ORANGE_PANTHAN_PATHS
                if piece.player is Player.ORANGE
                else BLACK_PANTHAN_PATHS
            )
        raise NotImplementedError(f"movement is not defined for {piece.kind}")

    def __str__(self) -> str:
        rows = ["    " + " ".join(chr(ord("a") + x) for x in range(BOARD_SIZE))]
        for y in reversed(range(BOARD_SIZE)):
            symbols = []
            for x in range(BOARD_SIZE):
                piece = self.at((x, y))
                if piece is None:
                    symbols.append(".")
                elif piece.player is Player.ORANGE:
                    symbols.append(piece.kind.value)
                else:
                    symbols.append(piece.kind.value.lower())
            rows.append(f"{y:2}  " + " ".join(symbols))
        return "\n".join(rows)


def _index(location: Location) -> int:
    x, y = location
    return y * BOARD_SIZE + x


def _on_board(location: Location) -> bool:
    x, y = location
    return 0 <= x < BOARD_SIZE and 0 <= y < BOARD_SIZE


def _location_name(location: Location) -> str:
    return f"{chr(ord('a') + location[0])}{location[1]}"
