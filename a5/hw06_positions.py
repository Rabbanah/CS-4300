"""Six fixed constructed positions using the existing Assignment 5 engine."""

from dataclasses import dataclass, replace

from jetan import JetanBoard, Piece, PieceType, Player


@dataclass(frozen=True)
class Position:
    identifier: str
    category: str
    board: JetanBoard
    construction: str


def _after_indices(indices: tuple[int, ...]) -> JetanBoard:
    board = JetanBoard.initial()
    for index in indices:
        board = board.result(board.actions()[index])
    return board


def _snapshot(
    pieces: dict[tuple[int, int], Piece], turn: Player, ply: int
) -> JetanBoard:
    return replace(JetanBoard.from_pieces(pieces, turn=turn), ply=ply)


POSITIONS = (
    Position("initial", "initial", JetanBoard.initial(), "constructed with JetanBoard.initial()"),
    Position("early-a", "early game", _after_indices((0, 0)), "constructed with canonical indices (0, 0)"),
    Position("early-b", "early game", _after_indices((8, 11, 20, 7)), "constructed with canonical indices (8, 11, 20, 7)"),
    Position(
        "middle-capture",
        "middle game with immediate capture",
        _snapshot(
            {
                (0, 0): Piece(Player.ORANGE, PieceType.PRINCESS),
                (4, 4): Piece(Player.ORANGE, PieceType.PANTHAN),
                (2, 3): Piece(Player.ORANGE, PieceType.FLIER),
                (5, 5): Piece(Player.BLACK, PieceType.PRINCESS),
                (8, 7): Piece(Player.BLACK, PieceType.CHIEF),
                (6, 6): Piece(Player.BLACK, PieceType.THOAT),
            },
            Player.ORANGE,
            24,
        ),
        "constructed legal six-piece state at absolute ply 24",
    ),
    Position(
        "middle-threat",
        "middle game with immediate threat",
        _snapshot(
            {
                (1, 1): Piece(Player.ORANGE, PieceType.PRINCESS),
                (3, 3): Piece(Player.ORANGE, PieceType.CHIEF),
                (6, 4): Piece(Player.ORANGE, PieceType.DWAR),
                (8, 8): Piece(Player.BLACK, PieceType.PRINCESS),
                (4, 4): Piece(Player.BLACK, PieceType.CHIEF),
                (5, 3): Piece(Player.BLACK, PieceType.PANTHAN),
                (7, 6): Piece(Player.BLACK, PieceType.FLIER),
            },
            Player.BLACK,
            35,
        ),
        "constructed legal seven-piece state at absolute ply 35",
    ),
    Position(
        "reduced",
        "reduced material",
        _snapshot(
            {
                (0, 0): Piece(Player.ORANGE, PieceType.PRINCESS),
                (2, 2): Piece(Player.ORANGE, PieceType.THOAT),
                (9, 9): Piece(Player.BLACK, PieceType.PRINCESS),
                (7, 7): Piece(Player.BLACK, PieceType.DWAR),
            },
            Player.ORANGE,
            80,
        ),
        "constructed legal four-piece state at absolute ply 80",
    ),
)


def get_position(identifier: str) -> JetanBoard:
    for position in POSITIONS:
        if position.identifier == identifier:
            return position.board
    raise KeyError(identifier)
