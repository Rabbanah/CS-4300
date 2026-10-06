"""Stable action-ordering wrappers and student ranking-key stubs."""

from collections.abc import Callable
from typing import Any

from jetan import JetanBoard, Move, Player, PieceType

OrderingFunction = Callable[
    [JetanBoard, tuple[Move, ...], Player], tuple[Move, ...]
]


PIECE_VALUES: dict[PieceType, int] = {
    PieceType.PRINCESS: 1000,
    PieceType.CHIEF: 100,
    PieceType.THOAT: 15,
    PieceType.FLIER: 12,
    PieceType.DWAR: 10,
    PieceType.PADWAR: 8,
    PieceType.WARRIOR: 5,
    PieceType.PANTHAN: 3,
}


def ordering_key_1(
    board: JetanBoard, action: Move, perspective: Player
) -> Any:
    """Rank moves primarily by captured piece value (Most Valuable Victim)."""
    target_piece = board.at(action.destination)
    score = 0
    if target_piece is not None:
        val = PIECE_VALUES.get(target_piece.kind, 1)
        score = val if target_piece.player is not board.player() else -val

    return score if board.player() is perspective else -score


def ordering_key_2(
    board: JetanBoard, action: Move, perspective: Player
) -> Any:
    """Combine capture value with destination centrality."""
    target_piece = board.at(action.destination)
    score = 0.0
    if target_piece is not None:
        val = PIECE_VALUES.get(target_piece.kind, 1)
        score += 50.0 * (val if target_piece.player is not board.player() else -val)

    # Centrality heuristic: closer to (4.5, 4.5) gets a higher bonus
    r, c = action.destination
    dist_center = abs(r - 4.5) + abs(c - 4.5)
    score += (9.0 - dist_center)

    return score if board.player() is perspective else -score

def _stable_order(
    board: JetanBoard,
    actions: tuple[Move, ...],
    perspective: Player,
    key: Callable[[JetanBoard, Move, Player], Any],
) -> tuple[Move, ...]:
    """Rank the supplied canonical tuple while retaining input-order ties."""
    return tuple(
        sorted(
            actions,
            key=lambda action: key(board, action, perspective),
            reverse=board.player() is perspective,
        )
    )


def order_actions_1(
    board: JetanBoard, actions: tuple[Move, ...], perspective: Player
) -> tuple[Move, ...]:
    """Apply the first student key to the supplied canonical action tuple."""
    return _stable_order(board, actions, perspective, ordering_key_1)


def order_actions_2(
    board: JetanBoard, actions: tuple[Move, ...], perspective: Player
) -> tuple[Move, ...]:
    """Apply the second student key to the supplied canonical action tuple."""
    return _stable_order(board, actions, perspective, ordering_key_2)
