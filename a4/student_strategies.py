"""Implement student evaluation, prompting, parsing, and fallback functions."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence

from jetan import JetanBoard, Move, PieceType, Player

PIECE_VALUES: dict[PieceType, float] = {
    PieceType.PANTHAN: 1.0,
    PieceType.WARRIOR: 3.0,
    PieceType.PADWAR: 3.0,
    PieceType.DWAR: 3.0,
    PieceType.THOAT: 4.0,
    PieceType.FLIER: 4.0,
    PieceType.CHIEF: 7.0,
    PieceType.PRINCESS: 10.0,
}


def _material_score(board: JetanBoard, player: Player) -> float:
    return sum(PIECE_VALUES[piece.kind] for _, piece in board.pieces(player))



def evaluate_position_1(board: JetanBoard, perspective: Player) -> float:
    """Evaluation 1: Pure normalized material balance."""
    my_mat = _material_score(board, perspective)
    opp_mat = _material_score(board, perspective.opponent)
    diff = my_mat - opp_mat
    score = math.tanh(diff / 12.0) * 0.95
    return max(-0.99, min(0.99, round(score, 4)))


def evaluate_position_2(board: JetanBoard, perspective: Player) -> float:
    """Evaluation 2: Material balance + board control + Princess safety."""
    opponent = perspective.opponent
    mat_diff = _material_score(board, perspective) - _material_score(board, opponent)

    pos_score = 0.0
    for (x, y), piece in board.pieces():
        sign = 1.0 if piece.player is perspective else -1.0
        if 3 <= x <= 6 and 3 <= y <= 6:
            pos_score += sign * 0.25
        rank = y if piece.player is Player.ORANGE else (9 - y)
        if piece.kind is PieceType.PANTHAN:
            pos_score += sign * (rank * 0.05)

    opp_attacks = board.attacked_locations(opponent)
    my_attacks = board.attacked_locations(perspective)
    princess_penalty = 0.0

    for loc, piece in board.pieces():
        if piece.kind is PieceType.PRINCESS:
            if piece.player is perspective and loc in opp_attacks:
                princess_penalty -= 1.5
            elif piece.player is opponent and loc in my_attacks:
                princess_penalty += 1.5

    combined = (mat_diff * 1.0) + pos_score + princess_penalty
    score = math.tanh(combined / 14.0) * 0.95
    return max(-0.99, min(0.99, round(score, 4)))


def evaluate_position_3(board: JetanBoard, perspective: Player) -> float:
    """Evaluation 3: Material + Position + Mobility + Escape availability."""
    opponent = perspective.opponent
    mat_diff = _material_score(board, perspective) - _material_score(board, opponent)

    pos_score = 0.0
    for (x, y), piece in board.pieces():
        sign = 1.0 if piece.player is perspective else -1.0
        if 3 <= x <= 6 and 3 <= y <= 6:
            pos_score += sign * 0.25
        rank = y if piece.player is Player.ORANGE else (9 - y)
        if piece.kind is PieceType.PANTHAN:
            pos_score += sign * (rank * 0.05)

    my_mobility = board.legal_action_count(perspective)
    opp_mobility = board.legal_action_count(opponent)
    mobility_diff = (my_mobility - opp_mobility) * 0.05

    opp_attacks = board.attacked_locations(opponent)
    my_attacks = board.attacked_locations(perspective)
    threat_score = 0.0
    for loc, piece in board.pieces():
        if piece.kind is PieceType.PRINCESS:
            if piece.player is perspective and loc in opp_attacks:
                threat_score -= 2.0
            elif piece.player is opponent and loc in my_attacks:
                threat_score += 2.0
        elif piece.kind is PieceType.CHIEF:
            if piece.player is perspective and loc in opp_attacks:
                threat_score -= 1.0
            elif piece.player is opponent and loc in my_attacks:
                threat_score += 1.0

    escape_score = 0.0
    if not board.used_escape[perspective - 1]:
        escape_score += 0.5
    if not board.used_escape[opponent - 1]:
        escape_score -= 0.5

    total = (
        (mat_diff * 1.0)
        + pos_score
        + mobility_diff
        + threat_score
        + escape_score
    )
    score = math.tanh(total / 16.0) * 0.95
    return max(-0.99, min(0.99, round(score, 4)))


def _no_duplicate_keys_decoder(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate key: {key}")
        result[key] = value
    return result



def build_evaluation_prompt(
    board: JetanBoard, perspective: Player
) -> Sequence[dict[str, str]]:
    """Build evaluation prompt instructing model to return {"score": NUMBER}."""
    lines = [
        f"You are an expert game evaluator for Jetan. Evaluate the board from the perspective of Player {perspective.name}.",
        "Return a score in [-0.99, 0.99] where +0.99 is winning, 0.0 is equal, and -0.99 is losing.",
        "Output ONLY a single raw JSON object on one line with no markdown fences, no formatting, and no commentary.",
        'Exact schema: {"score": 0.0}',
        "",
        f"Turn: {board.turn.name}",
        f"Perspective: {perspective.name}",
        f"Orange used escape: {board.used_escape[0]}, Black used escape: {board.used_escape[1]}",
        f"Board Representation:\n{str(board)}",
    ]
    return [
        {
            "role": "system",
            "content": (
                "You are a strict game position evaluator. "
                "Output ONLY a single raw JSON object: {\"score\": NUMBER}. "
                "No markdown fences, no explanations."
            ),
        },
        {"role": "user", "content": "\n".join(lines)},
    ]


def parse_evaluation_response(response: str) -> float:
    """Strictly parse JSON response for score in [-0.99, 0.99]."""
    text = response.strip()
    if not text:
        raise ValueError("Empty response")
    if "\n" in text or "```" in text:
        raise ValueError("Response contains multiple lines or markdown fences")

    parsed = json.loads(text, object_pairs_hook=_no_duplicate_keys_decoder)
    if not isinstance(parsed, dict) or set(parsed.keys()) != {"score"}:
        raise ValueError("Schema mismatch: must contain exactly 'score'")

    score_val = parsed["score"]
    if isinstance(score_val, bool) or not isinstance(score_val, (int, float)):
        raise ValueError("Score must be a numeric value")

    score_float = float(score_val)
    if not math.isfinite(score_float):
        raise ValueError("Score must be finite")
    if not (-0.99 <= score_float <= 0.99):
        raise ValueError(f"Score {score_float} is outside [-0.99, 0.99]")

    return score_float


def build_move_prompt(
    board: JetanBoard,
    actions: tuple[Move, ...],
    recent_moves: tuple[Move, ...],
) -> Sequence[dict[str, str]]:
    """Build move prompt instructing model to return {"move": "..."}."""
    action_strs = [str(a) for a in actions]
    lines = [
        f"You are an expert Jetan player playing as {board.turn.name}.",
        f"Your task is to select exactly one move from the authoritative legal move list.",
        "Output ONLY a single raw JSON object with no markdown fences, no surrounding commentary, and no extra text.",
        'Exact schema: {"move": "<source>-<dest>"}',
        "",
        f"Board:\n{str(board)}",
        f"Princess escape used: Orange={board.used_escape[0]}, Black={board.used_escape[1]}",
    ]
    if recent_moves:
        lines.append(f"Recent moves: {', '.join(str(m) for m in recent_moves)}")

    lines.append(f"Legal moves ({len(actions)} available): {', '.join(action_strs)}")
    lines.append("Choose your move:")

    return [
        {
            "role": "system",
            "content": (
                "You are an adversarial game agent. "
                "Output ONLY a single raw JSON object: {\"move\": \"<source>-<dest>\"}. "
                "The move MUST be from the provided legal move list."
            ),
        },
        {"role": "user", "content": "\n".join(lines)},
    ]


def parse_move_response(response: str, actions: tuple[Move, ...]) -> Move:
    """Strictly parse JSON response for move in actions."""
    text = response.strip()
    if not text:
        raise ValueError("Empty response")
    if "```" in text:
        raise ValueError("Response contains markdown fences")

    parsed = json.loads(text, object_pairs_hook=_no_duplicate_keys_decoder)
    if not isinstance(parsed, dict) or set(parsed.keys()) != {"move"}:
        raise ValueError("Schema mismatch: must contain exactly 'move'")

    move_val = parsed["move"]
    if not isinstance(move_val, str):
        raise ValueError("Move must be a string")

    action_map = {str(a): a for a in actions}
    if move_val not in action_map:
        raise ValueError(f"Move '{move_val}' is not in legal actions")

    return action_map[move_val]


def choose_fallback(board: JetanBoard, actions: tuple[Move, ...]) -> Move:
    """Greedy deterministic fallback: prefer capture of highest-value piece, or highest evaluate_position_3."""
    best_move = actions[0]
    best_score = -math.inf
    perspective = board.turn

    for action in actions:
        captured = board.at(action.destination)
        if captured is not None:
            score = 10.0 + PIECE_VALUES[captured.kind]
        else:
            succ = board.result(action)
            score = evaluate_position_3(succ, perspective)
        if score > best_score:
            best_score = score
            best_move = action

    return best_move