"""Jetan match environment, result records, and viewer interfaces."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, replace
from typing import Mapping, Protocol

from agents import JetanAgent
from jetan import JetanBoard, Move, Player


DEFAULT_MAX_THINK_SECONDS = 300.0


class JetanViewer(Protocol):
    def start(self, board: JetanBoard, names: Mapping[Player, str]) -> None: ...

    def update(self, before: JetanBoard, action: Move, after: JetanBoard) -> None: ...

    def close(self, board: JetanBoard) -> None: ...


class NullViewer:
    def start(self, board: JetanBoard, names: Mapping[Player, str]) -> None:
        del board, names

    def update(self, before: JetanBoard, action: Move, after: JetanBoard) -> None:
        del before, action, after

    def close(self, board: JetanBoard) -> None:
        del board


@dataclass(frozen=True)
class MatchResult:
    final_board: JetanBoard
    winner: Player | None
    plies: int
    reason: str
    orange_think_seconds: float
    black_think_seconds: float

    def utility(self, player: Player) -> int:
        """Return the terminal game utility from one player's perspective."""
        return self.final_board.utility(player)


class JetanEnvironment:
    """Run agents against one authoritative immutable board state."""

    def __init__(
        self,
        orange: JetanAgent,
        black: JetanAgent,
        board: JetanBoard | None = None,
        viewer: JetanViewer | None = None,
        max_plies: int = 500,
        max_think_seconds: float = DEFAULT_MAX_THINK_SECONDS,
    ) -> None:
        if max_plies < 1:
            raise ValueError("max_plies must be positive")
        if not math.isfinite(max_think_seconds) or max_think_seconds <= 0:
            raise ValueError("max_think_seconds must be positive and finite")
        self.agents = {Player.ORANGE: orange, Player.BLACK: black}
        self.board = (board or JetanBoard.initial()).resolve_terminal()
        self.viewer = viewer or NullViewer()
        self.max_plies = max_plies
        self.max_think_seconds = max_think_seconds
        self.think_seconds = {Player.ORANGE: 0.0, Player.BLACK: 0.0}
        self.time_forfeit: Player | None = None

    def step(self) -> Move:
        if self.board.is_terminal():
            raise RuntimeError("cannot step a completed game")
        before = self.board
        player = before.player()
        started = time.perf_counter()
        try:
            action = self.agents[player].choose_action(before)
        finally:
            self.think_seconds[player] += time.perf_counter() - started
        if self.think_seconds[player] > self.max_think_seconds:
            self.time_forfeit = player
            self.board = replace(before, winner=player.opponent)
            return action
        self.board = before.result(action)
        self.viewer.update(before, action, self.board)
        return action

    def run(self) -> MatchResult:
        names = {player: agent.name for player, agent in self.agents.items()}
        reason = "terminal"
        try:
            self.viewer.start(self.board, names)
            while not self.board.is_terminal() and self.board.ply < self.max_plies:
                self.step()
            if not self.board.is_terminal():
                self.board = self.board.as_draw()
                reason = "move_limit"
            elif self.time_forfeit is not None:
                reason = "time_limit"
            elif self.board.draw:
                reason = "draw"
            else:
                reason = "win"
            return MatchResult(
                self.board,
                self.board.winner,
                self.board.ply,
                reason,
                self.think_seconds[Player.ORANGE],
                self.think_seconds[Player.BLACK],
            )
        finally:
            self.viewer.close(self.board)
