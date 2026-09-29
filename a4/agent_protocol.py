"""Shared interface for Jetan agents."""

from typing import Protocol

from jetan import JetanBoard, Move


class JetanAgent(Protocol):
    name: str

    def choose_action(self, board: JetanBoard) -> Move:
        """Choose one legal action from an immutable board instance."""
