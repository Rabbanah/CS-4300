"""Seeded random baseline agent."""

import random

from jetan import JetanBoard, Move


class RandomJetanAgent:
    """Choose uniformly from the current legal actions."""

    def __init__(self, seed: int, name: str = "Random") -> None:
        self.name = name
        self._rng = random.Random(seed)

    def choose_action(self, board: JetanBoard) -> Move:
        actions = board.actions()
        if not actions:
            raise RuntimeError("RandomJetanAgent received a terminal board")
        return self._rng.choice(actions)
