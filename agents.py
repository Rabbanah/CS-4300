"""Agent implementations for U0-HW-02."""

from __future__ import annotations

import random
from typing import Protocol

from hallway import Action, Percept


class Agent(Protocol):
    def reset(self) -> None:
        """Clear episode-specific state."""

    def act(self, percept: Percept) -> Action:
        """Choose one action from the current percept."""


class RandomAgent:
    """Provided baseline that deliberately ignores every percept."""

    def __init__(self, seed: int) -> None:
        self._rng = random.Random(seed)

    def reset(self) -> None:
        pass

    def act(self, percept: Percept) -> Action:
        del percept
        return self._rng.choice(list(Action))


class SimpleReflexAgent:
    """A condition-action agent with no memory between calls to act."""

    def reset(self) -> None:
        pass

    def act(self, percept: Percept) -> Action:
        # TODO: Implement rules that depend only on the current percept.
        raise NotImplementedError("Implement SimpleReflexAgent.act")


class ModelBasedReflexAgent:
    """A reflex agent that first updates an internal hallway model."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        # TODO: Initialize all episode-specific internal state.
        pass

    def act(self, percept: Percept) -> Action:
        # TODO: Update the model from the percept, then apply condition-action rules.
        raise NotImplementedError("Implement ModelBasedReflexAgent.act")
