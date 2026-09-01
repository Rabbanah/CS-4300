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
        if percept.carrying and percept.destination_here:
            return Action.DROP_OFF

        if not percept.carrying and percept.package_here:
            return Action.PICK_UP

        if percept.location == "A":
            return Action.MOVE_RIGHT
        if percept.location == "E":
            return Action.MOVE_LEFT

        return Action.MOVE_RIGHT


class ModelBasedReflexAgent:
    """A reflex agent that first updates an internal hallway model."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.package_loc: str | None = None
        self.dest_loc: str | None = None
        self.explore_direction: Action = Action.MOVE_RIGHT

    def act(self, percept: Percept) -> Action:
        hallway = ["A", "B", "C", "D", "E"]
        curr_loc = percept.location

        if percept.package_here:
            self.package_loc = curr_loc
        if percept.destination_here:
            self.dest_loc = curr_loc
        if percept.carrying:
            self.package_loc = "CARRIED"

        if curr_loc == "A":
            self.explore_direction = Action.MOVE_RIGHT
        elif curr_loc == "E":
            self.explore_direction = Action.MOVE_LEFT

        if percept.carrying and percept.destination_here:
            return Action.DROP_OFF

        if not percept.carrying and percept.package_here:
            return Action.PICK_UP

        if percept.carrying and self.dest_loc is not None:
            if hallway.index(self.dest_loc) > hallway.index(curr_loc):
                return Action.MOVE_RIGHT
            elif hallway.index(self.dest_loc) < hallway.index(curr_loc):
                return Action.MOVE_LEFT

        if not percept.carrying and self.package_loc is not None and self.package_loc != "CARRIED":
            if hallway.index(self.package_loc) > hallway.index(curr_loc):
                return Action.MOVE_RIGHT
            elif hallway.index(self.package_loc) < hallway.index(curr_loc):
                return Action.MOVE_LEFT

        return self.explore_direction
