"""Delivery Hallway environment for U0-HW-02.

Students should not modify this file.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum


class Action(Enum):
    MOVE_LEFT = "move_left"
    MOVE_RIGHT = "move_right"
    PICK_UP = "pick_up"
    DROP_OFF = "drop_off"
    WAIT = "wait"


@dataclass(frozen=True)
class Percept:
    location: str
    package_here: bool
    destination_here: bool
    carrying: bool


@dataclass(frozen=True)
class Transition:
    percept: Percept
    reward: int
    done: bool
    action_succeeded: bool


class HallwayEnvironment:
    """A deterministic, partially observable five-location hallway."""

    LOCATIONS = ("A", "B", "C", "D", "E")

    def __init__(self, seed: int, max_steps: int = 30) -> None:
        self.seed = seed
        self.max_steps = max_steps
        self._rng = random.Random(seed)
        self.reset()

    def reset(self) -> Percept:
        agent, package, destination = self._rng.sample(range(len(self.LOCATIONS)), 3)
        self.agent_position = agent
        self.package_position: int | None = package
        self.destination_position = destination
        self.carrying = False
        self.delivered = False
        self.steps = 0
        self.total_reward = 0
        self.done = False
        return self.percept()

    def percept(self) -> Percept:
        return Percept(
            location=self.LOCATIONS[self.agent_position],
            package_here=self.package_position == self.agent_position,
            destination_here=self.destination_position == self.agent_position,
            carrying=self.carrying,
        )

    def step(self, action: Action) -> Transition:
        if self.done:
            raise RuntimeError("Cannot act after an episode has ended")
        if not isinstance(action, Action):
            raise TypeError("action must be an Action value")

        reward = -1
        succeeded = True

        if action is Action.MOVE_LEFT:
            if self.agent_position > 0:
                self.agent_position -= 1
            else:
                succeeded = False
        elif action is Action.MOVE_RIGHT:
            if self.agent_position < len(self.LOCATIONS) - 1:
                self.agent_position += 1
            else:
                succeeded = False
        elif action is Action.PICK_UP:
            if self.package_position == self.agent_position and not self.carrying:
                self.package_position = None
                self.carrying = True
                reward += 5
            else:
                succeeded = False
        elif action is Action.DROP_OFF:
            if self.destination_position == self.agent_position and self.carrying:
                self.carrying = False
                self.delivered = True
                reward += 25
            else:
                succeeded = False

        if not succeeded:
            reward -= 2

        self.steps += 1
        self.total_reward += reward
        self.done = self.delivered or self.steps >= self.max_steps
        return Transition(self.percept(), reward, self.done, succeeded)
