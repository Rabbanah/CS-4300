"""Tests for the provided environment and random baseline."""

import unittest

from agents import RandomAgent
from hallway import Action, HallwayEnvironment


class HallwayEnvironmentTests(unittest.TestCase):
    def test_seed_reproduces_initial_state(self) -> None:
        first = HallwayEnvironment(17)
        second = HallwayEnvironment(17)
        self.assertEqual(first.percept(), second.percept())
        self.assertEqual(first.package_position, second.package_position)
        self.assertEqual(first.destination_position, second.destination_position)

    def test_invalid_action_has_extra_penalty(self) -> None:
        environment = HallwayEnvironment(8)
        transition = environment.step(Action.DROP_OFF)
        self.assertFalse(transition.action_succeeded)
        self.assertEqual(transition.reward, -3)

    def test_package_can_be_delivered(self) -> None:
        environment = HallwayEnvironment(5)
        while environment.agent_position < environment.package_position:
            environment.step(Action.MOVE_RIGHT)
        while environment.agent_position > environment.package_position:
            environment.step(Action.MOVE_LEFT)
        pickup = environment.step(Action.PICK_UP)
        self.assertTrue(pickup.action_succeeded)

        while environment.agent_position < environment.destination_position:
            environment.step(Action.MOVE_RIGHT)
        while environment.agent_position > environment.destination_position:
            environment.step(Action.MOVE_LEFT)
        delivery = environment.step(Action.DROP_OFF)
        self.assertTrue(delivery.action_succeeded)
        self.assertTrue(delivery.done)
        self.assertTrue(environment.delivered)

    def test_random_agent_returns_an_action(self) -> None:
        environment = HallwayEnvironment(3)
        action = RandomAgent(3).act(environment.percept())
        self.assertIsInstance(action, Action)


if __name__ == "__main__":
    unittest.main()
