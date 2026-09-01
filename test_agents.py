"""Minimum behavioral contracts for the two student-implemented agents."""

import unittest

from agents import ModelBasedReflexAgent, SimpleReflexAgent
from hallway import Action, Percept


class RequiredAgentTests(unittest.TestCase):
    def test_simple_reflex_picks_up_visible_package(self) -> None:
        agent = SimpleReflexAgent()
        percept = Percept("C", package_here=True, destination_here=False, carrying=False)
        self.assertIs(agent.act(percept), Action.PICK_UP)

    def test_simple_reflex_drops_off_at_destination(self) -> None:
        agent = SimpleReflexAgent()
        percept = Percept("B", package_here=False, destination_here=True, carrying=True)
        self.assertIs(agent.act(percept), Action.DROP_OFF)

    def test_model_based_picks_up_visible_package(self) -> None:
        agent = ModelBasedReflexAgent()
        percept = Percept("D", package_here=True, destination_here=False, carrying=False)
        self.assertIs(agent.act(percept), Action.PICK_UP)

    def test_model_based_drops_off_at_destination(self) -> None:
        agent = ModelBasedReflexAgent()
        percept = Percept("A", package_here=False, destination_here=True, carrying=True)
        self.assertIs(agent.act(percept), Action.DROP_OFF)

    def test_model_based_reset_allows_new_episode(self) -> None:
        agent = ModelBasedReflexAgent()
        agent.reset()
        action = agent.act(Percept("E", False, False, False))
        self.assertIsInstance(action, Action)


if __name__ == "__main__":
    unittest.main()
