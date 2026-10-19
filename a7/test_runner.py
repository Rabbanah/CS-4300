"""Tests for the Rescue Rover agent-environment runner."""

import unittest

from agents import RandomAgent, StopAgent
from rover import Action, MissionStatus, Percept
from run_experiment import run_episode


class ScriptedAgent:
    def __init__(self, actions: list[Action]) -> None:
        self.actions = actions

    def reset(self) -> None:
        self.index = 0

    def act(self, percept: Percept) -> Action:
        del percept
        action = self.actions[self.index]
        self.index += 1
        return action


class ReusedAgent:
    def __init__(self) -> None:
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1

    def act(self, percept: Percept) -> Action:
        del percept
        return Action.stop()


class RunnerTests(unittest.TestCase):
    def test_stop_agent_produces_one_row_trace(self) -> None:
        result = run_episode(lambda seed: StopAgent(), scenario_id=0)

        self.assertIs(result.status, MissionStatus.STOPPED)
        self.assertEqual(result.actions, 1)
        self.assertEqual(len(result.trace), 1)
        self.assertEqual(result.score, -100)

    def test_random_agent_is_reproducible(self) -> None:
        first = run_episode(lambda seed: RandomAgent(seed), 0, seed=11)
        second = run_episode(lambda seed: RandomAgent(seed), 0, seed=11)

        self.assertEqual(first, second)

    def test_runner_resets_an_agent_returned_by_a_shared_factory(self) -> None:
        agent = ReusedAgent()
        factory = lambda seed: agent

        run_episode(factory, 0)
        run_episode(factory, 1)

        self.assertEqual(agent.reset_count, 2)


if __name__ == "__main__":
    unittest.main()
