"""Minimum behavioral requirements for the student LogicAgent."""

import unittest

from agents import LogicAgent
from rover import Action, ActionKind, MissionStatus, RescueRoverEnvironment
from run_experiment import FACTORIES, run_scenario
from scenarios import generated_scenarios, training_scenarios


class LogicAgentTests(unittest.TestCase):
    def assert_discovery_precedes_execution(self, result) -> None:
        kinds = tuple(row.action.kind for row in result.trace)
        first_move = kinds.index(ActionKind.MOVE)
        scan_kinds = (ActionKind.STRUCTURAL_SCAN, ActionKind.SURFACE_SCAN)

        self.assertTrue(any(kind in scan_kinds for kind in kinds[:first_move]))
        self.assertFalse(any(kind in scan_kinds for kind in kinds[first_move:]))

    def test_logic_agent_is_registered(self) -> None:
        self.assertIn("logic", FACTORIES)
        self.assertIsInstance(FACTORIES["logic"](4300), LogicAgent)

    def test_logic_agent_returns_an_action(self) -> None:
        scenario = training_scenarios()[0]
        environment = RescueRoverEnvironment(scenario)
        agent = LogicAgent(4300)
        agent.reset()

        self.assertIsInstance(agent.act(environment.percept()), Action)

    def test_reset_clears_episode_knowledge(self) -> None:
        agent = LogicAgent(4300)
        agent.knowledge.add_formula("temporary_fact")
        agent.initialized = True
        agent.last_processed_action = 7

        agent.reset()

        self.assertEqual(agent.knowledge.clauses, ())
        self.assertFalse(agent.initialized)
        self.assertEqual(agent.last_processed_action, -1)

    def test_logic_agent_completes_all_training_scenarios(self) -> None:
        for scenario in training_scenarios():
            with self.subTest(scenario=scenario.identifier):
                result = run_scenario(lambda seed: LogicAgent(seed), scenario, 4300)
                self.assertIs(result.status, MissionStatus.SUCCESS)
                self.assertEqual(result.unprotected_slips, 0)
                self.assertEqual(result.invalid_actions, 0)
                self.assert_discovery_precedes_execution(result)

    def test_logic_agent_completes_fixed_generated_suite(self) -> None:
        for scenario in generated_scenarios(9001, 30):
            with self.subTest(scenario=scenario.identifier):
                result = run_scenario(lambda seed: LogicAgent(seed), scenario, 4300)
                self.assertIs(result.status, MissionStatus.SUCCESS)
                self.assertEqual(result.unprotected_slips, 0)
                self.assertEqual(result.invalid_actions, 0)
                self.assert_discovery_precedes_execution(result)

    def test_logic_agent_is_reproducible(self) -> None:
        scenario = training_scenarios()[1]

        first = run_scenario(lambda seed: LogicAgent(seed), scenario, seed=19)
        second = run_scenario(lambda seed: LogicAgent(seed), scenario, seed=19)

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
