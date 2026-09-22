"""Interface checks that pass after the required student work is complete."""

import unittest

from agents import LLMRepairAgent, MinConflictsAgent
from constraints import evaluate
from llm_client import ScriptedClient
from scenarios import evaluation_scenarios


class RequiredImplementationTests(unittest.TestCase):
    def test_evaluator_and_min_conflicts(self) -> None:
        scenario = evaluation_scenarios()[0]
        initial = evaluate(scenario.problem, scenario.initial_schedule)
        self.assertGreater(initial.hard_violations, 0)
        result = MinConflictsAgent().repair(scenario.problem, scenario.initial_schedule, scenario.budget, 7)
        self.assertTrue(result.feasible)

    def test_llm_agent_applies_valid_json_move(self) -> None:
        scenario = evaluation_scenarios()[0]
        client = ScriptedClient([
            '{"tool":"move","talk_id":"T2","room":"Canyon","slot":"10:30"}',
            '{"tool":"move","talk_id":"T3","room":"Mesa","slot":"10:30"}',
            '{"tool":"move","talk_id":"T5","room":"Canyon","slot":"13:00"}',
        ])
        result = LLMRepairAgent(client).repair(scenario.problem, scenario.initial_schedule, 3, 0)
        self.assertEqual(result.metrics.model_calls, 3)
        self.assertEqual(result.metrics.edits, 3)


if __name__ == "__main__":
    unittest.main()
