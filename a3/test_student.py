"""Required student-authored tests for U0-HW-04."""

import unittest

from agents import LLMRepairAgent, MinConflictsAgent
from conference import Placement, make_schedule, move
from constraints import evaluate
from llm_client import ScriptedClient
from scenarios import evaluation_scenarios


class StudentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scenarios = evaluation_scenarios()
        self.scenario = self.scenarios[0]
        self.problem = self.scenario.problem

    def test_exact_hard_counting(self) -> None:
        """Verify room collisions, speaker overlaps, and audience conflicts are counted."""
        
        eval_res = evaluate(self.problem, self.scenario.initial_schedule)
        self.assertEqual(eval_res.hard_violations, 5)
        self.assertIn("T1", eval_res.conflicted_talks)
        self.assertIn("T2", eval_res.conflicted_talks)
        self.assertIn("T3", eval_res.conflicted_talks)
        self.assertIn("T4", eval_res.conflicted_talks)
        self.assertIn("T5", eval_res.conflicted_talks)
        self.assertEqual(len(eval_res.details), 5)

    def test_exact_soft_penalty(self) -> None:
        """Verify missed preferred slots and track dispersion penalty."""
        eval_res = evaluate(self.problem, self.scenario.initial_schedule)
        
        self.assertEqual(eval_res.soft_penalty, 5)

    def test_invalid_and_out_of_domain_move(self) -> None:
        """Ensure move() rejects out-of-domain placements."""
        invalid_placement = Placement("Canyon", "09:00")
        with self.assertRaises(ValueError):
            move(self.problem, self.scenario.initial_schedule, "T1", invalid_placement)

    def test_min_conflicts_determinism_and_success(self) -> None:
        """Verify min-conflicts reproduces the exact trace for the same seed and achieves feasibility."""
        agent = MinConflictsAgent()
        res1 = agent.repair(self.problem, self.scenario.initial_schedule, budget=20, seed=42)
        res2 = agent.repair(self.problem, self.scenario.initial_schedule, budget=20, seed=42)
        self.assertTrue(res1.feasible)
        self.assertEqual(res1.schedule, res2.schedule)
        self.assertEqual(res1.trace, res2.trace)
        self.assertEqual(res1.metrics.evaluator_calls, res2.metrics.evaluator_calls)

    def test_llm_duplicate_key_rejection(self) -> None:
        """Verify LLMRepairAgent rejects responses with duplicate JSON keys."""
        bad_json = '{"tool":"move","tool":"move","talk_id":"T2","room":"Canyon","slot":"10:30"}'
        client = ScriptedClient([bad_json])
        agent = LLMRepairAgent(client)
        res = agent.repair(self.problem, self.scenario.initial_schedule, budget=1, seed=0)
        self.assertEqual(res.metrics.malformed_requests, 1)
        self.assertEqual(res.metrics.edits, 0)
        self.assertIn("malformed", res.trace[0].outcome)

    def test_llm_zero_history_behavior(self) -> None:
        """Verify history_limit=0 includes no previous outcomes in prompt."""
        valid_json = '{"tool":"move","talk_id":"T2","room":"Canyon","slot":"10:30"}'
        client = ScriptedClient([valid_json, valid_json])
        agent = LLMRepairAgent(client, history_limit=0)
        agent.repair(self.problem, self.scenario.initial_schedule, budget=2, seed=0)
        second_prompt = client.messages_seen[1][1]["content"]
        self.assertNotIn("Recent Outcomes:", second_prompt)

    def test_llm_failure_categorization(self) -> None:
        """Verify malformed, invalid, repeated, and client failures are properly tracked."""
        responses = [
            "not a json",
            '{"tool":"move","talk_id":"UNKNOWN","room":"Canyon","slot":"10:30"}',
            '{"tool":"move","talk_id":"T2","room":"Canyon","slot":"10:30"}',
            '{"tool":"move","talk_id":"T2","room":"Canyon","slot":"10:30"}',
        ]
        client = ScriptedClient(responses)
        agent = LLMRepairAgent(client)
        res = agent.repair(self.problem, self.scenario.initial_schedule, budget=4, seed=0)

        self.assertEqual(res.metrics.malformed_requests, 1)
        self.assertEqual(res.metrics.invalid_requests, 1)
        self.assertEqual(res.metrics.edits, 1)
        self.assertEqual(res.metrics.repeated_requests, 1)

    def test_llm_client_failure_handling(self) -> None:
        """Verify standard network/client exceptions are caught and counted."""
        class FailingClient:
            def chat(self, messages: object) -> str:
                raise ConnectionResetError("Connection dropped")

        agent = LLMRepairAgent(FailingClient())
        res = agent.repair(self.problem, self.scenario.initial_schedule, budget=2, seed=0)
        self.assertEqual(res.metrics.client_failures, 2)
        self.assertEqual(res.metrics.model_calls, 2)
        self.assertEqual(res.metrics.edits, 0)
        self.assertIn("ConnectionResetError", res.trace[0].outcome)


if __name__ == "__main__":
    unittest.main()