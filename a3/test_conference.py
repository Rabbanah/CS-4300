"""Tests for provided immutable model and scenario resources."""

import unittest
from dataclasses import FrozenInstanceError

from conference import Placement, is_structurally_valid, move
from scenarios import evaluation_scenarios


class ConferenceResourceTests(unittest.TestCase):
    def test_domains_enforce_unary_requirements(self) -> None:
        scenario = evaluation_scenarios()[0]
        domain = scenario.problem.domain("T1")
        self.assertTrue(domain)
        self.assertEqual({p.room for p in domain}, {"Auditorium"})

    def test_schedule_and_move_are_immutable_and_complete(self) -> None:
        scenario = evaluation_scenarios()[0]
        before = scenario.initial_schedule
        after = move(scenario.problem, before, "T2", Placement("Canyon", "10:30"))
        self.assertNotEqual(before, after)
        self.assertTrue(is_structurally_valid(scenario.problem, after))
        self.assertEqual(len(after.assignments), len(scenario.problem.talks))
        with self.assertRaises(FrozenInstanceError):
            after.assignments = ()  # type: ignore[misc]

    def test_suite_is_fixed_and_initial_schedules_are_structural(self) -> None:
        first = evaluation_scenarios()
        second = evaluation_scenarios()
        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)
        self.assertTrue(all(is_structurally_valid(s.problem, s.initial_schedule) for s in first))


if __name__ == "__main__":
    unittest.main()
