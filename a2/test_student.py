"""Add the four required student-authored tests in this file."""

import unittest


class StudentRequiredTests(unittest.TestCase):
    def test_action_precondition_and_transition(self) -> None:
        self.fail("TODO: test one action precondition and transition")

    def test_goal_and_non_goal_states(self) -> None:
        self.fail("TODO: test one goal state and one non-goal state")

    def test_ucs_expected_optimal_cost(self) -> None:
        self.fail("TODO: test UCS on a scenario with a known optimal cost")

    def test_heuristic_values(self) -> None:
        self.fail("TODO: test a goal and at least two non-goal heuristic values")


if __name__ == "__main__":
    unittest.main()
