"""Minimum behavioral contracts for student-completed code."""

import unittest

from agents import AStarSearchAgent, UniformCostSearchAgent
from problem import RecoveryProblem
from recovery import Action, TrafficMode
from scenarios import get_scenario


class ProblemFormulationTests(unittest.TestCase):
    def test_example_initial_action_is_isolate(self) -> None:
        problem = RecoveryProblem(get_scenario(1).initial_state)

        self.assertEqual(
            problem.actions(problem.initial_state), (Action.ISOLATE_TRAFFIC,)
        )

    def test_isolate_returns_a_new_state(self) -> None:
        problem = RecoveryProblem(get_scenario(1).initial_state)

        next_state = problem.result(problem.initial_state, Action.ISOLATE_TRAFFIC)

        self.assertIs(next_state.traffic, TrafficMode.ISOLATED)
        self.assertIsNot(next_state, problem.initial_state)

    def test_goal_scenario_is_recognized(self) -> None:
        problem = RecoveryProblem(get_scenario(0).initial_state)

        self.assertTrue(problem.is_goal(problem.initial_state))

    def test_illegal_transition_raises_value_error(self) -> None:
        problem = RecoveryProblem(get_scenario(1).initial_state)

        with self.assertRaises(ValueError):
            problem.result(problem.initial_state, Action.RESTART_DATABASE)


class SearchAgentTests(unittest.TestCase):
    def test_ucs_finds_example_optimal_cost(self) -> None:
        problem = RecoveryProblem(get_scenario(1).initial_state)

        result = UniformCostSearchAgent().plan(problem)

        self.assertIsNotNone(result.plan)
        assert result.plan is not None
        cost = 0
        state = problem.initial_state
        for action in result.plan:
            next_state = problem.result(state, action)
            cost += problem.step_cost(state, action, next_state)
            state = next_state
        self.assertTrue(problem.is_goal(state))
        self.assertEqual(cost, 16)

    def test_astar_goal_heuristic_is_zero(self) -> None:
        goal = get_scenario(0).initial_state

        self.assertEqual(AStarSearchAgent().heuristic(goal), 0)

    def test_astar_finds_example_optimal_cost(self) -> None:
        problem = RecoveryProblem(get_scenario(1).initial_state)

        result = AStarSearchAgent().plan(problem)

        self.assertIsNotNone(result.plan)
        assert result.plan is not None
        state = problem.initial_state
        cost = 0
        for action in result.plan:
            next_state = problem.result(state, action)
            cost += problem.step_cost(state, action, next_state)
            state = next_state
        self.assertTrue(problem.is_goal(state))
        self.assertEqual(cost, 16)


if __name__ == "__main__":
    unittest.main()
