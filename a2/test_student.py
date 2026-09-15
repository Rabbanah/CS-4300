"""Add the four required student-authored tests in this file."""

import unittest

from agents import AStarSearchAgent, UniformCostSearchAgent
from problem import RecoveryProblem
from recovery import (
    Action,
    BackupStatus,
    DatabaseStatus,
    RecoveryState,
    ServiceStatus,
    StorageStatus,
    TrafficMode,
    ValidationStatus,
)


class StudentRequiredTests(unittest.TestCase):
    def test_action_precondition_and_transition(self) -> None:
        """Test one action precondition and transition."""
        state = RecoveryState(
            storage=StorageStatus.DOWN,
            database=DatabaseStatus.HEALTHY,
            auth=ServiceStatus.HEALTHY,
            api=ServiceStatus.HEALTHY,
            traffic=TrafficMode.NORMAL,
            backup=BackupStatus.AVAILABLE,
            validation=ValidationStatus.NOT_RUN,
        )
        problem = RecoveryProblem(state)

        self.assertNotIn(Action.RESTART_STORAGE, problem.actions(state))
        with self.assertRaises(ValueError):
            problem.result(state, Action.RESTART_STORAGE)

        self.assertIn(Action.ISOLATE_TRAFFIC, problem.actions(state))
        next_state = problem.result(state, Action.ISOLATE_TRAFFIC)
        self.assertEqual(next_state.traffic, TrafficMode.ISOLATED)

        problem_isolated = RecoveryProblem(next_state)
        self.assertIn(Action.RESTART_STORAGE, problem_isolated.actions(next_state))
        repaired_state = problem_isolated.result(next_state, Action.RESTART_STORAGE)
        self.assertEqual(repaired_state.storage, StorageStatus.HEALTHY)

    def test_goal_and_non_goal_states(self) -> None:
        """Test one goal state and one non-goal state."""
        goal_state = RecoveryState(
            storage=StorageStatus.HEALTHY,
            database=DatabaseStatus.HEALTHY,
            auth=ServiceStatus.HEALTHY,
            api=ServiceStatus.HEALTHY,
            traffic=TrafficMode.NORMAL,
            backup=BackupStatus.UNAVAILABLE,
            validation=ValidationStatus.PASSED,
        )
        non_goal_state = RecoveryState(
            storage=StorageStatus.HEALTHY,
            database=DatabaseStatus.HEALTHY,
            auth=ServiceStatus.HEALTHY,
            api=ServiceStatus.HEALTHY,
            traffic=TrafficMode.CANARY,
            backup=BackupStatus.UNAVAILABLE,
            validation=ValidationStatus.PASSED,
        )
        problem = RecoveryProblem(goal_state)
        self.assertTrue(problem.is_goal(goal_state))
        self.assertFalse(problem.is_goal(non_goal_state))

    def test_ucs_expected_optimal_cost(self) -> None:
        """Test UCS on a scenario with a known optimal cost.

        Storage is HEALTHY, database is DOWN, auth is HEALTHY, api is FAULTY, traffic is NORMAL.
        Optimal plan:
          ISOLATE_TRAFFIC (1) -> RESTART_DATABASE (4) -> ROLLBACK_API (5)
          -> VALIDATE_STACK (2) -> ENABLE_CANARY (2) -> ENABLE_NORMAL (2)
        Expected optimal cost = 16.
        """
        init_state = RecoveryState(
            storage=StorageStatus.HEALTHY,
            database=DatabaseStatus.DOWN,
            auth=ServiceStatus.HEALTHY,
            api=ServiceStatus.FAULTY,
            traffic=TrafficMode.NORMAL,
            backup=BackupStatus.AVAILABLE,
            validation=ValidationStatus.NOT_RUN,
        )
        problem = RecoveryProblem(init_state)
        agent = UniformCostSearchAgent()
        result = agent.plan(problem)

        self.assertIsNotNone(result.plan)
        expected_plan = (
            Action.ISOLATE_TRAFFIC,
            Action.RESTART_DATABASE,
            Action.ROLLBACK_API,
            Action.VALIDATE_STACK,
            Action.ENABLE_CANARY,
            Action.ENABLE_NORMAL,
        )
        self.assertEqual(result.plan, expected_plan)

        curr = init_state
        total_cost = 0
        for action in result.plan:
            nxt = problem.result(curr, action)
            total_cost += problem.step_cost(curr, action, nxt)
            curr = nxt
        self.assertEqual(total_cost, 16)

    def test_heuristic_values(self) -> None:
        """Test heuristic values for a goal state and two non-goal states."""
        agent = AStarSearchAgent()

        goal_state = RecoveryState(
            storage=StorageStatus.HEALTHY,
            database=DatabaseStatus.HEALTHY,
            auth=ServiceStatus.HEALTHY,
            api=ServiceStatus.HEALTHY,
            traffic=TrafficMode.NORMAL,
            backup=BackupStatus.AVAILABLE,
            validation=ValidationStatus.PASSED,
        )
        self.assertEqual(agent.heuristic(goal_state), 0)

        ready_to_validate = RecoveryState(
            storage=StorageStatus.HEALTHY,
            database=DatabaseStatus.HEALTHY,
            auth=ServiceStatus.HEALTHY,
            api=ServiceStatus.HEALTHY,
            traffic=TrafficMode.ISOLATED,
            backup=BackupStatus.AVAILABLE,
            validation=ValidationStatus.NOT_RUN,
        )
        self.assertEqual(agent.heuristic(ready_to_validate), 6)

        down_storage = RecoveryState(
            storage=StorageStatus.DOWN,
            database=DatabaseStatus.HEALTHY,
            auth=ServiceStatus.HEALTHY,
            api=ServiceStatus.HEALTHY,
            traffic=TrafficMode.NORMAL,
            backup=BackupStatus.AVAILABLE,
            validation=ValidationStatus.NOT_RUN,
        )
        self.assertEqual(agent.heuristic(down_storage), 11)


if __name__ == "__main__":
    unittest.main()