"""Tests for the provided environment, scenarios, and random baseline."""

import unittest

from agents import RandomRecoveryAgent
from recovery import (
    ACTION_COSTS,
    Action,
    BackupStatus,
    DatabaseStatus,
    RecoveryState,
    ServiceRecoveryEnvironment,
    ServiceStatus,
    StorageStatus,
    TrafficMode,
    ValidationStatus,
)
from scenarios import evaluation_scenarios, get_scenario


def example_state() -> RecoveryState:
    return RecoveryState(
        storage=StorageStatus.HEALTHY,
        database=DatabaseStatus.DOWN,
        auth=ServiceStatus.HEALTHY,
        api=ServiceStatus.FAULTY,
        traffic=TrafficMode.NORMAL,
        backup=BackupStatus.UNAVAILABLE,
        validation=ValidationStatus.NOT_RUN,
    )


class RecoveryEnvironmentTests(unittest.TestCase):
    def test_example_recovery_costs_16(self) -> None:
        environment = ServiceRecoveryEnvironment(example_state())
        plan = (
            Action.ISOLATE_TRAFFIC,
            Action.RESTART_DATABASE,
            Action.ROLLBACK_API,
            Action.VALIDATE_STACK,
            Action.ENABLE_CANARY,
            Action.ENABLE_NORMAL,
        )

        for action in plan:
            environment.step(action)

        self.assertTrue(environment.recovered)
        self.assertEqual(environment.total_cost, 16)
        self.assertEqual(environment.action_count, 6)

    def test_illegal_action_does_not_change_environment(self) -> None:
        environment = ServiceRecoveryEnvironment(example_state())

        with self.assertRaises(ValueError):
            environment.step(Action.RESTART_DATABASE)

        self.assertEqual(environment.state, example_state())
        self.assertEqual(environment.action_count, 0)
        self.assertEqual(environment.total_cost, 0)

    def test_service_change_resets_validation(self) -> None:
        state = RecoveryState(
            StorageStatus.HEALTHY,
            DatabaseStatus.DOWN,
            ServiceStatus.HEALTHY,
            ServiceStatus.HEALTHY,
            TrafficMode.ISOLATED,
            BackupStatus.AVAILABLE,
            ValidationStatus.PASSED,
        )
        environment = ServiceRecoveryEnvironment(state)

        transition = environment.step(Action.RESTART_DATABASE)

        self.assertIs(transition.after.validation, ValidationStatus.NOT_RUN)

    def test_restore_consumes_backup(self) -> None:
        state = RecoveryState(
            StorageStatus.HEALTHY,
            DatabaseStatus.CORRUPT,
            ServiceStatus.HEALTHY,
            ServiceStatus.HEALTHY,
            TrafficMode.ISOLATED,
            BackupStatus.AVAILABLE,
            ValidationStatus.NOT_RUN,
        )
        environment = ServiceRecoveryEnvironment(state)

        transition = environment.step(Action.RESTORE_DATABASE)

        self.assertIs(transition.after.database, DatabaseStatus.HEALTHY)
        self.assertIs(transition.after.backup, BackupStatus.UNAVAILABLE)
        self.assertEqual(transition.cost, ACTION_COSTS[Action.RESTORE_DATABASE])

    def test_restarting_faulty_auth_requires_a_second_restart(self) -> None:
        state = RecoveryState(
            StorageStatus.HEALTHY,
            DatabaseStatus.HEALTHY,
            ServiceStatus.FAULTY,
            ServiceStatus.HEALTHY,
            TrafficMode.ISOLATED,
            BackupStatus.AVAILABLE,
            ValidationStatus.NOT_RUN,
        )
        environment = ServiceRecoveryEnvironment(state)

        first = environment.step(Action.RESTART_AUTH)
        second = environment.step(Action.RESTART_AUTH)

        self.assertIs(first.after.auth, ServiceStatus.DOWN)
        self.assertIs(second.after.auth, ServiceStatus.HEALTHY)
        self.assertEqual(environment.total_cost, 6)

    def test_random_agent_returns_a_legal_action(self) -> None:
        environment = ServiceRecoveryEnvironment(example_state())
        legal_actions = environment.legal_actions()

        action = RandomRecoveryAgent(7).act(environment.state, legal_actions)

        self.assertIn(action, legal_actions)


class ScenarioTests(unittest.TestCase):
    def test_suite_contains_50_unique_scenarios(self) -> None:
        scenarios = evaluation_scenarios()
        states = {scenario.initial_state for scenario in scenarios}

        self.assertEqual(len(scenarios), 50)
        self.assertEqual(len(states), 50)

    def test_corrupt_initial_database_has_backup(self) -> None:
        for scenario in evaluation_scenarios():
            if scenario.initial_state.database is DatabaseStatus.CORRUPT:
                self.assertIs(
                    scenario.initial_state.backup, BackupStatus.AVAILABLE
                )

    def test_scenario_lookup_is_stable(self) -> None:
        self.assertEqual(get_scenario(17), get_scenario(17))
        with self.assertRaises(ValueError):
            get_scenario(50)


if __name__ == "__main__":
    unittest.main()
