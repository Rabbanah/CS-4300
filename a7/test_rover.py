"""Tests for the provided Rescue Rover environment."""

import unittest

from rover import (
    Action,
    Event,
    MissionStatus,
    Position,
    RescueRoverEnvironment,
    RoverScenario,
    Scanner,
    Terrain,
)


def small_scenario() -> RoverScenario:
    return RoverScenario(
        identifier="test",
        terrain=(
            (Terrain.STABLE, Terrain.SLIPPERY, Terrain.STABLE),
            (Terrain.STABLE, Terrain.FRAGILE, Terrain.STABLE),
            (Terrain.STABLE, Terrain.STABLE, Terrain.STABLE),
        ),
        start=Position(0, 0),
        module=Position(0, 2),
        exit=Position(2, 2),
    )


class RoverEnvironmentTests(unittest.TestCase):
    def test_percept_does_not_expose_hidden_terrain(self) -> None:
        percept = RescueRoverEnvironment(small_scenario()).percept()

        self.assertFalse(hasattr(percept, "terrain"))
        self.assertEqual(percept.position, Position(0, 0))
        self.assertEqual(percept.module_position, Position(0, 2))

    def test_structural_scan_returns_one_boolean_for_region(self) -> None:
        environment = RescueRoverEnvironment(small_scenario())
        transition = environment.step(
            Action.structural_scan(
                Position(1, 0), Position(1, 1), Position(1, 2)
            )
        )

        observation = transition.outcome.scan
        self.assertIsNotNone(observation)
        assert observation is not None
        self.assertIs(observation.scanner, Scanner.STRUCTURAL)
        self.assertTrue(observation.positive)
        self.assertEqual(transition.outcome.score_delta, -2)

    def test_surface_scan_detects_slippery_tile(self) -> None:
        environment = RescueRoverEnvironment(small_scenario())
        transition = environment.step(
            Action.surface_scan(
                Position(0, 0),
                Position(0, 1),
                Position(1, 0),
                Position(1, 1),
            )
        )

        self.assertTrue(transition.outcome.scan.positive)  # type: ignore[union-attr]

    def test_scan_observation_canonicalizes_region_order(self) -> None:
        environment = RescueRoverEnvironment(small_scenario())
        transition = environment.step(
            Action.surface_scan(
                Position(1, 1),
                Position(0, 0),
                Position(1, 0),
                Position(0, 1),
            )
        )

        self.assertEqual(
            transition.outcome.scan.region,  # type: ignore[union-attr]
            (
                Position(0, 0),
                Position(0, 1),
                Position(1, 0),
                Position(1, 1),
            ),
        )

    def test_vertical_remote_structural_scan_can_be_negative(self) -> None:
        environment = RescueRoverEnvironment(small_scenario())
        transition = environment.step(
            Action.structural_scan(
                Position(0, 2), Position(1, 2), Position(2, 2)
            )
        )

        self.assertTrue(transition.outcome.succeeded)
        self.assertFalse(transition.outcome.scan.positive)  # type: ignore[union-attr]
        self.assertEqual(environment.position, Position(0, 0))

    def test_invalid_scan_shape_costs_five_and_does_not_scan(self) -> None:
        environment = RescueRoverEnvironment(small_scenario())
        transition = environment.step(
            Action.structural_scan(Position(0, 0), Position(0, 1))
        )

        self.assertIs(transition.outcome.event, Event.INVALID_ACTION)
        self.assertEqual(transition.outcome.score_delta, -5)
        self.assertEqual(environment.scan_count, 0)

    def test_out_of_bounds_scan_is_invalid_and_preserves_state(self) -> None:
        environment = RescueRoverEnvironment(small_scenario())
        transition = environment.step(
            Action.structural_scan(
                Position(0, 1), Position(0, 2), Position(0, 3)
            )
        )

        self.assertIs(transition.outcome.event, Event.INVALID_ACTION)
        self.assertEqual(environment.position, Position(0, 0))
        self.assertFalse(environment.traction_enabled)

    def test_fragile_tile_destroys_rover(self) -> None:
        environment = RescueRoverEnvironment(small_scenario())
        environment.step(Action.move(Position(1, 0)))
        transition = environment.step(Action.move(Position(1, 1)))

        self.assertIs(transition.status, MissionStatus.DESTROYED)
        self.assertIs(transition.outcome.event, Event.FRAGILE_COLLAPSE)
        self.assertEqual(transition.outcome.score_delta, -1003)

    def test_slippery_tile_without_traction_penalizes_and_continues(self) -> None:
        environment = RescueRoverEnvironment(small_scenario())
        transition = environment.step(Action.move(Position(0, 1)))

        self.assertIs(transition.status, MissionStatus.RUNNING)
        self.assertTrue(transition.outcome.succeeded)
        self.assertIs(transition.outcome.event, Event.SLIPPED)
        self.assertEqual(transition.outcome.score_delta, -103)
        self.assertEqual(environment.position, Position(0, 1))
        self.assertEqual(environment.unprotected_slips, 1)

        continued = environment.step(Action.move(Position(0, 2)))

        self.assertIs(continued.status, MissionStatus.RUNNING)
        self.assertEqual(continued.outcome.score_delta, -3)

    def test_traction_allows_slippery_move_without_stable_surcharge(self) -> None:
        environment = RescueRoverEnvironment(small_scenario())
        enabled = environment.step(Action.enable_traction())
        moved = environment.step(Action.move(Position(0, 1)))

        self.assertEqual(enabled.outcome.score_delta, -1)
        self.assertIs(moved.status, MissionStatus.RUNNING)
        self.assertEqual(moved.outcome.score_delta, -3)
        self.assertEqual(environment.traction_moves, 1)
        self.assertEqual(environment.traction_toggles, 1)
        self.assertEqual(environment.unprotected_slips, 0)

    def test_crossing_one_slippery_tile_beats_three_move_detour(self) -> None:
        environment = RescueRoverEnvironment(small_scenario())

        environment.step(Action.enable_traction())
        environment.step(Action.move(Position(0, 1)))
        environment.step(Action.disable_traction())

        self.assertEqual(environment.total_score, -5)
        self.assertGreater(environment.total_score, 3 * environment.score_rules.move)

    def test_toggling_around_stable_move_beats_leaving_traction_on(self) -> None:
        left_on = RescueRoverEnvironment(small_scenario())
        left_on.step(Action.enable_traction())
        left_on_start = left_on.total_score
        left_on.step(Action.move(Position(1, 0)))

        switched = RescueRoverEnvironment(small_scenario())
        switched.step(Action.enable_traction())
        switched_start = switched.total_score
        switched.step(Action.disable_traction())
        switched.step(Action.move(Position(1, 0)))
        switched.step(Action.enable_traction())

        self.assertEqual(left_on.total_score - left_on_start, -6)
        self.assertEqual(switched.total_score - switched_start, -5)
        self.assertGreater(
            switched.total_score - switched_start,
            left_on.total_score - left_on_start,
        )

    def test_redundant_traction_toggle_is_invalid(self) -> None:
        environment = RescueRoverEnvironment(small_scenario())
        environment.step(Action.enable_traction())
        transition = environment.step(Action.enable_traction())

        self.assertIs(transition.outcome.event, Event.INVALID_ACTION)
        self.assertTrue(environment.traction_enabled)

    def test_retrieve_and_stop_at_exit_complete_mission(self) -> None:
        scenario = RoverScenario(
            identifier="short",
            terrain=((Terrain.STABLE, Terrain.STABLE, Terrain.STABLE),),
            start=Position(0, 0),
            module=Position(0, 1),
            exit=Position(0, 2),
        )
        environment = RescueRoverEnvironment(scenario)
        environment.step(Action.move(Position(0, 1)))
        environment.step(Action.retrieve())
        environment.step(Action.move(Position(0, 2)))
        transition = environment.step(Action.stop())

        self.assertIs(transition.status, MissionStatus.SUCCESS)
        self.assertEqual(environment.total_score, 9993)

    def test_success_bonus_preserves_mission_completion_priority(self) -> None:
        rules = RescueRoverEnvironment(small_scenario()).score_rules
        most_expensive_continuing_action = min(
            rules.move + rules.slippery_without_traction,
            rules.move + rules.stable_traction_surcharge,
            rules.structural_scan,
            rules.surface_scan,
            rules.traction_toggle,
            rules.retrieve,
            rules.invalid_action,
        )
        minimum_success_score = (
            rules.mission_success
            + 98 * most_expensive_continuing_action
            + rules.retrieve
        )

        self.assertEqual(minimum_success_score, -95)
        self.assertGreater(minimum_success_score, rules.stopped_early)

    def test_action_limit_applies_timeout_penalty(self) -> None:
        environment = RescueRoverEnvironment(small_scenario(), max_actions=1)
        transition = environment.step(
            Action.structural_scan(
                Position(2, 0), Position(2, 1), Position(2, 2)
            )
        )

        self.assertIs(transition.status, MissionStatus.TIMEOUT)
        self.assertIs(transition.outcome.event, Event.TIME_LIMIT)
        self.assertEqual(transition.outcome.score_delta, -102)


if __name__ == "__main__":
    unittest.main()
