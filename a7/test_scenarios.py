"""Tests for public and generated Rescue Rover scenarios."""

import unittest
from collections import deque

from rover import Position, RoverScenario, Terrain
from scenarios import generated_scenarios, training_scenarios


def certifiable_path_exists(
    scenario: RoverScenario, start: Position, destination: Position
) -> bool:
    frontier = deque((start,))
    reached = {start}
    while frontier:
        current = frontier.popleft()
        if current == destination:
            return True
        for neighbor in (
            Position(current.row - 1, current.column),
            Position(current.row + 1, current.column),
            Position(current.row, current.column - 1),
            Position(current.row, current.column + 1),
        ):
            if (
                scenario.contains(neighbor)
                and neighbor not in reached
                and scenario.tile(neighbor) is not Terrain.FRAGILE
                and has_negative_structural_window(scenario, neighbor)
            ):
                reached.add(neighbor)
                frontier.append(neighbor)
    return False


def has_negative_structural_window(
    scenario: RoverScenario, position: Position
) -> bool:
    windows: list[tuple[Position, ...]] = []
    for start_column in range(
        max(0, position.column - 2),
        min(position.column, scenario.columns - 3) + 1,
    ):
        windows.append(
            tuple(
                Position(position.row, start_column + offset)
                for offset in range(3)
            )
        )
    for start_row in range(
        max(0, position.row - 2),
        min(position.row, scenario.rows - 3) + 1,
    ):
        windows.append(
            tuple(
                Position(start_row + offset, position.column)
                for offset in range(3)
            )
        )
    return any(
        all(scenario.tile(tile) is not Terrain.FRAGILE for tile in window)
        for window in windows
    )


class GeneratedScenarioTests(unittest.TestCase):
    def test_generation_is_reproducible_and_seeded(self) -> None:
        first = generated_scenarios(4300, 10)
        second = generated_scenarios(4300, 10)
        different = generated_scenarios(4301, 10)

        self.assertEqual(first, second)
        self.assertNotEqual(first, different)

    def test_generated_suite_is_unique_and_contains_both_hazards(self) -> None:
        scenarios = generated_scenarios(4300, 30)
        signatures = {
            (scenario.terrain, scenario.start, scenario.module, scenario.exit)
            for scenario in scenarios
        }

        self.assertEqual(len(scenarios), 30)
        self.assertEqual(len(signatures), 30)
        for scenario in scenarios:
            terrain = {tile for row in scenario.terrain for tile in row}
            self.assertIn(Terrain.FRAGILE, terrain)
            self.assertIn(Terrain.SLIPPERY, terrain)

    def test_generated_mission_has_certifiable_routes(self) -> None:
        for scenario in generated_scenarios(4300, 30):
            with self.subTest(scenario=scenario.identifier):
                self.assertTrue(
                    certifiable_path_exists(
                        scenario, scenario.start, scenario.module
                    )
                )
                self.assertTrue(
                    certifiable_path_exists(
                        scenario, scenario.module, scenario.exit
                    )
                )

    def test_invalid_generation_arguments_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            generated_scenarios(4300, 0)
        with self.assertRaises(ValueError):
            generated_scenarios(4300, 1, rows=3, columns=5)

    def test_minimum_supported_dimensions_generate_certifiable_routes(self) -> None:
        scenarios = generated_scenarios(4300, 10, rows=4, columns=5)

        self.assertEqual(len(scenarios), 10)
        for scenario in scenarios:
            self.assertTrue(
                certifiable_path_exists(
                    scenario, scenario.start, scenario.module
                )
            )
            self.assertTrue(
                certifiable_path_exists(
                    scenario, scenario.module, scenario.exit
                )
            )

    def test_training_suite_remains_available(self) -> None:
        self.assertEqual(len(training_scenarios()), 3)


if __name__ == "__main__":
    unittest.main()
