"""Automated unit tests for BattleshipEnv, config validation, and terminal rules."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from battleship_env import (
    BattleshipEnv,
    EnvConfig,
    EpisodeStatus,
    InvalidActionError,
    InvalidConfigError,
    POC_FLEET,
    STANDARD_FLEET,
)


class TestBattleshipEnv(unittest.TestCase):
    def test_standard_fleet_generation_is_valid_and_deterministic(self) -> None:
        cfg = EnvConfig(rows=10, cols=10, fleet_spec=dict(STANDARD_FLEET), seed=4300)
        env1 = BattleshipEnv(cfg)
        env2 = BattleshipEnv(cfg)

        self.assertEqual(env1.total_ship_cells, 17)
        self.assertEqual(env1.status, EpisodeStatus.RUNNING)
        self.assertEqual(
            env1.inspect_state(reveal_ships=True)["ground_truth_ships"],
            env2.inspect_state(reveal_ships=True)["ground_truth_ships"],
        )

        # Verify distinct seeds produce distinct layouts
        env3 = BattleshipEnv(EnvConfig(seed=4301))
        self.assertNotEqual(
            env1.inspect_state(reveal_ships=True)["ground_truth_ships"],
            env3.inspect_state(reveal_ships=True)["ground_truth_ships"],
        )

    def test_inspect_state_hides_ships_by_default(self) -> None:
        env = BattleshipEnv(EnvConfig(seed=4300))
        hidden = env.inspect_state(reveal_ships=False)
        self.assertNotIn("ground_truth_ships", hidden)
        revealed = env.inspect_state(reveal_ships=True)
        self.assertIn("ground_truth_ships", revealed)

    def test_probe_hit_miss_sunk_and_victory_on_fixed_board(self) -> None:
        fixed = {
            "Cruiser": [(0, 0), (0, 1), (0, 2)],
            "Destroyer": [(2, 2), (3, 2)],
        }
        env = BattleshipEnv(
            EnvConfig(
                rows=6,
                cols=6,
                fleet_spec=dict(POC_FLEET),
                shot_budget=10,
                fixed_ships=fixed,
            )
        )
        self.assertEqual(len(env.available_actions()), 36)

        miss = env.probe((5, 5))
        self.assertFalse(miss.hit)
        self.assertIsNone(miss.sunk_ship)
        self.assertEqual(len(env.available_actions()), 35)

        h1 = env.probe((2, 2))
        self.assertTrue(h1.hit)
        self.assertIsNone(h1.sunk_ship)

        h2 = env.probe((3, 2))
        self.assertTrue(h2.hit)
        self.assertEqual(h2.sunk_ship, "Destroyer")
        self.assertEqual(env.status, EpisodeStatus.RUNNING)

        env.probe((0, 0))
        env.probe((0, 1))
        final = env.probe((0, 2))
        self.assertTrue(final.hit)
        self.assertEqual(final.sunk_ship, "Cruiser")
        self.assertTrue(env.is_won)
        self.assertEqual(env.status, EpisodeStatus.VICTORY)
        self.assertEqual(env.remaining_ship_cells, 0)

    def test_timeout_when_budget_exhausted(self) -> None:
        fixed = {"Destroyer": [(0, 0), (0, 1)]}
        env = BattleshipEnv(
            EnvConfig(
                rows=4,
                cols=4,
                fleet_spec={"Destroyer": 2},
                shot_budget=2,
                fixed_ships=fixed,
            )
        )
        env.probe((3, 3))
        self.assertEqual(env.status, EpisodeStatus.RUNNING)
        env.probe((3, 2))
        self.assertEqual(env.status, EpisodeStatus.TIMEOUT)
        self.assertEqual(env.budget_remaining, 0)
        with self.assertRaises(InvalidActionError):
            env.probe((0, 0))

    def test_victory_allowed_on_exact_last_budget_shot(self) -> None:
        fixed = {"Destroyer": [(0, 0), (0, 1)]}
        env = BattleshipEnv(
            EnvConfig(
                rows=4,
                cols=4,
                fleet_spec={"Destroyer": 2},
                shot_budget=2,
                fixed_ships=fixed,
            )
        )
        env.probe((0, 0))
        env.probe((0, 1))
        self.assertEqual(env.status, EpisodeStatus.VICTORY)

    def test_invalid_actions_rejected(self) -> None:
        env = BattleshipEnv(EnvConfig(rows=6, cols=6, fleet_spec=dict(POC_FLEET)))
        with self.assertRaises(InvalidActionError):
            env.probe((-1, 0))
        with self.assertRaises(InvalidActionError):
            env.probe((6, 0))
        with self.assertRaises(InvalidActionError):
            env.probe((0, 6))
        env.probe((1, 1))
        with self.assertRaises(InvalidActionError):
            env.probe((1, 1))

    def test_unsupported_configs_rejected(self) -> None:
        bad_configs = [
            EnvConfig(rows=0, cols=10),
            EnvConfig(rows=25, cols=10),
            EnvConfig(rows=4, cols=4, fleet_spec=dict(STANDARD_FLEET)),
            EnvConfig(rows=10, cols=10, fleet_spec={}),
            EnvConfig(rows=10, cols=10, shot_budget=0),
            EnvConfig(
                rows=5,
                cols=5,
                fleet_spec={"A": 2, "B": 2},
                fixed_ships={"A": [(0, 0), (0, 1)], "B": [(0, 1), (1, 1)]},
            ),
            EnvConfig(
                rows=5,
                cols=5,
                fleet_spec={"A": 3},
                fixed_ships={"A": [(0, 0), (0, 2), (0, 3)]},
            ),
        ]
        for cfg in bad_configs:
            with self.subTest(cfg=cfg):
                with self.assertRaises(InvalidConfigError):
                    BattleshipEnv(cfg)


if __name__ == "__main__":
    unittest.main()
