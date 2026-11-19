"""Integration tests for the end-to-end Battleship vertical slice and baseline harness."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from baseline_cases import run_all_baselines
from battleship_env import EnvConfig, EpisodeStatus, POC_FLEET, STANDARD_FLEET, ShotResult
from belief_agent import ActionDecision, BeliefUpdateMetrics, CandidateAction
from runner import run_episode


class _OutOfBoundsAgent:
    name = "bad_oob_agent"

    def inspect_belief(self) -> dict[str, object]:
        return {"agent": self.name, "mode": "BAD"}

    def select_action(self) -> ActionDecision:
        return ActionDecision(
            coord=(99, 99),
            probability=1.0,
            cell_entropy_bits=0.0,
            decision_time_ms=0.0,
            top_candidates=[CandidateAction((99, 99), 1.0, 0.0)],
            belief_snapshot=self.inspect_belief(),
        )

    def observe(self, result: ShotResult) -> BeliefUpdateMetrics:
        raise AssertionError("Should never observe after invalid action.")


class TestVerticalSliceRunner(unittest.TestCase):
    def test_standard_10x10_episode_reaches_victory_with_full_trace(self) -> None:
        cfg = EnvConfig(
            rows=10,
            cols=10,
            fleet_spec=dict(STANDARD_FLEET),
            shot_budget=75,
            seed=4300,
        )
        res = run_episode(
            config=cfg,
            agent_type="smc",
            case_id="test_10x10",
            num_particles=120,
            mcmc_sweeps=2,
            agent_seed=4300,
        )
        self.assertEqual(res.status, EpisodeStatus.VICTORY)
        self.assertTrue(res.success)
        self.assertEqual(res.hits, 17)
        self.assertLessEqual(res.shots_taken, 75)
        self.assertEqual(len(res.steps), res.shots_taken)
        self.assertIsNotNone(res.exact_fallback_step)

        # Verify trace inspectability schema
        first = res.steps[0].to_dict()
        for key in (
            "step",
            "available_actions_count",
            "belief_before",
            "top_candidates",
            "chosen_action",
            "chosen_prob",
            "percept",
            "belief_after",
            "env_effect",
        ):
            self.assertIn(key, first)
        self.assertEqual(first["available_actions_count"], 100)
        self.assertGreaterEqual(len(first["top_candidates"]), 1)

    def test_timeout_outcome_handled_explicitly(self) -> None:
        cfg = EnvConfig(
            rows=10,
            cols=10,
            fleet_spec=dict(STANDARD_FLEET),
            shot_budget=10,
            seed=4302,
        )
        res = run_episode(config=cfg, agent_type="smc", num_particles=80, agent_seed=4300)
        self.assertEqual(res.status, EpisodeStatus.TIMEOUT)
        self.assertFalse(res.success)
        self.assertEqual(res.shots_taken, 10)
        self.assertEqual(len(res.steps), 10)

    def test_invalid_config_handled_explicitly(self) -> None:
        cfg = EnvConfig(
            rows=4,
            cols=4,
            fleet_spec=dict(STANDARD_FLEET),
            shot_budget=16,
            seed=4300,
        )
        res = run_episode(config=cfg, agent_type="smc")
        self.assertEqual(res.status, EpisodeStatus.INVALID_CONFIG)
        self.assertFalse(res.success)
        self.assertEqual(res.shots_taken, 0)
        self.assertIsNotNone(res.error_message)
        assert res.error_message is not None
        self.assertIn("Carrier", res.error_message)

    def test_belief_collapse_handled_explicitly(self) -> None:
        cfg = EnvConfig(
            rows=6,
            cols=6,
            fleet_spec=dict(POC_FLEET),
            shot_budget=36,
            seed=4300,
        )

        def corrupt_hook(step: int, percept: ShotResult) -> ShotResult:
            if step == 1:
                return ShotResult(coord=percept.coord, hit=True, sunk_ship="Cruiser")
            return percept

        res = run_episode(
            config=cfg,
            agent_type="smc",
            percept_hook=corrupt_hook,
            agent_seed=4300,
        )
        self.assertEqual(res.status, EpisodeStatus.BELIEF_COLLAPSE)
        self.assertFalse(res.success)
        self.assertEqual(res.shots_taken, 1)
        self.assertIsNotNone(res.error_message)

    def test_invalid_action_handled_explicitly(self) -> None:
        cfg = EnvConfig(
            rows=6,
            cols=6,
            fleet_spec=dict(POC_FLEET),
            shot_budget=10,
            seed=4300,
        )
        res = run_episode(config=cfg, agent_override=_OutOfBoundsAgent())
        self.assertEqual(res.status, EpisodeStatus.INVALID_ACTION)
        self.assertFalse(res.success)
        self.assertEqual(res.shots_taken, 0)
        self.assertIsNotNone(res.error_message)
        assert res.error_message is not None
        self.assertIn("out of bounds", res.error_message)

    def test_retained_baseline_suite_statuses(self) -> None:
        payload = run_all_baselines(save_artifacts=False)
        rows = {row["case_id"]: row for row in payload["retained_cases"]}
        self.assertEqual(rows["case_01_standard_seed4300"]["status"], "VICTORY")
        self.assertEqual(rows["case_02_corner_clustered_boundary"]["status"], "VICTORY")
        self.assertEqual(rows["case_03_tight_budget_timeout"]["status"], "TIMEOUT")
        self.assertEqual(rows["case_04_unsupported_config"]["status"], "INVALID_CONFIG")
        self.assertEqual(
            rows["case_05_corrupted_sensor_collapse"]["status"], "BELIEF_COLLAPSE"
        )


if __name__ == "__main__":
    unittest.main()
