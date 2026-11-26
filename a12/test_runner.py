"""Integration tests for the U0-HW-12 runner and repeatable evaluation harness."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from battleship_env import EnvConfig, EpisodeStatus, POC_FLEET, STANDARD_FLEET, ShotResult
from belief_agent import ActionDecision, BeliefUpdateMetrics, CandidateAction
from evaluation import run_evaluation
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


class TestEvaluationRunner(unittest.TestCase):
    def test_standard_10x10_episode_reaches_victory_for_baseline_and_refined(self) -> None:
        cfg = EnvConfig(
            rows=10,
            cols=10,
            fleet_spec=dict(STANDARD_FLEET),
            shot_budget=75,
            seed=4300,
        )
        for ag in ("smc_baseline", "smc_refined", "smc_info_gain"):
            with self.subTest(agent_type=ag):
                res = run_episode(
                    config=cfg,
                    agent_type=ag,
                    case_id=f"test_10x10_{ag}",
                    num_particles=120,
                    mcmc_sweeps=3,
                    agent_seed=4300,
                )
                self.assertEqual(res.status, EpisodeStatus.VICTORY)
                self.assertTrue(res.success)
                self.assertEqual(res.hits, 17)
                self.assertLessEqual(res.shots_taken, 75)
                self.assertEqual(len(res.steps), res.shots_taken)
                self.assertIsNotNone(res.exact_fallback_step)
                first = res.steps[0].to_dict()
                for key in (
                    "step",
                    "available_actions_count",
                    "belief_before",
                    "top_candidates",
                    "chosen_action",
                    "chosen_prob",
                    "chosen_expected_info_gain_bits",
                    "percept",
                    "belief_after",
                    "env_effect",
                ):
                    self.assertIn(key, first)

    def test_timeout_outcome_handled_explicitly(self) -> None:
        cfg = EnvConfig(
            rows=10,
            cols=10,
            fleet_spec=dict(STANDARD_FLEET),
            shot_budget=10,
            seed=4302,
        )
        res = run_episode(
            config=cfg, agent_type="smc_info_gain", num_particles=80, agent_seed=4300
        )
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
        res = run_episode(config=cfg, agent_type="smc_info_gain")
        self.assertEqual(res.status, EpisodeStatus.INVALID_CONFIG)
        self.assertFalse(res.success)
        self.assertEqual(res.shots_taken, 0)
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
            agent_type="smc_info_gain",
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
        assert res.error_message is not None
        self.assertIn("out of bounds", res.error_message)

    def test_evaluation_harness_smoke_all_10_retained_cases(self) -> None:
        payload = run_evaluation(
            num_seeds=2, seed_start=4300, num_particles=80, save_artifacts=False
        )
        rows = {
            (r["case_id"], r["variant"]): r for r in payload["retained_cases"]
        }
        self.assertEqual(rows[("eval_01_standard_seed4300", "smc_info_gain")]["status"], "VICTORY")
        self.assertEqual(rows[("eval_04_corner_clustered_boundary", "smc_info_gain")]["status"], "VICTORY")
        self.assertEqual(rows[("eval_05_packed_block_difficult", "smc_info_gain")]["status"], "VICTORY")
        self.assertEqual(rows[("eval_07_tight_budget_timeout", "smc_info_gain")]["status"], "TIMEOUT")
        self.assertEqual(rows[("eval_08_unsupported_grid_config", "smc_info_gain")]["status"], "INVALID_CONFIG")
        self.assertEqual(rows[("eval_09_unsupported_budget_config", "smc_info_gain")]["status"], "INVALID_CONFIG")
        self.assertEqual(rows[("eval_10_corrupted_sensor_collapse", "smc_info_gain")]["status"], "BELIEF_COLLAPSE")


if __name__ == "__main__":
    unittest.main()
