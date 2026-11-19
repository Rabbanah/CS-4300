"""Automated unit tests for ParticleFilterAgent, exact consistency, repair, and baselines."""

from __future__ import annotations

import itertools
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
    POC_FLEET,
    STANDARD_FLEET,
    ShotResult,
)
from belief_agent import (
    BeliefCollapseError,
    HuntAndTargetAgent,
    ParticleFilterAgent,
    RandomAgent,
)


class TestBeliefAgent(unittest.TestCase):
    def test_exact_mode_matches_u0_hw10_poc_on_6x6(self) -> None:
        agent = ParticleFilterAgent(
            rows=6,
            cols=6,
            fleet_spec=dict(POC_FLEET),
            num_particles=100,
            exact_threshold=4000,
            seed=4300,
        )
        # U0-HW-10 PoC verified 2,472 consistent initial hypotheses on 6x6 {Cruiser:3, Destroyer:2}
        self.assertEqual(agent.mode, "EXACT_FALLBACK")
        self.assertEqual(len(agent.particles), 2472)

        # Verify exact posterior probabilities match brute-force enumeration at step 0
        probs = agent.posterior_probabilities()
        # Corner cell (0,0) vs interior cell (2,2): interior strictly higher than corner
        self.assertGreater(probs[(2, 2)], probs[(0, 0)])
        self.assertAlmostEqual(probs[(2, 2)], 0.184466, places=4)

        # Cross-check exact count after observing a hit and miss
        agent.observe(ShotResult(coord=(2, 2), hit=False, sunk_ship=None))
        agent.observe(ShotResult(coord=(0, 0), hit=True, sunk_ship=None))
        probs_after = agent.posterior_probabilities()
        # Verify against independent brute force
        ship_names = ["Cruiser", "Destroyer"]
        single = []
        for length in (3, 2):
            ps = []
            for r in range(6):
                for c in range(6 - length + 1):
                    ps.append(frozenset((r, c + i) for i in range(length)))
            for r in range(6 - length + 1):
                for c in range(6):
                    ps.append(frozenset((r + i, c) for i in range(length)))
            single.append(ps)
        brute_valid = []
        for p1, p2 in itertools.product(*single):
            if p1.isdisjoint(p2):
                occ = p1 | p2
                if (2, 2) not in occ and (0, 0) in occ:
                    brute_valid.append(occ)
        self.assertEqual(len(agent.particles), len(brute_valid))
        expected_01 = sum(1 for occ in brute_valid if (0, 1) in occ) / len(brute_valid)
        self.assertAlmostEqual(probs_after[(0, 1)], expected_01, places=9)

    def test_smc_particles_are_100_percent_consistent_and_reproducible(self) -> None:
        a1 = ParticleFilterAgent(
            rows=10,
            cols=10,
            fleet_spec=dict(STANDARD_FLEET),
            num_particles=120,
            mcmc_sweeps=2,
            seed=4300,
        )
        a2 = ParticleFilterAgent(
            rows=10,
            cols=10,
            fleet_spec=dict(STANDARD_FLEET),
            num_particles=120,
            mcmc_sweeps=2,
            seed=4300,
        )
        self.assertEqual(a1.mode, "SMC")
        self.assertEqual(len(a1.particles), 120)
        self.assertEqual(a1.particles, a2.particles)

        env = BattleshipEnv(EnvConfig(rows=10, cols=10, seed=4300, shot_budget=100))
        for _ in range(15):
            if env.status != EpisodeStatus.RUNNING:
                break
            d1 = a1.select_action()
            d2 = a2.select_action()
            self.assertEqual(d1.coord, d2.coord)
            percept = env.probe(d1.coord)
            a1.observe(percept)
            a2.observe(percept)

            # Verify every single particle in a1 strictly satisfies all constraints
            domains = a1._valid_domains()
            self.assertIsNotNone(domains)
            assert domains is not None
            domain_sets = [set(d) for d in domains]
            for particle in a1.particles:
                self.assertTrue(a1._is_particle_consistent(particle, domain_sets))

    def test_particle_deprivation_triggers_mrv_repair_and_recovers(self) -> None:
        agent = ParticleFilterAgent(
            rows=10,
            cols=10,
            fleet_spec=dict(STANDARD_FLEET),
            num_particles=60,
            mcmc_sweeps=2,
            seed=1234,
        )
        # Find a cell that happens not to be covered by a tiny synthetic particle set,
        # forcing survived_particles == 0 on a valid HIT observation.
        target_cell = (0, 0)
        target_bit = agent._coord_to_bit(target_cell)
        filtered = [
            p
            for p in agent.particles
            if not any(mask & target_bit for mask in p)
        ]
        self.assertGreater(len(filtered), 0)
        agent.particles = filtered

        metrics = agent.observe(ShotResult(coord=target_cell, hit=True, sunk_ship=None))
        self.assertEqual(metrics.survived_particles, 0)
        self.assertTrue(metrics.repair_triggered)
        self.assertTrue(metrics.rejuvenated)
        self.assertEqual(metrics.particle_count, 60)

        domains = agent._valid_domains()
        assert domains is not None
        domain_sets = [set(d) for d in domains]
        for p in agent.particles:
            self.assertTrue(agent._is_particle_consistent(p, domain_sets))

    def test_belief_collapse_on_contradictory_observations(self) -> None:
        # Case 1: Length-3 ship reported sunk after a single hit
        agent = ParticleFilterAgent(
            rows=6, cols=6, fleet_spec=dict(POC_FLEET), seed=4300
        )
        with self.assertRaises(BeliefCollapseError):
            agent.observe(ShotResult(coord=(2, 2), hit=True, sunk_ship="Cruiser"))

        # Case 2: Ship reported sunk on a MISS
        agent2 = ParticleFilterAgent(
            rows=6, cols=6, fleet_spec=dict(POC_FLEET), seed=4300
        )
        with self.assertRaises(BeliefCollapseError):
            agent2.observe(ShotResult(coord=(2, 2), hit=False, sunk_ship="Destroyer"))

        # Case 3: Unknown ship name
        agent3 = ParticleFilterAgent(
            rows=6, cols=6, fleet_spec=dict(POC_FLEET), seed=4300
        )
        with self.assertRaises(BeliefCollapseError):
            agent3.observe(ShotResult(coord=(2, 2), hit=True, sunk_ship="GhostShip"))

        # Case 4: Impossible miss pattern blocking all placements of length-3 Cruiser on 3x3
        agent4 = ParticleFilterAgent(
            rows=3, cols=3, fleet_spec={"Cruiser": 3}, seed=4300
        )
        agent4.observe(ShotResult(coord=(0, 0), hit=False))
        agent4.observe(ShotResult(coord=(1, 1), hit=False))
        with self.assertRaises(BeliefCollapseError):
            agent4.observe(ShotResult(coord=(2, 2), hit=False))

    def test_reference_baselines_complete_valid_10x10_episodes(self) -> None:
        for cls in (HuntAndTargetAgent, RandomAgent):
            with self.subTest(agent_cls=cls.__name__):
                env = BattleshipEnv(
                    EnvConfig(rows=10, cols=10, shot_budget=100, seed=4300)
                )
                agent = cls(rows=10, cols=10, fleet_spec=env.fleet_spec, seed=4300)
                while env.status == EpisodeStatus.RUNNING:
                    decision = agent.select_action()
                    percept = env.probe(decision.coord)
                    agent.observe(percept)
                self.assertEqual(env.status, EpisodeStatus.VICTORY)
                self.assertLessEqual(env.shots_count, 100)


if __name__ == "__main__":
    unittest.main()
