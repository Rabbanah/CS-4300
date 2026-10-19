"""Unit tests for LogicAgent knowledge model, search, and action conversion."""

import unittest

from agents import (
    LogicAgent,
    add_terrain_rules,
    astar_search,
    fragile_proposition,
    is_fragile,
    is_non_fragile,
    is_slippery,
    is_stable,
    record_scan_observation,
    route_to_actions,
    scan_observation_to_formulas,
    select_structural_scan_region,
    select_surface_scan_region,
    slippery_proposition,
    stable_proposition,
    valid_structural_scan_regions,
    valid_surface_scan_regions,
)
from proplogic import KnowledgeBase, atom
from rover import (
    Action,
    ActionKind,
    Percept,
    Position,
    RescueRoverEnvironment,
    ScanObservation,
    Scanner,
)
from scenarios import training_scenarios


class LogicComponentTests(unittest.TestCase):
    def test_proposition_names_are_consistent(self) -> None:
        pos = Position(2, 3)
        self.assertEqual(stable_proposition(pos), "stable_2_3")
        self.assertEqual(slippery_proposition(pos), "slippery_2_3")
        self.assertEqual(fragile_proposition(pos), "fragile_2_3")

    def test_terrain_rules_and_unknown_tile_queries(self) -> None:
        kb = KnowledgeBase()
        start = Position(0, 0)
        unknown = Position(1, 2)
        add_terrain_rules(kb, rows=3, columns=3, start=start)

        self.assertTrue(kb.is_consistent())
        self.assertTrue(is_stable(kb, start))
        self.assertTrue(is_non_fragile(kb, start))
        self.assertFalse(is_fragile(kb, start))
        self.assertFalse(is_slippery(kb, start))

        # Unknown tile: neither terrain claims nor their negations are entailed
        self.assertFalse(is_stable(kb, unknown))
        self.assertFalse(is_slippery(kb, unknown))
        self.assertFalse(is_fragile(kb, unknown))
        self.assertFalse(is_non_fragile(kb, unknown))
        self.assertFalse(kb.entails(~atom(stable_proposition(unknown))))
        self.assertFalse(kb.entails(~atom(slippery_proposition(unknown))))

    def test_negative_scans_entail_non_fragile_and_stable(self) -> None:
        kb = KnowledgeBase()
        add_terrain_rules(kb, rows=3, columns=3, start=Position(0, 0))

        struct_obs = ScanObservation(
            scanner=Scanner.STRUCTURAL,
            region=(Position(0, 0), Position(0, 1), Position(0, 2)),
            positive=False,
        )
        self.assertEqual(
            scan_observation_to_formulas(struct_obs),
            ("NOT fragile_0_0", "NOT fragile_0_1", "NOT fragile_0_2"),
        )
        record_scan_observation(kb, struct_obs)

        for col in range(3):
            pos = Position(0, col)
            self.assertTrue(is_non_fragile(kb, pos))
            self.assertFalse(is_fragile(kb, pos))

        # (0, 1) is non-fragile, but surface is still unknown until surface scan
        self.assertFalse(is_stable(kb, Position(0, 1)))
        self.assertFalse(is_slippery(kb, Position(0, 1)))

        surf_obs = ScanObservation(
            scanner=Scanner.SURFACE,
            region=(
                Position(0, 0),
                Position(0, 1),
                Position(1, 0),
                Position(1, 1),
            ),
            positive=False,
        )
        self.assertEqual(
            scan_observation_to_formulas(surf_obs),
            (
                "NOT slippery_0_0",
                "NOT slippery_0_1",
                "NOT slippery_1_0",
                "NOT slippery_1_1",
            ),
        )
        record_scan_observation(kb, surf_obs)

        # (0, 1) is now both non-fragile and non-slippery -> entailed stable
        self.assertTrue(is_stable(kb, Position(0, 1)))
        self.assertFalse(is_slippery(kb, Position(0, 1)))
        # (1, 0) is only non-slippery so far, not yet known non-fragile
        self.assertFalse(is_stable(kb, Position(1, 0)))
        self.assertFalse(is_non_fragile(kb, Position(1, 0)))

    def test_overlapping_scans_infer_fragile_and_slippery_tiles(self) -> None:
        kb = KnowledgeBase()
        add_terrain_rules(kb, rows=3, columns=3, start=Position(0, 0))

        # Negative structural scan on row 0: (0,0), (0,1), (0,2) are non-fragile
        record_scan_observation(
            kb,
            ScanObservation(
                scanner=Scanner.STRUCTURAL,
                region=(Position(0, 0), Position(0, 1), Position(0, 2)),
                positive=False,
            ),
        )
        # Negative structural scan on col 0: (0,0), (1,0), (2,0) are non-fragile
        record_scan_observation(
            kb,
            ScanObservation(
                scanner=Scanner.STRUCTURAL,
                region=(Position(0, 0), Position(1, 0), Position(2, 0)),
                positive=False,
            ),
        )
        # Positive structural scan on col 1: (0,1), (1,1), (2,1) has fragile
        pos_struct = ScanObservation(
            scanner=Scanner.STRUCTURAL,
            region=(Position(0, 1), Position(1, 1), Position(2, 1)),
            positive=True,
        )
        self.assertEqual(
            scan_observation_to_formulas(pos_struct),
            ("fragile_0_1 OR fragile_1_1 OR fragile_2_1",),
        )
        record_scan_observation(kb, pos_struct)

        # Negative structural scan on row 2: (2,0), (2,1), (2,2) are non-fragile
        record_scan_observation(
            kb,
            ScanObservation(
                scanner=Scanner.STRUCTURAL,
                region=(Position(2, 0), Position(2, 1), Position(2, 2)),
                positive=False,
            ),
        )

        # Since (0,1) and (2,1) are non-fragile, (1,1) must be entailed fragile!
        self.assertTrue(is_fragile(kb, Position(1, 1)))
        self.assertFalse(is_non_fragile(kb, Position(1, 1)))

        # Now add a negative surface scan on (1,0), (1,1), (2,0), (2,1) -> wait:
        # Suppose (0,0) is stable (start), (1,0) is non-slippery, and (1,1) is fragile (thus non-slippery).
        # Then a positive surface scan on (0,0), (0,1), (1,0), (1,1) must entail that (0,1) is slippery!
        kb.add_formula("NOT slippery_1_0")
        pos_surf = ScanObservation(
            scanner=Scanner.SURFACE,
            region=(
                Position(0, 0),
                Position(0, 1),
                Position(1, 0),
                Position(1, 1),
            ),
            positive=True,
        )
        self.assertEqual(
            scan_observation_to_formulas(pos_surf),
            ("slippery_0_0 OR slippery_0_1 OR slippery_1_0 OR slippery_1_1",),
        )
        record_scan_observation(kb, pos_surf)

        self.assertTrue(is_slippery(kb, Position(0, 1)))
        self.assertFalse(is_stable(kb, Position(0, 1)))
        self.assertTrue(is_non_fragile(kb, Position(0, 1)))

    def test_inconsistent_observation_raises_runtime_error(self) -> None:
        kb = KnowledgeBase()
        add_terrain_rules(kb, rows=3, columns=3, start=Position(0, 0))
        record_scan_observation(
            kb,
            ScanObservation(
                scanner=Scanner.STRUCTURAL,
                region=(Position(0, 0), Position(0, 1), Position(0, 2)),
                positive=False,
            ),
        )
        contradictory = ScanObservation(
            scanner=Scanner.STRUCTURAL,
            region=(Position(0, 0), Position(0, 1), Position(0, 2)),
            positive=True,
        )
        with self.assertRaises(RuntimeError):
            record_scan_observation(kb, contradictory)

    def test_valid_scan_region_counts_and_selection(self) -> None:
        structural = valid_structural_scan_regions(6, 8)
        surface = valid_surface_scan_regions(6, 8)
        self.assertEqual(len(structural), 68)
        self.assertEqual(len(surface), 35)

        chosen = select_structural_scan_region(
            rows=4,
            columns=5,
            start=Position(0, 0),
            module=Position(0, 3),
            exit_pos=Position(3, 4),
            entailed_non_fragile={Position(0, 0)},
            entailed_fragile=set(),
            scanned_structural=set(),
            positive_structural_counts={},
            need_start_to_module=True,
            need_module_to_exit=True,
        )
        self.assertIsNotNone(chosen)
        assert chosen is not None
        self.assertIn(chosen, valid_structural_scan_regions(4, 5))

        route_moves = (Position(2, 3), Position(2, 4), Position(3, 4))
        non_fragile = {
            Position(2, 2),
            Position(2, 3),
            Position(2, 4),
            Position(3, 4),
        }
        surf_chosen = select_surface_scan_region(
            rows=4,
            columns=5,
            route_moves=route_moves,
            entailed_non_fragile=non_fragile,
            entailed_stable={Position(0, 0)},
            entailed_slippery=set(),
            scanned_surface=set(),
            positive_surface_regions=(),
        )
        self.assertEqual(
            surf_chosen,
            (Position(2, 3), Position(2, 4), Position(3, 3), Position(3, 4)),
        )

    def test_astar_search_finds_route_and_prefers_stable_tiles(self) -> None:
        # 3x3 grid with fragile center (1, 1) omitted from non_fragile
        non_fragile = {
            Position(0, 0),
            Position(0, 1),
            Position(0, 2),
            Position(1, 0),
            Position(1, 2),
            Position(2, 0),
            Position(2, 1),
            Position(2, 2),
        }
        # Top path via (0,1), (0,2), (1,2) is unknown/slippery; bottom path via (1,0), (2,0), (2,1) is known stable
        stable = {
            Position(0, 0),
            Position(1, 0),
            Position(2, 0),
            Position(2, 1),
            Position(2, 2),
        }
        route = astar_search(
            Position(0, 0),
            Position(2, 2),
            non_fragile=non_fragile,
            stable=stable,
        )
        self.assertEqual(
            route,
            (
                Position(0, 0),
                Position(1, 0),
                Position(2, 0),
                Position(2, 1),
                Position(2, 2),
            ),
        )

        # Disconnected goal returns None
        unreachable = astar_search(
            Position(0, 0),
            Position(2, 2),
            non_fragile={Position(0, 0), Position(0, 1)},
        )
        self.assertIsNone(unreachable)

    def test_route_to_actions_handles_traction_retrieve_and_stop(self) -> None:
        start_to_module = (Position(0, 0), Position(0, 1), Position(0, 2))
        module_to_exit = (Position(0, 2), Position(1, 2), Position(2, 2))
        # (0, 0), (0, 2), and (2, 2) are known stable; (0, 1) and (1, 2) are not known stable
        stable_tiles = {Position(0, 0), Position(0, 2), Position(2, 2)}

        actions = route_to_actions(
            start_to_module,
            module_to_exit,
            stable_tiles=stable_tiles,
            initial_traction=False,
            carrying_module=False,
        )
        expected = (
            Action.enable_traction(),
            Action.move(Position(0, 1)),
            Action.disable_traction(),
            Action.move(Position(0, 2)),
            Action.retrieve(),
            Action.enable_traction(),
            Action.move(Position(1, 2)),
            Action.disable_traction(),
            Action.move(Position(2, 2)),
            Action.stop(),
        )
        self.assertEqual(actions, expected)
        self.assertIs(actions[-1].kind, ActionKind.STOP)
        self.assertEqual(
            sum(1 for a in actions if a.kind is ActionKind.RETRIEVE), 1
        )

        # Non-adjacent route raises ValueError
        with self.assertRaises(ValueError):
            route_to_actions((Position(0, 0), Position(0, 2)))

    def test_repeated_percept_is_processed_once(self) -> None:
        scenario = training_scenarios()[0]
        environment = RescueRoverEnvironment(scenario)
        agent = LogicAgent(4300)
        agent.reset()

        percept0 = environment.percept()
        first_action = agent.act(percept0)
        clause_count_after_first = len(agent.knowledge.clauses)

        # Replay identical percept0
        replayed_action = agent.act(percept0)
        self.assertEqual(first_action, replayed_action)
        self.assertEqual(len(agent.knowledge.clauses), clause_count_after_first)

        # Step the environment once and replay percept1
        transition = environment.step(first_action)
        percept1 = transition.percept
        next_action_1 = agent.act(percept1)
        clauses_after_step_1 = agent.knowledge.clauses
        next_action_1_replay = agent.act(percept1)

        self.assertEqual(next_action_1, next_action_1_replay)
        self.assertEqual(agent.knowledge.clauses, clauses_after_step_1)

        # Also verify during execution phase that replaying a percept returns the same move
        while not agent.planned_actions:
            act = agent.act(environment.percept())
            environment.step(act)

        exec_percept = environment.percept()
        exec_action_1 = agent.act(exec_percept)
        exec_action_2 = agent.act(exec_percept)
        self.assertEqual(exec_action_1, exec_action_2)


if __name__ == "__main__":
    unittest.main()

