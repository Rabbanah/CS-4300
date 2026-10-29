"""Unit tests for `kb.py` derived from SPECIFICATIONS.md (SPEC-KB-1..3)."""

from __future__ import annotations

import unittest

from kb import PropositionalKB, QueryStatus, cell_symbol


class TestPropositionalKBSpec(unittest.TestCase):
    """Black-box and supplementary white-box tests for PropositionalKB."""

    # --- SPEC-KB-1: Canonical Sudoku Symbols and Formula Syntax ---

    def test_cell_symbol_formatting_and_bounds(self) -> None:
        self.assertEqual(cell_symbol(0, 3, 7), "c_r0_c3_d7")
        self.assertEqual(cell_symbol(8, 8, 9), "c_r8_c8_d9")
        with self.assertRaises(ValueError):
            cell_symbol(-1, 0, 1)
        with self.assertRaises(ValueError):
            cell_symbol(0, 9, 1)
        with self.assertRaises(ValueError):
            cell_symbol(0, 0, 0)
        with self.assertRaises(ValueError):
            cell_symbol(0, 0, 10)
        with self.assertRaises(TypeError):
            cell_symbol(True, 0, 1)  # type: ignore[arg-type]

    # --- SPEC-KB-2: Fact and Rule Recording with Transactional Consistency ---

    def test_add_fact_accepts_unit_literals_and_rejects_disjunctions(self) -> None:
        kb = PropositionalKB()
        res1 = kb.add_fact("c_r0_c2_d4")
        self.assertTrue(res1.accepted)
        self.assertTrue(res1.consistent)
        self.assertEqual(res1.kind, "fact")
        self.assertIsNone(res1.error)

        res2 = kb.add_fact("~c_r0_c2_d1 & ~c_r0_c2_d2")
        self.assertTrue(res2.accepted)
        self.assertTrue(res2.consistent)

        # Disjunction is a rule, not a unit fact
        res_disj = kb.add_fact("c_r0_c2_d4 | c_r0_c2_d5")
        self.assertFalse(res_disj.accepted)
        self.assertTrue(res_disj.consistent)
        self.assertIsNotNone(res_disj.error)

    def test_add_rule_and_modus_ponens_inference(self) -> None:
        kb = PropositionalKB()
        r1 = kb.add_rule("c_r7_c8_d5 | c_r7_c8_d6")
        self.assertTrue(r1.accepted)
        self.assertTrue(r1.consistent)

        r2 = kb.add_fact("~c_r7_c8_d6")
        self.assertTrue(r2.accepted)
        self.assertTrue(r2.consistent)

        q = kb.query("c_r7_c8_d5")
        self.assertEqual(q.status, QueryStatus.ENTAILED)
        self.assertTrue(q.consistent)

    def test_transactional_rollback_on_contradictory_fact_or_rule(self) -> None:
        kb = PropositionalKB()
        kb.add_fact("c_r0_c0_d5")
        kb.add_rule("c_r0_c0_d5 -> ~c_r0_c1_d5")
        initial_clauses = kb.clause_count

        # Contradictory fact must be rejected and rolled back
        bad_fact = kb.add_fact("c_r0_c1_d5")
        self.assertFalse(bad_fact.accepted)
        self.assertFalse(bad_fact.consistent)
        self.assertIsNotNone(bad_fact.error)

        # Stored KB must remain consistent and unchanged!
        self.assertTrue(kb.is_consistent())
        self.assertEqual(kb.clause_count, initial_clauses)
        self.assertEqual(kb.query("c_r0_c1_d5").status, QueryStatus.CONTRADICTED)
        # Unrelated symbol must still be UNRESOLVED (no principle of explosion)
        self.assertEqual(kb.query("c_r8_c8_d9").status, QueryStatus.UNRESOLVED)

    def test_invalid_syntax_rejected_without_corrupting_kb(self) -> None:
        kb = PropositionalKB()
        kb.add_fact("c_r0_c0_d5")
        bad = kb.add_rule("c_r0_c0_d5 ->")
        self.assertFalse(bad.accepted)
        self.assertTrue(bad.consistent)
        self.assertIsNotNone(bad.error)
        self.assertEqual(kb.clause_count, 1)

    # --- SPEC-KB-3: Three-Valued Query Semantics ---

    def test_three_valued_query_entailed_contradicted_unresolved(self) -> None:
        kb = PropositionalKB()
        kb.add_rule("(c_r8_c6_d1 | c_r8_c6_d7) & (~c_r8_c6_d1 | ~c_r8_c6_d7)")

        # Before disambiguation, both candidates are UNRESOLVED
        self.assertEqual(kb.query("c_r8_c6_d1").status, QueryStatus.UNRESOLVED)
        self.assertEqual(kb.query("c_r8_c6_d7").status, QueryStatus.UNRESOLVED)

        # Eliminate d7 -> d1 becomes ENTAILED and d7 becomes CONTRADICTED
        kb.add_fact("~c_r8_c6_d7")
        self.assertEqual(kb.query("c_r8_c6_d1").status, QueryStatus.ENTAILED)
        self.assertEqual(kb.query("c_r8_c6_d7").status, QueryStatus.CONTRADICTED)
        self.assertEqual(kb.query("c_r1_c1_d2").status, QueryStatus.UNRESOLVED)


if __name__ == "__main__":
    unittest.main()

