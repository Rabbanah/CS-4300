"""Propositional knowledge-base service with transactional consistency and 3-valued queries."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from proplogic import Clause, KnowledgeBase, to_cnf


class QueryStatus(str, Enum):
    ENTAILED = "entailed"
    CONTRADICTED = "contradicted"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class KBEntry:
    kind: str
    formula: str
    clauses_added: int


@dataclass(frozen=True)
class KBUpdateResult:
    accepted: bool
    consistent: bool
    kind: str
    formula: str
    clause_count: int
    error: str | None = None


@dataclass(frozen=True)
class KBQueryResult:
    formula: str
    status: QueryStatus
    consistent: bool
    error: str | None = None


def cell_symbol(row: int, col: int, digit: int) -> str:
    """Return the canonical propositional symbol 'c_r{row}_c{col}_d{digit}'."""
    for name, val, low, high in (
        ("row", row, 0, 8),
        ("col", col, 0, 8),
        ("digit", digit, 1, 9),
    ):
        if isinstance(val, bool) or not isinstance(val, int):
            raise TypeError(f"{name} must be an integer")
        if val < low or val > high:
            raise ValueError(f"{name} must be in {low}..{high}, got {val}")
    return f"c_r{row}_c{col}_d{digit}"


class PropositionalKB:
    """Transactional propositional knowledge base backed by DPLL refutation."""

    def __init__(self) -> None:
        self._kb = KnowledgeBase()
        self._entries: list[KBEntry] = []

    @property
    def clause_count(self) -> int:
        """Return the number of CNF clauses currently stored."""
        return len(self._kb.clauses)

    @property
    def entries(self) -> tuple[KBEntry, ...]:
        """Return the history of accepted facts and rules."""
        return tuple(self._entries)

    def is_consistent(self) -> bool:
        """Return True iff the stored clauses are satisfiable."""
        return self._kb.is_consistent()

    def snapshot(self) -> tuple[Clause, ...]:
        """Return an immutable snapshot of stored CNF clauses."""
        return self._kb.clauses

    def restore(self, clauses: Iterable[Clause]) -> None:
        """Restore the KB to the supplied CNF clauses."""
        self._kb.clear()
        for clause in clauses:
            self._kb.add_clause(*clause)

    def clear(self) -> None:
        """Remove all clauses and entries from the KB."""
        self._kb.clear()
        self._entries.clear()

    def add_fact(self, formula: str) -> KBUpdateResult:
        """Validate and transactionally add a unit fact or conjunction of unit facts."""
        return self._add_internal("fact", formula, require_unit_clauses=True)

    def add_rule(self, formula: str) -> KBUpdateResult:
        """Validate and transactionally add a general propositional rule/formula."""
        return self._add_internal("rule", formula, require_unit_clauses=False)

    def _add_internal(
        self, kind: str, formula: str, require_unit_clauses: bool
    ) -> KBUpdateResult:
        if not isinstance(formula, str) or not formula.strip():
            return KBUpdateResult(
                accepted=False,
                consistent=self.is_consistent(),
                kind=kind,
                formula=str(formula),
                clause_count=self.clause_count,
                error="formula must be a non-empty string",
            )
        try:
            clauses = to_cnf(formula)
        except (TypeError, ValueError) as exc:
            return KBUpdateResult(
                accepted=False,
                consistent=self.is_consistent(),
                kind=kind,
                formula=formula,
                clause_count=self.clause_count,
                error=f"invalid formula syntax: {exc}",
            )

        if not clauses:
            return KBUpdateResult(
                accepted=False,
                consistent=self.is_consistent(),
                kind=kind,
                formula=formula,
                clause_count=self.clause_count,
                error="tautological formula adds no clauses",
            )

        if any(len(c) == 0 for c in clauses):
            return KBUpdateResult(
                accepted=False,
                consistent=False,
                kind=kind,
                formula=formula,
                clause_count=self.clause_count,
                error="self-contradictory formula cannot be added",
            )

        if require_unit_clauses and any(len(c) != 1 for c in clauses):
            return KBUpdateResult(
                accepted=False,
                consistent=self.is_consistent(),
                kind=kind,
                formula=formula,
                clause_count=self.clause_count,
                error="fact must be a unit literal or conjunction of unit literals; use kind='rule' for disjunctions/implications",
            )

        prior = self.snapshot()
        before_count = len(prior)
        for clause in clauses:
            self._kb.add_clause(*clause)

        if not self._kb.is_consistent():
            self.restore(prior)
            return KBUpdateResult(
                accepted=False,
                consistent=False,
                kind=kind,
                formula=formula,
                clause_count=self.clause_count,
                error="update contradicts existing knowledge base; rolled back",
            )

        added = self.clause_count - before_count
        self._entries.append(KBEntry(kind=kind, formula=formula, clauses_added=added))
        return KBUpdateResult(
            accepted=True,
            consistent=True,
            kind=kind,
            formula=formula,
            clause_count=self.clause_count,
            error=None,
        )

    def query(self, formula: str) -> KBQueryResult:
        """Return whether `formula` is ENTAILED, CONTRADICTED, or UNRESOLVED."""
        consistent = self.is_consistent()
        if not isinstance(formula, str) or not formula.strip():
            return KBQueryResult(
                formula=str(formula),
                status=QueryStatus.UNRESOLVED,
                consistent=consistent,
                error="query formula must be a non-empty string",
            )
        try:
            to_cnf(formula)
        except (TypeError, ValueError) as exc:
            return KBQueryResult(
                formula=formula,
                status=QueryStatus.UNRESOLVED,
                consistent=consistent,
                error=f"invalid query syntax: {exc}",
            )

        if not consistent:
            return KBQueryResult(
                formula=formula,
                status=QueryStatus.UNRESOLVED,
                consistent=False,
                error="knowledge base is inconsistent",
            )

        if self._kb.entails(formula):
            return KBQueryResult(
                formula=formula,
                status=QueryStatus.ENTAILED,
                consistent=True,
                error=None,
            )

        negated_formula = f"NOT ({formula})"
        if self._kb.entails(negated_formula):
            return KBQueryResult(
                formula=formula,
                status=QueryStatus.CONTRADICTED,
                consistent=True,
                error=None,
            )

        return KBQueryResult(
            formula=formula,
            status=QueryStatus.UNRESOLVED,
            consistent=True,
            error=None,
        )
