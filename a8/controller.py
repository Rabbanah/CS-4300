"""Controller policies, request validation, and tool execution."""

from __future__ import annotations

from dataclasses import dataclass

from kb import PropositionalKB, cell_symbol
from sudoku import SudokuBoard, box_index
from tools import (
    InspectArgs,
    KBAddArgs,
    KBQueryArgs,
    PlaceArgs,
    SubmitArgs,
    ToolCall,
    ToolName,
    ToolObservation,
    ToolParseError,
    UndoArgs,
    parse_tool_call,
)


@dataclass(frozen=True)
class ControllerConfig:
    max_steps: int = 100
    forbid_immediate_conflicts: bool = True
    reject_repeated_failures: bool = True
    max_consecutive_rejections: int = 10
    stop_on_invalid_submit: bool = False
    sync_kb_on_place: bool = False

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_steps, bool)
            or not isinstance(self.max_steps, int)
            or self.max_steps < 1
        ):
            raise ValueError("max_steps must be an integer >= 1")
        if (
            isinstance(self.max_consecutive_rejections, bool)
            or not isinstance(self.max_consecutive_rejections, int)
            or self.max_consecutive_rejections < 1
        ):
            raise ValueError("max_consecutive_rejections must be an integer >= 1")


class SudokuController:
    """Validates tool requests against controller policies and executes accepted tools."""

    def __init__(
        self,
        board: SudokuBoard,
        kb: PropositionalKB | None = None,
        config: ControllerConfig | None = None,
    ) -> None:
        self.board = board
        self.kb = kb if kb is not None else PropositionalKB()
        self.config = config if config is not None else ControllerConfig()

        self.steps_used: int = 0
        self.consecutive_rejections: int = 0
        self.rejected_requests: int = 0
        self.malformed_requests: int = 0
        self.accepted_placements: int = 0
        self.accepted_undos: int = 0
        self.kb_updates: int = 0
        self.kb_queries: int = 0
        self.inspections: int = 0
        self.submissions: int = 0
        self.submitted: bool = False
        self.submission_accepted: bool = False
        self._last_rejected_signature: str | None = None

    @property
    def remaining_budget(self) -> int:
        """Return remaining action budget (max(0, max_steps - steps_used))."""
        return max(0, self.config.max_steps - self.steps_used)

    def _record_rejection(
        self,
        tool: str,
        reason: str,
        signature: str | None = None,
        kb_consistent: bool | None = None,
        force_terminal_reason: str | None = None,
    ) -> ToolObservation:
        self.rejected_requests += 1
        self.consecutive_rejections += 1
        self._last_rejected_signature = signature

        terminal = False
        terminal_reason: str | None = None
        if force_terminal_reason is not None:
            terminal = True
            terminal_reason = force_terminal_reason
        elif self.consecutive_rejections >= self.config.max_consecutive_rejections:
            terminal = True
            terminal_reason = (
                f"maximum consecutive rejections reached ({self.consecutive_rejections})"
            )
        elif self.remaining_budget == 0:
            terminal = True
            terminal_reason = "action limit reached"

        return ToolObservation(
            tool=tool,
            accepted=False,
            remaining_budget=self.remaining_budget,
            message=f"Rejected: {reason}",
            kb_consistent=kb_consistent,
            rejection_reason=reason,
            terminal=terminal,
            terminal_reason=terminal_reason,
        )

    def _record_acceptance(
        self,
        tool: str,
        message: str,
        board_view: str | None = None,
        kb_consistent: bool | None = None,
        query_status: str | None = None,
        solved_terminal: bool = False,
    ) -> ToolObservation:
        self.consecutive_rejections = 0
        self._last_rejected_signature = None

        terminal = False
        terminal_reason: str | None = None
        if solved_terminal:
            terminal = True
            terminal_reason = "valid solution submitted"
        elif self.remaining_budget == 0:
            terminal = True
            terminal_reason = "action limit reached"

        return ToolObservation(
            tool=tool,
            accepted=True,
            remaining_budget=self.remaining_budget,
            message=message,
            board_view=board_view,
            kb_consistent=kb_consistent,
            query_status=query_status,
            rejection_reason=None,
            terminal=terminal,
            terminal_reason=terminal_reason,
        )

    def step_raw(self, raw_response: str) -> tuple[ToolCall | None, ToolObservation]:
        """Parse raw model text, validate against policies, and execute if valid."""
        if self.remaining_budget <= 0:
            return None, ToolObservation(
                tool="none",
                accepted=False,
                remaining_budget=0,
                message="Action limit already reached.",
                rejection_reason="action limit reached",
                terminal=True,
                terminal_reason="action limit reached",
            )

        try:
            call = parse_tool_call(raw_response)
        except ToolParseError as exc:
            self.steps_used += 1
            self.malformed_requests += 1
            obs = self._record_rejection(
                tool="malformed",
                reason=f"malformed tool call: {exc}",
                signature=f"RAW:{raw_response.strip()[:80]}",
            )
            return None, obs

        return call, self.execute(call)

    def execute(self, call: ToolCall) -> ToolObservation:
        """Validate and execute a parsed ToolCall within the action budget."""
        if self.remaining_budget <= 0:
            return ToolObservation(
                tool=call.tool.value,
                accepted=False,
                remaining_budget=0,
                message="Action limit already reached.",
                rejection_reason="action limit reached",
                terminal=True,
                terminal_reason="action limit reached",
            )

        self.steps_used += 1
        sig = call.signature()

        if (
            self.config.reject_repeated_failures
            and self._last_rejected_signature is not None
            and sig == self._last_rejected_signature
        ):
            return self._record_rejection(
                tool=call.tool.value,
                reason=f"repeated rejected request: {sig}",
                signature=sig,
            )

        if call.tool == ToolName.INSPECT:
            assert isinstance(call.args, InspectArgs)
            return self._handle_inspect(call.args)
        if call.tool == ToolName.PLACE:
            assert isinstance(call.args, PlaceArgs)
            return self._handle_place(call.args, sig)
        if call.tool == ToolName.UNDO:
            assert isinstance(call.args, UndoArgs)
            return self._handle_undo(call.args, sig)
        if call.tool == ToolName.KB_ADD:
            assert isinstance(call.args, KBAddArgs)
            return self._handle_kb_add(call.args, sig)
        if call.tool == ToolName.KB_QUERY:
            assert isinstance(call.args, KBQueryArgs)
            return self._handle_kb_query(call.args, sig)
        if call.tool == ToolName.SUBMIT:
            assert isinstance(call.args, SubmitArgs)
            return self._handle_submit(sig)

        raise AssertionError(f"Unhandled tool {call.tool}")

    def _handle_inspect(self, args: InspectArgs) -> ToolObservation:
        self.inspections += 1
        if args.target == "board":
            view = self.board.format_board()
            return self._record_acceptance(
                tool=ToolName.INSPECT.value,
                message="Inspected full 9x9 board.",
                board_view=view,
            )
        if args.target == "cell":
            assert args.row is not None and args.col is not None
            val = self.board.get_cell(args.row, args.col)
            given = self.board.is_given(args.row, args.col)
            cands = self.board.candidates(args.row, args.col)
            b = box_index(args.row, args.col)
            view = (
                f"cell=({args.row}, {args.col}) box={b} value={val} "
                f"given={given} legal_candidates={list(cands)}"
            )
            return self._record_acceptance(
                tool=ToolName.INSPECT.value,
                message=f"Inspected cell ({args.row}, {args.col}).",
                board_view=view,
            )
        if args.target == "row":
            assert args.row is not None
            vals = self.board.get_row(args.row)
            missing = [d for d in range(1, 10) if d not in vals]
            view = f"row={args.row} values={list(vals)} missing={missing}"
            return self._record_acceptance(
                tool=ToolName.INSPECT.value,
                message=f"Inspected row {args.row}.",
                board_view=view,
            )
        if args.target == "col":
            assert args.col is not None
            vals = self.board.get_col(args.col)
            missing = [d for d in range(1, 10) if d not in vals]
            view = f"col={args.col} values={list(vals)} missing={missing}"
            return self._record_acceptance(
                tool=ToolName.INSPECT.value,
                message=f"Inspected col {args.col}.",
                board_view=view,
            )
        if args.target == "box":
            assert args.box is not None
            vals = self.board.get_box(args.box)
            missing = [d for d in range(1, 10) if d not in vals]
            view = f"box={args.box} values={list(vals)} missing={missing}"
            return self._record_acceptance(
                tool=ToolName.INSPECT.value,
                message=f"Inspected box {args.box}.",
                board_view=view,
            )
        raise AssertionError(f"Invalid inspect target: {args.target}")

    def _handle_place(self, args: PlaceArgs, sig: str) -> ToolObservation:
        if self.board.is_given(args.row, args.col):
            return self._record_rejection(
                tool=ToolName.PLACE.value,
                reason=f"cannot modify given clue at ({args.row}, {args.col})",
                signature=sig,
            )
        if self.board.get_cell(args.row, args.col) == args.digit:
            return self._record_rejection(
                tool=ToolName.PLACE.value,
                reason=f"cell ({args.row}, {args.col}) already holds digit {args.digit}",
                signature=sig,
            )
        if self.config.forbid_immediate_conflicts:
            conflict, reason = self.board.would_conflict(args.row, args.col, args.digit)
            if conflict:
                return self._record_rejection(
                    tool=ToolName.PLACE.value,
                    reason=f"Sudoku constraint violation: {reason}",
                    signature=sig,
                )

        rec = self.board.place(
            args.row,
            args.col,
            args.digit,
            allow_conflict=not self.config.forbid_immediate_conflicts,
        )
        self.accepted_placements += 1
        if self.config.sync_kb_on_place and rec.previous_digit == 0:
            self.kb.add_fact(cell_symbol(args.row, args.col, args.digit))

        return self._record_acceptance(
            tool=ToolName.PLACE.value,
            message=(
                f"Placed digit {args.digit} at ({args.row}, {args.col}) "
                f"(previous={rec.previous_digit})."
            ),
            board_view=self.board.format_board(),
        )

    def _handle_undo(self, args: UndoArgs, sig: str) -> ToolObservation:
        if len(self.board.history) < args.steps:
            return self._record_rejection(
                tool=ToolName.UNDO.value,
                reason=(
                    f"cannot undo {args.steps} step(s): placement history has "
                    f"{len(self.board.history)} entry/entries"
                ),
                signature=sig,
            )
        undone = self.board.undo(args.steps)
        self.accepted_undos += 1
        desc = ", ".join(
            f"({r.row},{r.col}): {r.new_digit}->{r.previous_digit}" for r in undone
        )
        return self._record_acceptance(
            tool=ToolName.UNDO.value,
            message=f"Undid {len(undone)} placement(s): {desc}.",
            board_view=self.board.format_board(),
        )

    def _handle_kb_add(self, args: KBAddArgs, sig: str) -> ToolObservation:
        if args.kind == "fact":
            res = self.kb.add_fact(args.formula)
        else:
            res = self.kb.add_rule(args.formula)

        if not res.accepted:
            return self._record_rejection(
                tool=ToolName.KB_ADD.value,
                reason=f"KB {args.kind} rejected: {res.error}",
                signature=sig,
                kb_consistent=res.consistent,
            )

        self.kb_updates += 1
        return self._record_acceptance(
            tool=ToolName.KB_ADD.value,
            message=(
                f"Added {args.kind} {args.formula!r} to KB "
                f"(total clauses={res.clause_count})."
            ),
            kb_consistent=res.consistent,
        )

    def _handle_kb_query(self, args: KBQueryArgs, sig: str) -> ToolObservation:
        res = self.kb.query(args.formula)
        if res.error is not None:
            return self._record_rejection(
                tool=ToolName.KB_QUERY.value,
                reason=f"KB query rejected: {res.error}",
                signature=sig,
                kb_consistent=res.consistent,
            )
        self.kb_queries += 1
        return self._record_acceptance(
            tool=ToolName.KB_QUERY.value,
            message=f"Query {args.formula!r} -> {res.status.value}.",
            kb_consistent=res.consistent,
            query_status=res.status.value,
        )

    def _handle_submit(self, sig: str) -> ToolObservation:
        self.submissions += 1
        self.submitted = True

        if not self.board.givens_preserved():
            reason = "submission invalid: given clues were not preserved"
        elif not self.board.is_complete():
            empty_count = sum(
                1
                for r in range(9)
                for c in range(9)
                if self.board.get_cell(r, c) == 0
            )
            reason = f"submission invalid: board has {empty_count} empty cell(s)"
        else:
            conflicts = self.board.find_conflicts()
            if conflicts:
                reason = f"submission invalid: conflicts present ({conflicts[0]})"
            elif not self.board.is_solved():
                reason = "submission invalid: Sudoku constraints not satisfied"
            else:
                self.submission_accepted = True
                return self._record_acceptance(
                    tool=ToolName.SUBMIT.value,
                    message="Sudoku solution validated and accepted!",
                    board_view=self.board.format_board(),
                    solved_terminal=True,
                )

        force_term = reason if self.config.stop_on_invalid_submit else None
        return self._record_rejection(
            tool=ToolName.SUBMIT.value,
            reason=reason,
            signature=sig,
            force_terminal_reason=force_term,
        )
