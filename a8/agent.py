"""Bounded ReAct Sudoku agent and ordered terminal condition checks."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum

from controller import ControllerConfig, SudokuController
from kb import PropositionalKB
from llm_client import ChatClient
from sudoku import SudokuBoard
from tools import ToolCall, ToolName, ToolObservation, tool_schema_prompt


class TerminalStatus(str, Enum):
    SOLVED = "SOLVED"
    INVALID_SUBMISSION = "INVALID_SUBMISSION"
    ACTION_LIMIT_REACHED = "ACTION_LIMIT_REACHED"
    REJECTION_LIMIT_REACHED = "REJECTION_LIMIT_REACHED"
    CLIENT_FAILURE_LIMIT = "CLIENT_FAILURE_LIMIT"


@dataclass(frozen=True)
class TraceStep:
    step: int
    raw_response: str
    parsed_call: ToolCall | None
    observation: ToolObservation


@dataclass(frozen=True)
class RunMetrics:
    steps_taken: int = 0
    model_calls: int = 0
    accepted_actions: int = 0
    rejected_actions: int = 0
    malformed_requests: int = 0
    placements: int = 0
    undos: int = 0
    kb_adds: int = 0
    kb_queries: int = 0
    inspections: int = 0
    submissions: int = 0
    client_failures: int = 0
    runtime_seconds: float = 0.0


@dataclass(frozen=True)
class AgentRunResult:
    terminal_status: TerminalStatus
    terminal_reason: str
    final_board: tuple[tuple[int, ...], ...]
    solved: bool
    metrics: RunMetrics
    trace: tuple[TraceStep, ...]


class ReActSudokuAgent:
    """Bounded ReAct agent that coordinates a ChatClient and SudokuController."""

    def __init__(
        self,
        client: ChatClient,
        config: ControllerConfig | None = None,
        history_limit: int = 12,
    ) -> None:
        if isinstance(history_limit, bool) or not isinstance(history_limit, int) or history_limit < 0:
            raise ValueError("history_limit must be an integer >= 0")
        self.client = client
        self.config = config if config is not None else ControllerConfig()
        self.history_limit = history_limit

    def _build_messages(
        self,
        initial_board_view: str,
        latest_board_view: str,
        remaining_budget: int,
        trace: list[TraceStep],
    ) -> list[dict[str, str]]:
        system_content = (
            "You are a bounded ReAct agent solving a 9-by-9 Sudoku puzzle.\n"
            "Preserve all initial given clues and fill every row (0..8), column (0..8), "
            "and 3-by-3 box (0..8) with digits 1..9 without duplicates.\n"
            "You must interact with the environment strictly through typed tool calls.\n\n"
            + tool_schema_prompt()
        )

        user_lines = [
            f"Remaining Action Budget: {remaining_budget}",
            "Initial Given Board:",
            initial_board_view,
            "\nCurrent Observed Board:",
            latest_board_view,
        ]

        if self.history_limit > 0 and trace:
            user_lines.append("\nRecent Tool Call & Observation Trace:")
            for entry in trace[-self.history_limit :]:
                call_desc = (
                    entry.parsed_call.signature()
                    if entry.parsed_call is not None
                    else entry.raw_response[:60]
                )
                user_lines.append(
                    f"  Step {entry.step}: request={call_desc} -> {entry.observation.format_for_prompt()}"
                )

        user_lines.append("\nSelect your next single JSON tool call:")
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": "\n".join(user_lines)},
        ]

    def _evaluate_ordered_terminal_checks(
        self,
        controller: SudokuController,
        last_call: ToolCall | None,
        last_obs: ToolObservation,
        consecutive_client_failures: int,
    ) -> tuple[TerminalStatus, str] | None:
        """Evaluate terminal conditions in the strict order specified by SPEC-REACT-2."""
        # 1. SOLVED: valid submission verified by application state
        if (
            last_call is not None
            and last_call.tool == ToolName.SUBMIT
            and last_obs.accepted
            and controller.submission_accepted
            and controller.board.is_solved()
        ):
            return TerminalStatus.SOLVED, "Submitted board is complete and satisfies all Sudoku constraints."

        # 2. INVALID_SUBMISSION (when configured to halt on invalid submit)
        if (
            last_call is not None
            and last_call.tool == ToolName.SUBMIT
            and not last_obs.accepted
            and self.config.stop_on_invalid_submit
        ):
            return (
                TerminalStatus.INVALID_SUBMISSION,
                last_obs.rejection_reason or "Submitted board failed validation.",
            )

        # 3. CLIENT_FAILURE_LIMIT
        if consecutive_client_failures >= self.config.max_consecutive_rejections:
            return (
                TerminalStatus.CLIENT_FAILURE_LIMIT,
                f"Reached consecutive client failure limit ({consecutive_client_failures}).",
            )

        # 4. REJECTION_LIMIT_REACHED
        if controller.consecutive_rejections >= self.config.max_consecutive_rejections:
            return (
                TerminalStatus.REJECTION_LIMIT_REACHED,
                f"Reached consecutive controller rejection limit ({controller.consecutive_rejections}).",
            )

        # 5. ACTION_LIMIT_REACHED
        if controller.remaining_budget <= 0:
            return (
                TerminalStatus.ACTION_LIMIT_REACHED,
                f"Action budget of {self.config.max_steps} steps exhausted without a valid submission.",
            )

        return None

    def solve(self, board: SudokuBoard) -> AgentRunResult:
        """Run the bounded ReAct loop on a fresh copy of `board` until a terminal condition holds."""
        started = time.perf_counter()
        working_board = SudokuBoard(board.to_grid())
        kb = PropositionalKB()
        controller = SudokuController(working_board, kb=kb, config=self.config)

        initial_board_view = working_board.format_board()
        latest_board_view = initial_board_view
        trace: list[TraceStep] = []

        model_calls = 0
        client_failures = 0
        consecutive_client_failures = 0

        terminal_status = TerminalStatus.ACTION_LIMIT_REACHED
        terminal_reason = "Action limit reached."

        for step_num in range(1, self.config.max_steps + 1):
            messages = self._build_messages(
                initial_board_view=initial_board_view,
                latest_board_view=latest_board_view,
                remaining_budget=controller.remaining_budget,
                trace=trace,
            )

            model_calls += 1
            try:
                raw_response = self.client.chat(messages)
                consecutive_client_failures = 0
            except Exception as exc:
                client_failures += 1
                consecutive_client_failures += 1
                controller.steps_used += 1
                exc_type = type(exc).__name__
                obs = ToolObservation(
                    tool="client_call",
                    accepted=False,
                    remaining_budget=controller.remaining_budget,
                    message=f"Client call failed with {exc_type}.",
                    rejection_reason=f"client_error: {exc_type}",
                    terminal=(
                        consecutive_client_failures >= self.config.max_consecutive_rejections
                        or controller.remaining_budget == 0
                    ),
                    terminal_reason=(
                        f"client failure: {exc_type}"
                        if consecutive_client_failures >= self.config.max_consecutive_rejections
                        else ("action limit reached" if controller.remaining_budget == 0 else None)
                    ),
                )
                trace.append(
                    TraceStep(
                        step=step_num,
                        raw_response=f"<client_exception:{exc_type}>",
                        parsed_call=None,
                        observation=obs,
                    )
                )
                term = self._evaluate_ordered_terminal_checks(
                    controller, None, obs, consecutive_client_failures
                )
                if term is not None:
                    terminal_status, terminal_reason = term
                    break
                continue

            parsed_call, obs = controller.step_raw(raw_response)
            if obs.board_view is not None and parsed_call is not None:
                if parsed_call.tool in (ToolName.PLACE, ToolName.UNDO, ToolName.SUBMIT):
                    latest_board_view = obs.board_view
                elif (
                    parsed_call.tool == ToolName.INSPECT
                    and getattr(parsed_call.args, "target", None) == "board"
                ):
                    latest_board_view = obs.board_view

            trace.append(
                TraceStep(
                    step=step_num,
                    raw_response=raw_response,
                    parsed_call=parsed_call,
                    observation=obs,
                )
            )

            term = self._evaluate_ordered_terminal_checks(
                controller, parsed_call, obs, consecutive_client_failures
            )
            if term is not None:
                terminal_status, terminal_reason = term
                break

        accepted_actions = sum(1 for t in trace if t.observation.accepted)
        rejected_actions = sum(1 for t in trace if not t.observation.accepted)
        metrics = RunMetrics(
            steps_taken=len(trace),
            model_calls=model_calls,
            accepted_actions=accepted_actions,
            rejected_actions=rejected_actions,
            malformed_requests=controller.malformed_requests,
            placements=controller.accepted_placements,
            undos=controller.accepted_undos,
            kb_adds=controller.kb_updates,
            kb_queries=controller.kb_queries,
            inspections=controller.inspections,
            submissions=controller.submissions,
            client_failures=client_failures,
            runtime_seconds=time.perf_counter() - started,
        )

        return AgentRunResult(
            terminal_status=terminal_status,
            terminal_reason=terminal_reason,
            final_board=working_board.to_grid(),
            solved=(terminal_status == TerminalStatus.SOLVED),
            metrics=metrics,
            trace=tuple(trace),
        )
