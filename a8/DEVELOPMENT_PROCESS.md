# Software Development Process & Traceability (`a8`)

This document records the five-stage specification-driven development workflow used to construct the bounded ReAct Sudoku agent.

---

## Stage 1: Module Boundaries, State Ownership, and Allowed Interactions

Before writing any functional code, we partitioned the system into six single-responsibility modules (documented in [`ARCHITECTURE.md`](ARCHITECTURE.md)):

1. **`sudoku.py` (`SudokuBoard`)**: Owns the 9x9 grid, immutable given clues (`_givens`), and LIFO mutable placement history (`_history`). Zero knowledge of LLMs, prompts, or propositional formulas.
2. **`kb.py` (`PropositionalKB` + `proplogic/`)**: Owns the CNF clause set and entry audit log. Enforces transactional consistency so contradictory additions are rejected and rolled back before they can cause classical-logic explosion (`ex falso quodlibet`).
3. **`tools.py` (`ToolCall`, `ToolObservation`, `parse_tool_call`)**: Stateless boundary layer defining typed schemas for `inspect`, `place`, `undo`, `kb_add`, `kb_query`, and `submit`. Rejects duplicate JSON keys, extra fields, boolean-as-integer coercions, and free-form prose.
4. **`controller.py` (`SudokuController`, `ControllerConfig`)**: Owns the active `SudokuBoard`, `PropositionalKB`, action budget (`max_steps`), and policy state (`consecutive_rejections`, `_last_rejected_signature`). Validates every tool call before mutating state and produces structured `ToolObservation` records.
5. **`agent.py` (`ReActSudokuAgent`)**: Runs the bounded ReAct loop, builds developer prompts strictly from `ToolObservation` views and bounded history (`history_limit`), sanitizes client exceptions, and evaluates the ordered terminal checks.
6. **`llm_client.py` (`ChatClient`, `OpenAICompatibleClient`, `ScriptedClient`, `RuleBasedReActClient`)**: Isolates model communication behind the `ChatClient` protocol with no credentials or service secrets stored in code.

---

## Stage 2: Functional Specifications (`SPECIFICATIONS.md`)

All contracts were written in [`SPECIFICATIONS.md`](SPECIFICATIONS.md) prior to implementation:
- **Puzzle contracts (`SPEC-SUDOKU-1`..`5`)**: 0-based coordinates `0..8`, box formula $3(\text{row}//3)+(\text{col}//3)$, immutable givens, peer conflict checks, and complete solution verification.
- **Undo contracts (`SPEC-UNDO-1`..`2`)**: LIFO stack recording only accepted mutable placements; multi-step undo restores prior digits and never alters given clues.
- **Knowledge-base contracts (`SPEC-KB-1`..`3`)**: `c_r{row}_c{col}_d{digit}` symbol convention, transactional consistency rollback on `add_fact`/`add_rule`, and 3-valued query semantics (`entailed`, `contradicted`, `unresolved`).
- **Tool schema contracts (`SPEC-TOOLS-1`..`3`)**: Strict JSON parsing with `_no_duplicate_keys_hook`, support for optional `"thought"` and either flat or nested `"args"`, and structured `ToolObservation` formatting.
- **Controller policy contracts (`SPEC-CTRL-1`..`6`)**: Budget decrement on every request (including malformed/rejected), immutable clue protection, immediate conflict prevention, empty-undo rejection, repeated-failure rejection, and consecutive rejection termination.
- **ReAct loop & ordered terminal checks (`SPEC-REACT-1`..`2`)**: Strict evaluation priority (`SOLVED` -> `INVALID_SUBMISSION` -> `CLIENT_FAILURE_LIMIT` -> `REJECTION_LIMIT_REACHED` -> `ACTION_LIMIT_REACHED`).

---

## Stage 3: Specification-to-Test Traceability Matrix

Black-box and supplementary white-box unit tests were derived directly from [`SPECIFICATIONS.md`](SPECIFICATIONS.md) against interface stubs before implementing the module bodies:

| Specification ID | Requirement Summary | Derived Unit Tests |
| :--- | :--- | :--- |
| `SPEC-SUDOKU-1` | 9x9 grid, `box_index`, string parsing, invalid grid/type rejection | `test_sudoku.py::test_box_index_mapping`, `test_from_string_and_grid_dimensions`, `test_from_string_accepts_dots_and_whitespace`, `test_invalid_initialization_rejected` |
| `SPEC-SUDOKU-2` | Immutable given clues vs. mutable cells | `test_sudoku.py::test_given_clues_are_immutable` |
| `SPEC-SUDOKU-3` | Cell, row, col, box, and candidate inspection | `test_sudoku.py::test_inspect_cell_row_col_box_and_candidates` |
| `SPEC-SUDOKU-4` | Placement and peer conflict detection | `test_sudoku.py::test_place_and_conflict_detection` |
| `SPEC-SUDOKU-5` | Completion and full Sudoku solution validation | `test_sudoku.py::test_completion_and_solution_validation`, `test_complete_board_with_conflict_is_not_solved` |
| `SPEC-UNDO-1..2` | LIFO undo stack, single/multi-step undo, empty history error | `test_sudoku.py::test_lifo_undo_single_and_multi_step`, `test_controller.py::test_controller_rejects_undo_when_history_empty` |
| `SPEC-KB-1` | `cell_symbol` formatting and bounds checking | `test_kb.py::test_cell_symbol_formatting_and_bounds` |
| `SPEC-KB-2` | `add_fact` unit-clause enforcement, `add_rule`, transactional consistency rollback | `test_kb.py::test_add_fact_accepts_unit_literals_and_rejects_disjunctions`, `test_add_rule_and_modus_ponens_inference`, `test_transactional_rollback_on_contradictory_fact_or_rule`, `test_invalid_syntax_rejected_without_corrupting_kb` |
| `SPEC-KB-3` | 3-valued query semantics (`entailed`, `contradicted`, `unresolved`) | `test_kb.py::test_three_valued_query_entailed_contradicted_unresolved` |
| `SPEC-TOOLS-1..3` | Typed tool schemas, duplicate JSON key rejection, strict type validation | `test_tools.py::test_parse_inspect_variants`, `test_parse_place_undo_kb_and_submit`, `test_reject_duplicate_json_keys`, `test_reject_unknown_tools_extra_fields_and_wrong_types` |
| `SPEC-CTRL-1..6` | Action budget, given protection, conflict policy, repeated failure & consecutive rejection limits | `test_controller.py::test_action_budget_decrements_on_every_request`, `test_controller_rejects_modifying_given_clue`, `test_controller_rejects_conflicting_placement`, `test_repeated_rejected_request_and_consecutive_limit`, `test_controller_kb_and_submit_validation` |
| `SPEC-REACT-1..2` | Bounded ReAct loop, ordered terminal checks, exception sanitization, end-to-end solving | `test_agent.py::test_scripted_agent_solves_near_complete_puzzle`, `test_model_text_cannot_establish_success_without_valid_submit`, `test_undo_recovery_in_react_loop`, `test_ordered_terminal_check_rejection_limit`, `test_client_exception_does_not_leak_sensitive_text`, `test_rule_based_react_client_solves_all_scenarios` |

---

## Stage 4: Coding Agent Implementation, Review, and Revisions

During implementation and review against the specification-derived test suite, two key refinements were identified and resolved:

1. **Board View Refresh on Region Inspections (`agent.py`)**:
   - *Finding*: When the agent executed `{"tool": "inspect", "target": "cell", "row": r, "col": c}`, `obs.board_view` contained the single-cell summary rather than the 9x9 ASCII grid. Overwriting `latest_board_view` with a single-cell string caused subsequent prompts to lose the full 9x9 ASCII grid view needed by the model.
   - *Fix*: Updated `ReActSudokuAgent.solve()` so `latest_board_view` is updated only when `obs.board_view` is a full 9x9 board (`place`, `undo`, `submit`, or `inspect` with `target="board"`), while region/cell inspection outputs remain in the step-by-step observation trace.
2. **Branching Scenario & Undo Verification (`scenarios.py`, `test_agent.py`)**:
   - *Finding*: An initial `branching-1` puzzle contained naked singles that allowed completion without exercising `undo` during autonomous solving, and updating `branching-1` changed cell `(0, 2)` from empty to a given clue (`4`), causing `test_undo_recovery_in_react_loop` to fail with `placements: 0 != 1`.
   - *Fix*: Constructed `branching-1` so that multi-candidate ambiguity triggers a trial placement, downstream dead-end detection (`0` legal candidates), and an explicit `undo` recovery (`undos=1`) before completing the board in 41 steps, and updated `test_undo_recovery_in_react_loop` to target mutable cell `(0, 3)`.

---

## Stage 5: Verification Summary

All 62 unit tests (`proplogic` + `test_sudoku` + `test_kb` + `test_tools` + `test_controller` + `test_agent`) pass using only the Python standard library:

```sh
python -m unittest discover -v -s a8 -t a8
```

