# Functional Specifications (`a8`: Bounded ReAct Sudoku Agent)

This document defines the behavioral contracts for every component in `a8`. Unit tests are derived directly from the numbered specifications below (`SPEC-*`).

---

## 1. Puzzle Representation and Operations (`sudoku.py`)

### `SPEC-SUDOKU-1`: Grid Coordinate System and Representation
- A Sudoku board is a 9-by-9 grid indexed by 0-based rows `row in 0..8` and 0-based columns `col in 0..8`.
- 3-by-3 boxes are indexed `box in 0..8` in row-major order:
  $$\text{box}(\text{row}, \text{col}) = 3 \times (\text{row} // 3) + (\text{col} // 3)$$
- Cell values are integers in `0..9`, where `0` denotes an empty cell and `1..9` denote placed digits.
- Initializing `SudokuBoard(grid)` requires a 9x9 sequence of integers in `0..9` whose non-zero initial clues contain no duplicate digits in any row, column, or 3x3 box. Violations raise `TypeError` (for non-integer/bool entries) or `ValueError` (for wrong dimensions, out-of-range digits, or conflicting initial clues).
- `SudokuBoard.from_string(puzzle_str)` parses an 81-character string (ignoring whitespace and grid separators `.` or `0` for empty cells, `1`..`9` for clues) into a `SudokuBoard`.

### `SPEC-SUDOKU-2`: Immutable Given Clues vs. Mutable Cells
- Every cell with a non-zero value at initialization is recorded as an **immutable given clue** (`is_given(row, col) == True`).
- Every cell initialized to `0` is **mutable** (`is_given(row, col) == False`).
- Attempting to `place` a digit into a given clue cell raises `ValueError("cannot modify a given clue")`.
- `givens_preserved()` returns `True` if and only if every initial given cell still holds its exact initial clue value.

### `SPEC-SUDOKU-3`: Inspection Operations
- `get_cell(row, col) -> int` returns the current value (`0..9`) at `(row, col)`. Out-of-range coordinates raise `ValueError`.
- `get_row(row) -> tuple[int, ...]` returns the 9 values in `row` from `col = 0` to `8`. Out-of-range `row` raises `ValueError`.
- `get_col(col) -> tuple[int, ...]` returns the 9 values in `col` from `row = 0` to `8`. Out-of-range `col` raises `ValueError`.
- `get_box(box) -> tuple[int, ...]` returns the 9 values in 3x3 `box` (`0..8`) in row-major order. Out-of-range `box` raises `ValueError`.
- `candidates(row, col) -> tuple[int, ...]` returns the sorted tuple of digits in `1..9` that do not conflict with any other cell in the same row, column, or 3x3 box (or `(get_cell(row, col),)` if `(row, col)` is a given clue).

### `SPEC-SUDOKU-4`: Placement and Conflict Detection
- `would_conflict(row, col, digit) -> tuple[bool, str | None]` validates `row in 0..8`, `col in 0..8`, `digit in 1..9` and checks whether any *other* cell in the same row, column, or 3x3 box already holds `digit`. Returns `(True, reason)` if a duplicate exists, or `(False, None)` otherwise.
- `place(row, col, digit, allow_conflict=False) -> PlacementRecord`:
  - Rejects `bool` values with `TypeError`.
  - Validates `row in 0..8`, `col in 0..8`, `digit in 1..9` (raising `ValueError` otherwise).
  - Raises `ValueError` if `(row, col)` is a given clue.
  - Raises `ValueError` if `get_cell(row, col) == digit` (redundant no-op placement).
  - When `allow_conflict=False` (default), raises `ValueError` if `would_conflict(row, col, digit)[0]` is `True`.
  - Mutates `_grid[row][col] = digit`, pushes `PlacementRecord(row, col, previous_digit, digit)` onto the undo history stack, and returns that record.

### `SPEC-SUDOKU-5`: Completion and Solution Validation
- `find_conflicts() -> tuple[str, ...]` returns human-readable conflict descriptions for any row, column, or 3x3 box containing duplicate non-zero digits.
- `is_complete() -> bool` returns `True` iff no cell on the board equals `0`.
- `is_solved() -> bool` returns `True` iff:
  1. `givens_preserved()` is `True`,
  2. `is_complete()` is `True`, and
  3. every row `0..8`, column `0..8`, and box `0..8` contains the exact set `{1, 2, 3, 4, 5, 6, 7, 8, 9}` (`len(find_conflicts()) == 0`).

---

## 2. Undo Behavior (`sudoku.py` & `controller.py`)

### `SPEC-UNDO-1`: LIFO Mutable History Stack
- Only accepted `place` calls on mutable cells are pushed onto the board's undo history stack.
- Rejected placements, inspections, KB operations, and submissions never modify the undo stack.

### `SPEC-UNDO-2`: Single and Multi-Step Undo Execution
- `board.undo(steps=1) -> tuple[PlacementRecord, ...]`:
  - Rejects `steps < 1` or non-integer `steps` (`ValueError` / `TypeError`).
  - Raises `ValueError("cannot undo: placement history is empty")` if the history stack has fewer than `steps` entries.
  - Pops `steps` records in LIFO order, restoring each cell `(record.row, record.col)` to `record.previous_digit`, and returns the popped records in the order they were undone.
  - Never modifies given clues.

---

## 3. Propositional Knowledge-Base Syntax and Query Semantics (`kb.py` & `proplogic/`)

### `SPEC-KB-1`: Formula Syntax and Canonical Sudoku Symbols
- Propositional symbols start with an ASCII letter or `_` followed by ASCII letters, digits, or `_`.
- Canonical Sudoku proposition helper: `cell_symbol(row, col, digit)` returns `"c_r{row}_c{col}_d{digit}"` (e.g., `c_r0_c3_d7` meaning row 0, col 3 holds digit 7) for `row in 0..8`, `col in 0..8`, `digit in 1..9`.
- Supported logical connectives (case-insensitive keywords or symbols, in decreasing precedence):
  1. Negation: `~`, `!`, `NOT`
  2. Conjunction: `&`, `AND`
  3. Disjunction: `|`, `OR`
  4. Implication (right-associative): `->`, `IMPLIES`
  5. Biconditional: `<->`, `IFF`
- Reserved words (`NOT`, `AND`, `OR`, `IMPLIES`, `IFF`) cannot be used as propositional symbols.

### `SPEC-KB-2`: Fact and Rule Recording with Transactional Consistency
- `add_fact(formula: str) -> KBUpdateResult` and `add_rule(formula: str) -> KBUpdateResult`:
  - `add_fact` requires `formula` to represent a single literal or conjunction of unit literals (every resulting CNF clause has length 1); if a non-unit clause is supplied to `add_fact`, it returns `KBUpdateResult(accepted=False, consistent=self.is_consistent(), kind="fact", formula=formula, clause_count=self.clause_count, error="fact must be a unit literal or conjunction of unit literals")`.
  - `add_rule` accepts any syntactically valid propositional formula (including implications, disjunctions, biconditionals, or unit clauses).
  - If `formula` is syntactically invalid or empty, returns `KBUpdateResult(accepted=False, consistent=self.is_consistent(), ...)` with a descriptive `error` and leaves the KB unchanged.
  - **Transactional Consistency Guarantee**: Before committing the converted CNF clauses of `formula`, `PropositionalKB` takes a snapshot of existing clauses, adds the new clauses, and checks `is_consistent()`.
    - If the resulting KB is satisfiable (`consistent == True`), the update is committed and `KBUpdateResult(accepted=True, consistent=True, ...)` is returned.
    - If the resulting KB is unsatisfiable (`consistent == False`), the snapshot is restored so the stored KB remains consistent, and `KBUpdateResult(accepted=False, consistent=False, ..., error="update contradicts existing knowledge base")` is returned.

### `SPEC-KB-3`: Three-Valued Query Semantics
- `query(formula: str) -> KBQueryResult`:
  - Evaluates a syntactically valid formula $q$ against the current consistent knowledge base $KB$ using DPLL refutation:
    1. **`QueryStatus.ENTAILED` (`"entailed"`)**: $KB \models q$ (i.e., $KB \land \neg q$ is unsatisfiable).
    2. **`QueryStatus.CONTRADICTED` (`"contradicted"`)**: $KB \models \neg q$ (i.e., $KB \land q$ is unsatisfiable).
    3. **`QueryStatus.UNRESOLVED` (`"unresolved"`)**: Neither $KB \models q$ nor $KB \models \neg q$ holds (both $KB \land q$ and $KB \land \neg q$ are satisfiable).
  - If `formula` has invalid syntax, returns `KBQueryResult(formula=formula, status=QueryStatus.UNRESOLVED, consistent=self.is_consistent(), error=...)`.

---

## 4. Typed Tool Schemas and Constrained Model Responses (`tools.py`)

### `SPEC-TOOLS-1`: Strict JSON Envelope and Duplicate Key Rejection
- Every model response must be a single JSON object (optionally enclosed in a single markdown code fence) with no duplicate keys.
- Required key: `"tool"` (string matching one of the 6 tool names).
- Optional key: `"thought"` (string recording the model's concise ReAct reasoning for the step; defaults to `""`).
- Arguments may be provided either inside an `"args": { ... }` object OR as top-level fields alongside `"tool"` and `"thought"`, provided no unknown keys exist and all field types strictly match the tool's schema.
- `bool` values are never accepted where `int` is required.

### `SPEC-TOOLS-2`: Defined Tool Schemas
1. **`inspect`** (`InspectArgs`):
   - `"target"`: `"board"` | `"cell"` | `"row"` | `"col"` | `"box"`
   - When `target == "board"`: no coordinate arguments required (`row`, `col`, `box` must be omitted or `None`).
   - When `target == "cell"`: requires `"row": int (0..8)` and `"col": int (0..8)`.
   - When `target == "row"`: requires `"row": int (0..8)`.
   - When `target == "col"`: requires `"col": int (0..8)`.
   - When `target == "box"`: requires `"box": int (0..8)`.
2. **`place`** (`PlaceArgs`):
   - Requires `"row": int (0..8)`, `"col": int (0..8)`, `"digit": int (1..9)`.
3. **`undo`** (`UndoArgs`):
   - Optional `"steps": int (>= 1, default 1)`.
4. **`kb_add`** (`KBAddArgs`):
   - Requires `"kind": "fact" | "rule"` and `"formula": str (non-empty)`.
5. **`kb_query`** (`KBQueryArgs`):
   - Requires `"formula": str (non-empty)`.
6. **`submit`** (`SubmitArgs`):
   - Takes no additional arguments.

### `SPEC-TOOLS-3`: Structured Tool Observations (`ToolObservation`)
- Every step produces an immutable `ToolObservation` containing:
  - `tool: str` (tool name or `"invalid_request"`),
  - `accepted: bool` (whether the controller accepted and executed the request),
  - `remaining_budget: int` (actions remaining after this step),
  - `board_view: str | None` (formatted board or cell/row/col/box details for `inspect`, `place`, `undo`, `submit`),
  - `kb_consistent: bool | None` (whether the KB is consistent after `kb_add` / `kb_query`),
  - `query_status: str | None` (`"entailed"`, `"contradicted"`, or `"unresolved"` for `kb_query`),
  - `rejection_reason: str | None` (populated whenever `accepted == False`),
  - `terminal: bool` (whether this step triggered a terminal condition),
  - `terminal_reason: str | None` (populated when `terminal == True`).

---

## 5. Action Limits and Controller Policies (`controller.py`)

### `SPEC-CTRL-1`: Action Budget Accounting
- `ControllerConfig.max_steps` must be a positive integer (`>= 1`).
- Every call to `step_raw()` or `execute()` (whether accepted, policy-rejected, or malformed) increments `steps_used` by 1 and reduces `remaining_budget = max(0, max_steps - steps_used)`.
- If `remaining_budget == 0` prior to a step, the controller rejects the request with `terminal=True` and `terminal_reason="action limit reached"`.

### `SPEC-CTRL-2`: Immutable Clue Policy
- Any `place` call targeting a given clue cell (`board.is_given(row, col) == True`) is rejected (`accepted=False`, `rejection_reason="cannot modify given clue at (r, c)"`), leaving the board unchanged.

### `SPEC-CTRL-3`: Immediate Conflict Policy
- When `config.forbid_immediate_conflicts == True` (default), any `place(row, col, digit)` that conflicts with an existing digit in the same row, column, or 3x3 box (or places the same digit already in `(row, col)`) is rejected by the controller with `accepted=False` and a descriptive `rejection_reason`.

### `SPEC-CTRL-4`: Empty Undo Policy
- Any `undo(steps)` call when the board's placement stack has fewer than `steps` entries is rejected (`accepted=False`, `rejection_reason="cannot undo: insufficient placement history"`).

### `SPEC-CTRL-5`: Repeated Rejected Request Policy
- When `config.reject_repeated_failures == True` (default), if the agent issues the exact same tool call immediately after that call was rejected, the controller rejects it with `rejection_reason` prefixed by `"repeated rejected request: ..."` and increments `consecutive_rejections`.

### `SPEC-CTRL-6`: Consecutive Rejection Limit Policy
- Every malformed or policy-rejected step increments `consecutive_rejections`. Every accepted step resets `consecutive_rejections = 0`.
- When `consecutive_rejections >= config.max_consecutive_rejections`, the observation marks `terminal=True` and `terminal_reason="maximum consecutive rejections reached"`.

---

## 6. Bounded ReAct Loop and Ordered Terminal Checks (`agent.py`)

### `SPEC-REACT-1`: Observation-Only Prompt Construction
- At each step, `ReActSudokuAgent` constructs the model prompt using only:
  1. System instructions defining the Sudoku rules and typed JSON tool schemas,
  2. Initial board given clues view (or latest `ToolObservation` views), and
  3. Up to `history_limit` recent `(ToolCall, ToolObservation)` trace entries (`history_limit >= 0`).
- If `client.chat(messages)` raises an exception, the agent catches it, records only `type(e).__name__` (never raw exception text that could contain credentials/URLs), counts a client failure, and evaluates terminal checks.

### `SPEC-REACT-2`: Ordered Terminal Checks
After each step (and before starting step 1), terminal conditions are checked in this exact priority order:
1. **`SOLVED`**: The executed tool is `submit`, `accepted == True`, and `board.is_solved() == True`. The loop terminates immediately with `TerminalStatus.SOLVED`.
2. **`INVALID_SUBMISSION`** (when `config.stop_on_invalid_submit == True`): The executed tool is `submit` and `board.is_solved() == False`. Terminates with `TerminalStatus.INVALID_SUBMISSION`. (When `stop_on_invalid_submit == False`, an invalid `submit` is treated as a rejected request so the agent can continue fixing the board).
3. **`CLIENT_FAILURE_LIMIT`**: Consecutive or total client exceptions reach `max_consecutive_rejections`. Terminates with `TerminalStatus.CLIENT_FAILURE_LIMIT`.
4. **`REJECTION_LIMIT_REACHED`**: `controller.consecutive_rejections >= config.max_consecutive_rejections`. Terminates with `TerminalStatus.REJECTION_LIMIT_REACHED`.
5. **`ACTION_LIMIT_REACHED`**: `controller.remaining_budget == 0` (and status is not `SOLVED`). Terminates with `TerminalStatus.ACTION_LIMIT_REACHED`.

