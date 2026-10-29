# CS-4300 Assignment 8: Bounded ReAct Sudoku Agent

Requires Python 3.10+ and uses only the Python standard library.

## Repository Contents

- [`ARCHITECTURE.md`](ARCHITECTURE.md): Module boundaries, responsibilities, public interfaces, state ownership, and allowed component interactions.
- [`SPECIFICATIONS.md`](SPECIFICATIONS.md): Formal behavioral contracts (`SPEC-SUDOKU-*`, `SPEC-UNDO-*`, `SPEC-KB-*`, `SPEC-TOOLS-*`, `SPEC-CTRL-*`, `SPEC-REACT-*`).
- [`DEVELOPMENT_PROCESS.md`](DEVELOPMENT_PROCESS.md): Documentation of the specification-first workflow, test traceability matrix, and implementation review notes.
- [`sudoku.py`](sudoku.py): 9-by-9 `SudokuBoard` with immutable given clues, mutable placements, LIFO `undo`, region inspections, and solution validation.
- [`kb.py`](kb.py) & [`proplogic/`](proplogic/README.md): `PropositionalKB` service supporting CNF conversion, transactional consistency preservation, and 3-valued DPLL refutation queries (`entailed`, `contradicted`, `unresolved`).
- [`tools.py`](tools.py): Typed tool schemas (`inspect`, `place`, `undo`, `kb_add`, `kb_query`, `submit`), strict JSON parser rejecting duplicate keys and invalid types, and structured `ToolObservation` formatting.
- [`controller.py`](controller.py): `SudokuController` and `ControllerConfig` enforcing action limits, given-clue protection, conflict policies, repeated-failure rejection, and submission validation.
- [`agent.py`](agent.py): `ReActSudokuAgent` implementing the bounded ReAct loop, observation-only prompt construction, exception sanitization, and ordered terminal condition checks.
- [`llm_client.py`](llm_client.py): `ChatClient` protocol, `OpenAICompatibleClient` (for live OpenAI-compatible course endpoints), `ScriptedClient` (for deterministic unit tests), and `RuleBasedReActClient` (autonomous offline ReAct solver).
- [`scenarios.py`](scenarios.py): 9-by-9 Sudoku puzzle scenarios (`near-complete`, `easy-1`, `branching-1`).
- [`run_experiment.py`](run_experiment.py): Command-line runner for offline and live evaluation.
- `test_sudoku.py`, `test_kb.py`, `test_tools.py`, `test_controller.py`, `test_agent.py`: Specification-derived unit test suite.

## Running the Test Suite

From the repository root (`CS-4300`):

```sh
python -m unittest discover -v -s a8 -t a8
```

Or from inside `CS-4300/a8`:

```sh
python -m unittest discover -v
```

## Running Offline Experiments

Run all scenarios offline using the deterministic `RuleBasedReActClient`:

```sh
python a8/run_experiment.py --all
```

Inspect the full step-by-step ReAct action-observation trace on a single scenario:

```sh
python a8/run_experiment.py --scenario near-complete --trace
python a8/run_experiment.py --scenario branching-1 --trace
```

## Running Against a Live OpenAI-Compatible Endpoint

No credentials or authenticated URLs are stored in the repository. To run against the course endpoint over Tailscale, set `OPENAI_API_KEY` in your shell environment and pass `--live`:

```powershell
$env:OPENAI_API_KEY = "course-placeholder"
python a8/run_experiment.py --live --model Gemma-4-26B-A4B-it-oQ4e-mtp --endpoint http://golem:8000/v1 --scenario near-complete --trace
```

