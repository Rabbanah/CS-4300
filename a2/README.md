# U0-HW-03 Starter: Planning Cloud Service Recovery

Use Python 3.10 or newer. This starter uses only the Python standard library.

## Files

- `recovery.py`: provided state definitions and execution environment; do not
  modify it.
- `scenarios.py`: provided deterministic 50-scenario suite; do not modify it.
- `problem.py`: incomplete problem formulation you must implement.
- `agents.py`: provided interfaces and random baseline plus the UCS and A*
  agents you must implement.
- `run_experiment.py`: provided plan validator, trace printer, and experiment
  runner; do not modify it.
- `test_recovery.py`: passing tests for provided resources.
- `test_interfaces.py`: minimum tests that pass after your implementation is
  correct.
- `test_student.py`: incomplete required tests you must write.

Use the separately supplied `U0-HW-03-report-template.md` for your analysis.

## Setup

Create a private GitHub repository for the assignment and place these files at
its root. Invite GitHub user `fractal13` as a collaborator with at least read
access. Commit and push your work before submitting your report.

First, confirm that the provided environment, scenarios, and random baseline
pass their tests:

```sh
python3 -m unittest -v test_recovery.py
```

The complete suite intentionally fails against the incomplete starter. As you
implement the problem, agents, heuristic, and your own tests, rerun:

```sh
python3 -m unittest -v
```

## Running Experiments

Inspect one traced A* run:

```sh
python3 run_experiment.py --agent astar --scenario 7 --trace
```

Run all three agents on all 50 required scenarios:

```sh
python3 run_experiment.py --all --scenarios 0:50
```

You can replace `astar` with `ucs` or `random`, and you can replace `7` with any
scenario identifier from 0 through 49. The full experiment uses the same
initial states for every agent and prints the required aggregate table and
paired UCS/A* cost check.

## Student Work

Complete all methods that raise `NotImplementedError` in `problem.py` and
`agents.py`. Replace every placeholder failure in `test_student.py` with a real
test. Follow the action definitions and metrics requirements in the assignment
description.

`RecoveryProblem` is a pure planning model. Do not call
`ServiceRecoveryEnvironment.step`, `_legal_actions`, `_apply_legal_action`, or
another live-environment operation from `problem.py` or while searching. The
runner executes the returned plan in a separate environment to check it.

## Constraints

- Do not modify `recovery.py`, `scenarios.py`, `run_experiment.py`, or supplied
  tests.
- Preserve all public names and method signatures.
- Use `RecoveryState` directly as the graph-search key.
- Preserve `Action` enumeration order when generating successors.
- Use deterministic insertion-order tie-breaking in both priority frontiers.
- Return an empty tuple when the initial state is a goal.
- Return `None` when the frontier becomes empty without finding a goal.
- Count generated, expanded, and peak-frontier metrics exactly as defined in the
  assignment.
- Do not add third-party dependencies.
- Do not hard-code scenario identifiers, complete states, or stored plans.
- Do not include secrets, access tokens, virtual environments, or generated
  cache directories.

Your repository may include additional tests and documentation.
