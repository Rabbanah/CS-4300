# U0-HW-02 Starter: Agents in the Delivery Hallway

Use Python 3.10 or newer. This starter uses only the Python standard library.

## Files

- `hallway.py`: provided environment; do not modify it.
- `agents.py`: provided random agent and the two agents you must implement.
- `experiment.py`: episode runner, trace printer, and repeated experiment.
- `test_hallway.py`: basic tests for the provided code.

## Setup

Create a private GitHub repository for the assignment and place these files at
its root. Invite GitHub user `fractal13` as a collaborator with at least read
access. Commit and push your work before submitting your report.

First, confirm that the provided environment and random baseline pass their
tests:

```sh
python3 -m unittest -v test_hallway.py
```

The tests in `test_agents.py` describe minimum behavior for your two agents.
They fail against the incomplete starter. After implementing both agents, run
the full suite with `python3 -m unittest -v`.

Run the random baseline over 100 episodes or inspect one seeded trace:

```sh
python3 experiment.py random --episodes 100
python3 experiment.py random --trace-seed 7
```

After implementing both agents in `agents.py`, replace `random` with `reflex`
or `model`. Use the same seeds for all agents so that they face the same 100
initial configurations.

## Constraints

- Do not change `hallway.py` or the public interfaces in the starter files.
- `SimpleReflexAgent` may use only the current `Percept`; it may not retain
  information between calls to `act`.
- `ModelBasedReflexAgent` must retain and update internal state derived from its
  percept history and the known effects of hallway actions.
- When the current percept permits a successful pickup or delivery, both agents
  must take that action.
- Both implementations must return one of the provided `Action` values.
- Do not add third-party dependencies.

Your repository may include additional tests and documentation. Do not include
secrets, access tokens, virtual environments, or generated cache directories.
