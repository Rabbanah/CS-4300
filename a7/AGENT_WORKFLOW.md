# Building A Logic-Based Rover Agent

Use this workflow to develop an agent incrementally. Refer to `README.md` for
the environment and agent interfaces, `SCORING.md` for performance measures,
and `proplogic/README.md` for logic syntax and API details.

## Project Boundaries

Complete `LogicAgent` in `agents.py`. You may add student-authored agent modules
and tests. Register any additional agent variation in `run_experiment.py` as
described in `README.md`.

Do not modify:

- `rover.py`;
- `scenarios.py`;
- the implementation under `proplogic/`;
- supplied tests;
- supplied score or metric definitions; or
- supplied baseline-agent behavior.

Agent decisions may depend only on percepts, action outcomes, and conclusions
from the public inference interface. Use the supplied `Action`, `Percept`, and
`Position` APIs to interact with the environment. Agent code may not import
`scenarios`, access `RoverScenario.terrain`, access
`RescueRoverEnvironment._scenario`, infer terrain from scenario identifiers, or
hard-code routes or terrain for supplied maps. Grading uses independent terrain
maps in addition to public tests.

## 1. Design The Logic

Define a consistent proposition name for each terrain claim, such as
`stable_2_3`, `slippery_2_3`, and `fragile_2_3`.

For every mapped tile, encode exactly one terrain category with these formulas:

1. At least one of `stable`, `slippery`, and `fragile` is true.
2. `stable` and `slippery` cannot both be true.
3. `stable` and `fragile` cannot both be true.
4. `slippery` and `fragile` cannot both be true.

Add the fact that the starting tile is stable. Translate scan outcomes as
follows:

- a positive structural scan adds one disjunction of the three scanned
  `fragile` propositions;
- a negative structural scan adds a negative `fragile` fact for each scanned
  tile;
- a positive surface scan adds one disjunction of the four scanned `slippery`
  propositions; and
- a negative surface scan adds a negative `slippery` fact for each scanned
  tile.

Keep terrain inference separate from graph search and action execution. The
knowledge base answers questions about tiles. Before the first movement action,
the discovery policy must scan until entailed non-fragile tiles contain a route
from the start to the module and a route from the module to the exit. The agent
must then use A* or another graph-search algorithm over that inferred subgraph
and convert the resulting route or routes into movement, traction, retrieval,
and stopping actions. Execute the planned sequence without further scans.

## 2. Test The Logic With The CLI

Open the formula tester:

```sh
python3 -m proplogic
```

Use `tell` to enter terrain rules and representative scan observations. Use
`ask` to check the conclusions your agent will rely on:

```text
logic> tell fragile_a OR fragile_b OR fragile_c
logic> tell NOT fragile_b AND NOT fragile_c
logic> ask fragile_a
entailed
```

Test positive, negative, and unknown cases. For an unknown tile, neither a
claim nor its negation should be entailed. Use `consistent` after adding each
kind of rule. Use `cnf FORMULA` when the generated clauses are unclear.

## 3. Implement The Knowledge Model

Create small functions in your agent code for:

- generating proposition names from `Position` values;
- adding the terrain rules for every mapped tile;
- translating each `ScanObservation` into formulas;
- asking whether a tile is non-fragile, stable, or slippery; and
- selecting valid structural and surface scan regions.

Also create separate functions for graph search and for converting a route into
mission actions. Function names are student-defined, but the interfaces should
be testable without running a full generated suite.

Initialize a new `KnowledgeBase` for every episode. Add the map-wide rules and
start fact after receiving the first percept. On each `act` call, process the
preceding outcome in `percept.last_outcome` once before choosing the next
action. Tracking `percept.action_count` prevents duplicate processing when a
percept is replayed in a test.

Check `kb.is_consistent()` after every update during development. An agent must
not select an action from an inconsistent knowledge base because classical
logic would then entail every query. Treat inconsistency as an implementation
error and stop the test or raise an exception with the relevant observation.

Write unit tests for these functions before connecting them to navigation.
Include overlapping-scan inference, negative scans, unknown tiles,
knowledge-base consistency, `reset()` behavior, repeated percept handling,
terrain queries used by search and traction decisions, graph search on a small
inferred map, route-to-action conversion, retrieval and stopping preconditions,
and seeded reproducibility.

## 4. Create The Agent

Complete the provided `LogicAgent` class in `agents.py`. It implements the
`RoverAgent` protocol and initializes a fresh `KnowledgeBase` in `reset()`:

```python
class LogicAgent:
    def __init__(self, seed: int) -> None:
        self.seed = seed
        self.reset()

    def reset(self) -> None:
        # Recreate episode-specific knowledge and planning state.
        ...

    def act(self, percept: Percept) -> Action:
        # Update knowledge, choose a justified action, and return it.
        ...
```

Build in stages:

1. initialize the terrain knowledge base;
2. issue valid scans and record their results;
3. continue discovery until inferred non-fragile tiles contain both required
   route legs;
4. use graph search to plan the start-to-module and module-to-exit routes;
5. convert the routes into movement, traction, retrieval, and stopping actions;
6. execute the complete sequence without further scans; and
7. reduce scans, moves, and traction use after the agent is reliable.

## 5. Register Additional Variations

The provided `LogicAgent` is already registered under `logic`. Import and
register each additional variation in `run_experiment.py`:

```python
from agents import LogicAgent, RandomAgent, RoverAgent, StopAgent

FACTORIES: dict[str, AgentFactory] = {
    "random": lambda seed: RandomAgent(seed),
    "stop": lambda seed: StopAgent(),
    "logic": lambda seed: LogicAgent(seed),
}
```

The key `logic` becomes the command-line agent name.

## 6. Debug One Episode

The untouched starter intentionally fails `test_logic_agent.py` because
`LogicAgent.act` is unfinished. Run its public behavior tests while completing
the stages above:

```sh
python3 -m unittest -v test_logic_agent.py
```

The tests require valid actions, clean reset behavior, successful training and
generated missions, zero unprotected slips, zero invalid actions, and seeded
reproducibility. They also require every scan to precede the first movement.
Student-authored tests, code review, and hidden grading verify that entailed
non-fragile tiles and graph search produce the plan. Run the complete suite
after each substantial change. It should pass before evaluation:

```sh
python3 -m unittest -v
```

Then inspect one complete trace:

```sh
python3 run_experiment.py logic --scenario 0 --seed 4300
```

Check each step for:

- correct interpretation of the preceding scan;
- consistent use of inferred terrain by the route and action policies;
- valid retrieval and stopping actions; and
- unprotected slips, unnecessary scans, moves, traction-enabled moves, traction
  toggles, or invalid actions.

Repeat traced runs on every public training scenario before collecting summary
results.

## 7. Measure Performance Reproducibly

The three public maps support debugging. `evaluate_agents.py` creates a larger
suite from a fixed scenario seed, runs every agent on identical scenario/agent
seed pairs, writes raw CSV data, and prints summaries:

```sh
python3 evaluate_agents.py random logic \
  --scenario-seed 4300 \
  --scenarios 30 \
  --agent-seeds 30 \
  --seed-start 4300 \
  --output results.csv
```

The evaluator requires a clean Git working tree and records its commit in every
CSV row. Commit the agent before evaluation. Use `--allow-dirty` only for quick
development checks. The first named agent is the paired-comparison baseline, so
this ordering reports the `logic` score difference relative to `random`.

The default protocol creates 30 distinct 6x8 scenarios and uses 30 agent seeds
per scenario. The generator guarantees a non-fragile route through the module
to the exit while varying locations and terrain. Each route tile belongs to at
least one negative structural scan window, so the route can be established from
scanner evidence. The same suite seed reproduces the same maps exactly.

The 100-action limit supports a simple complete fallback for these generated
maps. There are 68 valid structural-scan regions. Scanning all of them exposes
the generator's certified route through entailed non-fragile tiles. The two
shortest route legs require at most 23 moves, and one traction action, retrieval,
and stopping produce a maximum of 94 actions. Surface scans are optional for
efficiency and must fit within the same action limit. Independent grading suites
retain the 6x8 dimensions and route-certification guarantee but may use another
scenario seed.

The summary includes mission-status counts, success rate, mean and median
score, action and unprotected-slip measures, descriptive scenario-level
bootstrap intervals, and paired score differences. Seed repetitions are
averaged within each scenario before the across-scenario intervals are
calculated. Failures remain in every denominator. Preserve the command and
`results.csv` with the report. Results describe the generated terrain
distribution; instructor grading may use an independent suite or a different
scenario seed.

Commit `results.csv` after the final evaluation. Before another evaluation that
requires a clean tree, commit or remove the previous output file.
