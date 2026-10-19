# U0-HW-07 Starter: Rescue Rover

Use Python 3.10 or newer. This starter uses only the Python standard library.

## Files

- `rover.py`: provided environment, actions, percepts, outcomes, and score
  rules; do not modify it.
- `scenarios.py`: provided training maps and seeded scenario generator; do not
  modify it.
- `proplogic/`: importable propositional-logic package, CLI, package README,
  and package tests; do not modify its implementation.
- `agents.py`: agent protocol, two provided baselines, and required
  `LogicAgent` scaffold.
- `run_experiment.py`: episode runner, trace printer, and training summaries.
- `evaluate_agents.py`: generated-suite evaluator, CSV writer, and statistical
  summaries.
- `AGENT_WORKFLOW.md`: suggested development and evaluation workflow for a
  logic-based rover agent.
- `SCORING.md`: complete score and metric definitions.
- `test_rover.py`: tests for environment behavior and score accounting.
- `test_logic_agent.py`: required public behavior for `LogicAgent`.
- `test_scenarios.py`: tests for reproducible scenario generation.
- `test_evaluate_agents.py`: tests for paired evaluation and CSV output.
- `test_runner.py`: tests for the agent-environment framework.

## Rescue Rover Task Environment

An earthquake has damaged a remote research station. A rescue rover must cross
the station's maintenance floor, retrieve a data module, carry it to the exit,
and stop there safely. The station map identifies the floor layout and the
module and exit positions, but damage beneath the surface has made the condition
of most tiles unknown.

The mapped floor is a rectangular grid. The rover starts on a tile known to be
stable. Each tile has exactly one fixed terrain category:

- **Stable:** supports the rover and does not require traction.
- **Slippery:** supports the rover, but entering it without traction causes a
  costly slip.
- **Fragile:** collapses when entered and ends the episode.

The rover knows the grid dimensions, its current position, the module and exit
positions, whether traction is enabled, whether it carries the module, its
action count, and the outcome of its preceding action. It does not receive the
terrain map.

The rover can move one tile horizontally or vertically, enable or disable
traction, scan a mapped region remotely, retrieve the module, or stop. A
structural scan covers exactly three consecutive horizontal or vertical tiles
and reports whether at least one is fragile. A surface scan covers exactly one
2-by-2 square and reports whether at least one is slippery. A positive result
does not identify which tile or how many tiles caused it.

Traction allows the rover to cross slippery tiles, but using it on stable ground
reduces efficiency. Every move costs three points. Entering a stable tile with
traction enabled costs three additional points, while entering a slippery tile
with traction adds no surcharge. Entering a slippery tile without traction adds
a 100-point penalty, moves the rover onto the tile, and allows the episode to
continue. Enabling or disabling traction costs one point.

The environment ends an episode when the rover completes the mission, enters a
fragile tile, stops early, or reaches 100 actions. Read `SCORING.md` for the
complete action costs, outcome scores, and reported metrics.

## Environment Interface

An agent implements two methods:

```python
class MyAgent:
    def reset(self) -> None:
        ...

    def act(self, percept: Percept) -> Action:
        ...
```

The percept exposes the mapped dimensions, current position, known module and
exit positions, traction state, carrying state, action count, and the outcome
of the preceding action. It never exposes the scenario's terrain map. A scan
result appears in `percept.last_outcome.scan` after a valid scan action.

Actions use constructors such as:

```python
Action.move(Position(1, 2))
Action.enable_traction()
Action.disable_traction()
Action.structural_scan(Position(0, 0), Position(0, 1), Position(0, 2))
Action.surface_scan(
    Position(0, 0), Position(0, 1), Position(1, 0), Position(1, 1)
)
Action.retrieve()
Action.stop()
```

Structural scan regions must be three consecutive horizontal or vertical
tiles. Surface scan regions must be the four tiles of a 2x2 square. Region
order does not affect validity or the scanner result.

The public training terrain can be inspected while debugging the framework. An
agent implementation may not import `scenarios`, access `RoverScenario.terrain`,
or access `RescueRoverEnvironment._scenario`. Grading uses separate hidden
terrain maps. The only terrain evidence available to an agent at run time is
the known stable start tile and the scan observations in its percept history.

## Using `proplogic` In `LogicAgent`

Import the package's public names in `agents.py`. Create a fresh
`KnowledgeBase` in `LogicAgent.reset()` so observations from one episode cannot
affect another episode.

```python
from proplogic import KnowledgeBase


class LogicAgent:
    def reset(self) -> None:
        self.knowledge = KnowledgeBase()
        # Reset the remaining episode-specific state.
```

On the first percept, add the terrain constraints and known start fact. On each
later call to `act`, translate any unprocessed scan observation into formulas,
clauses, or facts before selecting an action. Use `KnowledgeBase.entails()` to
derive terrain conclusions.

`LogicAgent` must complete two sequential phases. Before its first movement, it
must scan and infer a non-fragile route from the start to the module and from the
module to the exit. It must then use graph search over the inferred non-fragile
tiles, convert the route or routes into movement, traction, retrieval, and
stopping actions, and execute that sequence without further scans. Keep
knowledge translation, graph search, and action execution in separate parts of
the implementation.

Read `proplogic/README.md` for the complete Python API, formula language, CNF
semantics, entailment behavior, errors, performance guidance, interactive CLI,
batch CLI, and package-test command.

## Running the Starter

Confirm that the supplied infrastructure passes:

```sh
python3 -m unittest -v \
  proplogic.tests.test_core proplogic.tests.test_cli \
  test_rover test_scenarios test_runner test_evaluate_agents
```

`test_logic_agent.py` intentionally fails in the untouched starter because
`LogicAgent.act` raises `NotImplementedError`. These failures are the public
implementation targets. They also check that all scan actions precede the first
movement action. Student-authored tests, code review, and hidden grading check
that knowledge-base entailment and graph search produce the plan. Run the file
while developing the agent:

```sh
python3 -m unittest -v test_logic_agent.py
```

Run the complete suite after each substantial change. It should pass when the
required agent is complete:

```sh
python3 -m unittest -v
```

Inspect one seeded random-agent trace or summarize all training scenarios:

```sh
python3 run_experiment.py random --scenario 0 --seed 4300
python3 run_experiment.py random --seed 4300
```

## Adding and Selecting an Agent

Use `AGENT_WORKFLOW.md` for the suggested design, testing, debugging, and
evaluation sequence.

Complete the provided `LogicAgent` in `agents.py`. Additional variations do not
need to inherit from a base class, but each must implement the `RoverAgent`
protocol:

```python
class MyAgent:
    def __init__(self, seed: int) -> None:
        self.seed = seed

    def reset(self) -> None:
        # Clear all state retained from a previous episode.
        pass

    def act(self, percept: Percept) -> Action:
        # Update internal state and return one Action.
        return Action.stop()
```

`LogicAgent` is already registered as `logic`. To add another variation, import
the class in `run_experiment.py` and add one entry to `FACTORIES`:

```python
from agents import LogicAgent, MyAgent, RandomAgent, RoverAgent, StopAgent

FACTORIES: dict[str, AgentFactory] = {
    "logic": lambda seed: LogicAgent(seed),
    "random": lambda seed: RandomAgent(seed),
    "stop": lambda seed: StopAgent(),
    "my-agent": lambda seed: MyAgent(seed),
}
```

The dictionary key becomes the command-line agent name. Run one scenario with
a complete action trace:

```sh
python3 run_experiment.py my-agent --scenario 0 --seed 4300
```

Omit `--scenario` to run that agent on every public training scenario and print
aggregate results:

```sh
python3 run_experiment.py my-agent --seed 4300
```

Use the same scenarios and seeds when comparing agent variations. The factory
receives the command-line seed, so stochastic agents should use it rather than
global random state. Deterministic agents may accept and ignore it.

After implementation, run the reproducible generated-suite evaluation:

```sh
python3 evaluate_agents.py random logic \
  --scenario-seed 4300 \
  --scenarios 30 \
  --agent-seeds 30 \
  --seed-start 4300 \
  --output results.csv
```

The evaluator requires a clean Git working tree, writes one CSV row per paired
run, and prints status counts, score and action summaries, scenario-level
success intervals, and paired score differences. The first named agent is the
baseline, so this command reports the logic-agent score difference relative to
random. This is a reproducible development benchmark for the generator's
terrain distribution. Grading may use an independent suite or a different
scenario seed. Grading scenarios retain the supplied generator's 6-by-8 map
dimensions and route-certification guarantee.

A complete fallback remains within the 100-action limit: a 6-by-8 map has 68
structural-scan regions, the two certified shortest route legs require at most
23 moves, and one traction action, retrieval, and stopping bring the maximum to
94 actions. Surface scans may improve traction efficiency, but the complete
agent must stay within 100 actions.

## Behavioral Rules

- Retrieval succeeds only at the module position.
- Stopping succeeds only while carrying the module at the exit position.
- The environment applies physical outcomes from the hidden map. It does not
  inspect an agent's knowledge base.
- Agent code may depend only on percepts, action outcomes, and the provided
  inference interface. It may not inspect scenario terrain or hard-code routes.

See `SCORING.md` for action costs, outcome scores, and reported metrics.
