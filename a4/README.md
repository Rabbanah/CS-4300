# Jetan Adversarial-Search Environment Prototype

This Python 3.10+ prototype uses only the standard library. It does not require
a project-specific virtual environment. It separates the game state and rules,
agent policies, match execution, and display code so a search agent can reason
over copied game states without changing the live game.

## Architecture

- `jetan.py` defines the immutable `JetanBoard`, pieces, and moves.
- `agents.py` is the public aggregation point for agent classes and interfaces.
- `agent_protocol.py` defines the common `JetanAgent` interface.
- `agent_factory.py` contains the command-line agent registry.
- `agent_builders.py` supplies one-call registration helpers for each required
  agent modality.
- `random_agent.py` and `terminal_agent.py` contain the supplied baseline agents.
- `minimax_agent.py` supplies exhaustive depth-limited minimax and search metrics.
- `evaluations.py` defines the interface for student evaluation functions.
- `llm_client.py` supplies the course model client and deterministic test client.
- `llm_evaluation.py` manages model calls, caching, budgets, and fallback for an
  LLM cutoff evaluator while accepting student prompt and parser functions.
- `llm_minimax_agent.py` connects an LLM evaluator to minimax.
- `direct_llm_agent.py` manages one model request per move while accepting
  student prompt, parser, and fallback functions.
- `student_strategies.py` is the designated location for student strategy
  functions.
- `student_agents.py` composes those functions into agents and registers them
  with the supplied runner.
- `environment.py` owns the authoritative board, asks the current agent for a
  move, applies the move, and informs a viewer.
- `viewers.py` provides optional terminal and Tk graphical viewers.
- `play.py` runs the sample random-versus-random match.

An agent receives a `JetanBoard` instance through
`choose_action(board: JetanBoard)`. The same board class provides the AIMA game
interface:

```python
player = board.player()
actions = board.actions()
successor = board.result(actions[0])
finished = successor.is_terminal()
value = successor.utility(player) if finished else None
```

The methods also accept the explicit-state forms `player(state)`,
`actions(state)`, `result(state, action)`, `is_terminal(state)`, and
`utility(state, player)`. `terminal_test(state)` is an alias for
`is_terminal(state)` for compatibility with AIMA Python implementations.
This permits an agent to treat one board as the game definition and other board
instances as states when following the pseudocode notation used by AIMA.

The supplied classes separate experiment infrastructure from strategy. Student
work belongs in focused functions with these interfaces:

```python
def evaluate_position(board: JetanBoard, perspective: Player) -> float: ...

def build_evaluation_prompt(
    board: JetanBoard, perspective: Player
) -> Sequence[dict[str, str]]: ...

def parse_evaluation_response(response: str) -> float: ...

def build_move_prompt(
    board: JetanBoard,
    actions: tuple[Move, ...],
    recent_moves: tuple[Move, ...],
) -> Sequence[dict[str, str]]: ...

def parse_move_response(response: str, actions: tuple[Move, ...]) -> Move: ...
```

Evaluation features, weights, prompts, response formats, validation rules, and
fallback policies are student design decisions. Implement them in
`student_strategies.py`. The framework adds final safety checks that evaluation
values are finite and selected moves are legal; student parsers remain
responsible for enforcing their declared response schema. Do not modify the
game rules, environment, or supplied agent lifecycle code.

Construct a minimax agent by injecting an evaluation function:

```python
agent = DepthLimitedMinimaxAgent(
    name="My Minimax Agent",
    depth=2,
    evaluation_function=evaluate_position,
)
```

`agent.last_metrics` reports generated actions, evaluated states, and maximum
visited depth for the most recent move search. Match results separately report
the winner, termination reason, plies, and cumulative thinking time.

Construct an LLM cutoff evaluator by injecting the student prompt, parser, and
deterministic fallback:

```python
llm_evaluation = LLMEvaluationFunction(
    client,
    build_evaluation_prompt,
    parse_evaluation_response,
    fallback=evaluate_position,
    max_model_calls=64,
)
agent = LLMMinimaxAgent("My LLM Minimax Agent", 1, llm_evaluation)
```

The evaluator exposes per-search and cumulative model-call, fallback, and cache
metrics. `DirectLLMAgent` exposes cumulative model and fallback counts plus its
bounded recent-move history. Minimax exposes both `last_metrics` and
`total_metrics`.

`student_agents.py` supplies the five required registrations and imports the
strategy stubs from `student_strategies.py`. After Stage A, update the evaluator
passed as the LLM fallback there if needed. The runner imports the registration
module automatically. The supplied builders handle depth, model clients,
model-call budgets, names, and factories. Preserve the five command-line names
already defined in that module.

Offline tests should inject `ScriptedClient` directly and must not make live
network requests.

## Efficient Workflow

1. Run `python3 -m unittest -v` before editing and keep development offline
   while implementing strategy functions and tests.
2. Implement the three deterministic evaluators. Use `play.py` for short
   diagnostic matches, then use `run_experiments.py` for Stage A1.
3. Select the provisional evaluator, run Stage A2, and select the final
   evaluator-depth configuration.
4. Implement and test both LLM modalities with `ScriptedClient`. Use only small
   live pilots while revising prompts and validation behavior.
5. Freeze and commit the final strategies, fallback choices, prompts, model
   settings, and experiment configuration before running Stages B and C.
6. Launch Stages B and C with `run_experiments.py` and let them run unattended.
   Work on report sections that do not require their final results. Do not edit
   agent code or configuration during a final matrix.
7. Read aggregate values and denominators from the summary CSV. Use the raw CSV
   to inspect seed/color variation, failures, termination reasons, and individual
   behavior.
8. Resume an interrupted stage with the same command and `--resume`. Successful
   configurations are skipped and failed configurations are retried.

## Watching A Match

Run an animated terminal match:

```sh
python3 play.py --view text --delay 0.1
```

Run the graphical viewer on a machine with Tk and a graphical display:

```sh
python3 play.py --view tk --delay 0.25
```

Tkinter is part of Python's standard-library interface but some Linux
distributions package its native libraries separately. Install them once on
Ubuntu or Debian with:

```sh
sudo apt install python3-tk
```

Creating `.venv` or adding `pyproject.toml` does not install these operating
system libraries. The text viewer remains available on machines without Tk or
a graphical display.

The graphical viewer uses piece artwork derived from
[`Jetan Board.svg`](https://commons.wikimedia.org/wiki/File:Jetan_Board.svg) by
Wikimedia Commons user Ninjatacoshell. The artwork is licensed under CC BY-SA
3.0. Attribution, license details, source files, and the reproducible extraction
script are in `assets/`.

Run without a viewer for experiments or automated checks:

```sh
python3 play.py --view none --orange-seed 1 --black-seed 2
```

Set independent search depths for registered student agents:

```sh
python3 play.py --orange-agent minimax_1 --black-agent random \
  --orange-depth 2 --view none
```

For live model runs, copy `.env.example` to `.env`, keep the key out of version
control, connect to the course network, and opt in explicitly:

```sh
python3 play.py --orange-agent llm_evaluator --black-agent random \
  --orange-depth 1 --max-llm-calls 100 --max-think-seconds 3600 \
  --live --view none
```

The runner reports match outcome, utility, thinking time, cumulative search
work, and available model-call metrics. Search counts are summed across all
moves made by that agent in the match; maximum depth is the largest depth it
visited. Direct-agent model counts cover the match. LLM-evaluator model,
fallback, and cache counts are cumulative across all of its move searches. Use
`ScriptedClient` for reproducible offline tests before enabling live requests.

## Batch Experiments

`run_experiments.py` runs the required experiment matrices without a viewer and
writes one raw CSV row for every attempted match. It also writes an aggregate
summary beside the requested output. For example, `--output results/a1.csv`
creates `results/a1.csv` and `results/a1-summary.csv`. Supply an output path for
every stage. Stages that use a selected deterministic agent also require its
registered evaluator name; Stages B and C require its selected depth.

```sh
python3 run_experiments.py a1 --output results/a1.csv
python3 run_experiments.py a2 --evaluator minimax_2 --output results/a2.csv
python3 run_experiments.py b --evaluator minimax_2 --depth 2 \
  --output results/b.csv
python3 run_experiments.py c --evaluator minimax_2 --depth 2 \
  --output results/c.csv
```

Stages A1 and A2 run every focal agent with seeds 0 and 1 in both colors against
the random agent. These rows put the same seed in both player slots so the CSV
shows the random-opponent configuration directly. Stage A2 contains the four
new depth-2 matches. Use the selected evaluator's depth-1 rows from Stage A1
when preparing the Stage A2 comparison. Stage B runs the selected deterministic
agent, the LLM evaluator, and the direct LLM agent with both seeds and colors.
Stage C runs the direct LLM agent in both colors against the selected
deterministic agent with both seeds.
Stages B and C use 60-ply games. The batch runner overrides `play.py`'s general
300-second default with a cumulative think-time limit of 3600 seconds per
player, which is suitable for unattended LLM evaluation. Set a different
positive limit when needed:

```sh
python3 run_experiments.py b --evaluator minimax_2 --depth 2 \
  --max-think-seconds 7200 --output results/b.csv
```

Stages B and C enable live requests only for matches containing an LLM agent,
so configure `.env` as described above before running them. The runner prints
the raw and summary paths before starting and checkpoints both CSV files after
every attempted match. Failed attempts retain their exit status, captured
output, error, and any metrics emitted before failure.

Resume an interrupted stage with the same configuration and output path:

```sh
python3 run_experiments.py b --evaluator minimax_2 --depth 2 \
  --output results/b.csv --resume
```

`--resume` skips only exact scheduled configurations whose raw row has
`status=ok`. It reruns failed configurations and replaces their rows, so the
summary counts each configuration once. Keep the evaluator, depth, and
think-time limit unchanged when resuming the same matrix.

The summary reports each tested agent from its own color perspective, excludes
missing metrics from means, and gives the denominator for every mean. Stage C
includes separate rows for the direct LLM and selected deterministic agent.

For direct `play.py` use, each player has a cumulative wall-clock think-time
budget with a 300-second default; set it with `--max-think-seconds`. The clock
measures only the player's `choose_action` call, so viewer rendering and
animation delays do not consume the budget. The final output reports both
players' measured times. An action returned after its player's cumulative time
exceeds the limit is discarded and that player loses by `time_limit`.

```sh
python3 play.py --view none --max-think-seconds 60
```

The environment checks the clock when `choose_action` returns. Agents run in
the match process, so a call that never returns cannot be interrupted safely by
this interface.

Select the agent in each player slot independently:

```sh
python3 play.py --orange-agent random --black-agent random --view text
```

Play Orange against the random agent:

```sh
python3 play.py --orange-agent terminal --black-agent random --view text
```

At the prompt, enter a coordinate move such as `c1-d2`, enter `list` to print
numbered legal moves, or enter one of those action numbers. Use two `terminal`
agents for a local two-person game. Both viewers label columns `a-j` and rows
`0-9` to match this move notation.

Run `python3 play.py --help` to see every registered agent type.

Uppercase terminal symbols are Orange pieces and lowercase symbols are Black
pieces: Warrior `W`, Padwar `A`, Dwar `D`, Flier `F`, Chief `C`, Princess `R`,
Thoat `T`, and Panthan `P`.

## Implemented Rules

The initial arrangement and movement behavior use a declarative implementation
of the rule descriptions in the 2017 course C++ source. This avoids copying its
global move cache and malformed Dwar path entry. Non-jumping pieces require
clear intermediate squares. Fliers, thoats, and princesses jump. A princess
cannot capture or land on an attacked square. An attacked princess may use one
board-wide escape. Capturing a princess wins; a chief capturing the opposing
chief also wins. A player with no legal move loses.

The environment declares a draw at `--max-plies` to ensure every automated
match terminates. The historical C++ implementation did not resolve stalled or
unbounded games. This prototype treats a position with no legal move as a loss
for the current player. The default move limit is 500 plies.

The legacy socket messages and explicit quit action are outside this prototype.
They can be added at the environment boundary without changing the board's
search interface.

## Tests

```sh
python3 -m unittest -v
```

The tests cover setup, legal-action ordering, immutable transitions, both AIMA
calling forms, terminal utility, princess rules, seeded random behavior,
environment-agent integration, minimax lifecycle and metrics, callback-based LLM
agents, model fallbacks, viewer notifications, time forfeits, think-time
accounting, and move-limit draws.
