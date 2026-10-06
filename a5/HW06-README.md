# Assignment 6 Extension: Alpha-Beta Search

This is an extension package for your completed Assignment 5 repository. It
uses the `jetan.py`, `student_strategies.py`, agent registry, and `play.py` that
are already in that repository. It does not include replacements for those
files and does not include MCTS or LLM implementation code.

## Install The Extension

From the root of your completed Assignment 5 repository, extract or copy every
file in this package into that root directory. The extension filenames use
`hw06_` or `alpha_beta` prefixes so they do not overwrite your Assignment 5
work.

## Select Your Assignment 5 Evaluator

Open `hw06_config.py`. It contains the one clearly marked configuration choice:

```python
from student_strategies import evaluate_position_3 as SELECTED_EVALUATOR
```

Change only `evaluate_position_3` if your final Assignment 5 evaluator was
`evaluate_position_1` or `evaluate_position_2`. Both measured minimax and
alpha-beta import this same alias. Keep the evaluator implementation in
`student_strategies.py`; do not copy, paste, migrate, or wrap it.

## Register The Agents

Add this one line at the bottom of your existing `student_agents.py`:

```python
import hw06_registration  # noqa: E402,F401
```

The import registers the exact names `alpha_beta_order_1` and
`alpha_beta_order_2`. Your existing `play.py` then works unchanged:

```bash
python3 play.py --view text --orange-agent alpha_beta_order_1 --black-agent random --orange-depth 2 --max-plies 20 --max-think-seconds 60
python3 play.py --view text --orange-agent alpha_beta_order_2 --black-agent minimax_3 --orange-depth 2 --black-depth 2 --max-plies 20 --max-think-seconds 60
```

In the second command, replace `minimax_3` with the Assignment 5 registration
name that uses your selected evaluator. These single 1v1 development games are
diagnostic and are excluded from the required experiment counts. Each command
stops after 20 plies and gives each player at most 60 cumulative thinking
seconds.

## Student Work

1. Implement `AlphaBetaAgent._value` in `alpha_beta_agent.py`.
2. Implement `ordering_key_1` and `ordering_key_2` in
   `alpha_beta_ordering.py` as distinct general move-ordering strategies.
3. Keep your selected evaluator unchanged while collecting experiment data.

The alpha-beta scaffold requires a fixed root perspective, terminal testing
before the depth cutoff, canonical root traversal, ordering only below the
root, strict root tie handling, and the selected move and root value. Its
metrics are `visited_states`, `expanded_nodes`, `generated_actions`,
`evaluated_states`, `pruned_actions`, `maximum_depth`, and `thinking_time`.
The `generated` and `evaluated` aliases keep the existing
`play.py` metric formatter compatible.

Each move-ordering function receives the canonical action tuple produced once
by `board.actions()`. Do not call `board.actions()` in an ordering key. The
supplied stable-sort wrapper preserves canonical input order when keys tie.

`hw06_measured_minimax.py` is a supplied exhaustive minimax implementation for
the position experiments. It does not replace your Assignment 5
`minimax_agent.py`. It uses the same selected evaluator, root perspective,
terminal-before-cutoff semantics, and metric definitions as alpha-beta.

## Required Alpha-Cutoff Case

Create one depth-2 `ToyBoard` test that causes an alpha cutoff at a MIN node.
Use the `ToyBoard` format and `AlphaBetaRequirements` test class in
`test_hw06_student.py`. Replace the body of
`test_student_alpha_cutoff_case`; the supplied `self.fail(...)` keeps this
student requirement visible until you complete it. The first root branch must
establish alpha. In a later MIN branch, its first child must produce a value
less than or equal to alpha so the remaining sibling is pruned. Choose your own
tree and leaf values; do not add a separate beta-cutoff requirement.

In your report, document the tree, leaf values, traversal order, alpha value
established by the first root branch, and sibling pruned in the later MIN
branch.

## Tests

Run the supplied framework and support tests:

```bash
python3 -m unittest -v test_hw06_framework.py
```

Run the student-owned requirements:

```bash
python3 -m unittest -v test_hw06_student.py
```

When these files are merged into the HW5 repository, the unmodified
framework/support tests pass. Student tests fail only for the unfinished
alpha-beta recursion, two ordering-key TODOs, and the student-created
alpha-cutoff case. A completed valid implementation passes both test files.

The package was integration-validated by copying an unmodified HW5 starter and
these extension files to a temporary directory, assigning a deterministic test
evaluator at runtime with `hw06_config.SELECTED_EVALUATOR = test_evaluator`,
and running the framework tests. This fixture avoids editing the HW5 evaluator
stubs and is only a package validation technique; use your completed evaluator
for your work.

The six supplied positions are used by the public tests and by the required
experiment runner. Keep their identifiers and definitions unchanged.

## Complexity And Empirical Work

Let `b` be the branching factor and `d` be the search depth. Depth-limited
minimax takes `O(b^d)` time. Alpha-beta has the same `O(b^d)` worst-case time;
with ideal move ordering, its time is near `O(b^(d/2))`. Both depth-first
searches use `O(bd)` space.

The runner's node counts measure empirical search work on the supplied finite
positions. Elapsed time also depends on the implementation, hardware, and the
cost of each ordering key. A finite experiment does not prove an asymptotic
bound. This assignment does not collect memory measurements.

## Measurements

Run the required matrix with:

```bash
python3 hw06_run_experiments.py --output hw06_results
```

Stage A runs measured exhaustive minimax and both alpha-beta move-ordering
strategies once at depth 2 on all six fixed positions: 18 searches. It checks
selected-move and root-value equivalence and measures alpha-beta pruning against
exhaustive minimax at the same depth. Stage B runs both alpha-beta move-ordering
strategies at depth 3 on all six positions with three repetitions: 36 attempts.
Stage B measures the relative effects of the two strategies within alpha-beta.
It has no depth-3 unpruned baseline.

The default command runs both stages: 54 attempts total. The default timeout is
30 seconds per search and is enforced by a separate child process on Windows,
macOS, and Linux. Child-process startup is outside the agent's recorded
`thinking_time`.

### Trial Commands And Bounds

Use these trials before the required run:

```bash
# One search: 30-second search limit; allow about 1 minute elapsed.
python3 hw06_run_experiments.py --stage B --position reduced --agent alpha_beta_order_1 --repeats 1 --timeout 30 --output hw06_trial_one

# Stage B with one repetition: 6-minute search ceiling; allow about 7 minutes.
python3 hw06_run_experiments.py --stage B --repeats 1 --timeout 30 --output hw06_trial_stage_b

# Required Stage B: 18-minute search ceiling; allow about 20 minutes.
python3 hw06_run_experiments.py --stage B --timeout 30 --output hw06_stage_b

# Required full matrix: 27-minute search ceiling; allow about 30 minutes.
python3 hw06_run_experiments.py --timeout 30 --output hw06_results
```

The elapsed-time allowances include ordinary child-process startup and cleanup.
Practical runtime is likely lower because the depth-2 searches and many
reduced-material or middle-game searches finish before the limit. Trial outputs
do not count as final evidence. Use the default three repetitions for final
Stage B evidence.

The runner writes `raw.csv` and `summary.csv`. `raw.csv` retains every attempt,
including its stage, position, category, depth, agent, repetition, status,
selected move, root value, seven metrics, and exception details. Status is
`ok`, `timeout`, or `error`; unavailable values are `N/A`.

`summary.csv` has one row per selected stage-position-agent group. It reports
attempt, completion, timeout, and error counts; a selected move and root value
when completed repetitions agree; stable work metrics; and median
`thinking_time`. Stage A's `equivalent_move` and `equivalent_value` compare all
three matched-depth searches when the complete comparison is present.

For a position where both depth-3 configurations complete with stable expanded
node counts, `relative_expanded_nodes_change_order2_vs_order1` is
`(order1 - order2) / order1`. A positive value means Ordering 2 expanded fewer
nodes, zero means the same count, and a negative value means Ordering 2 expanded
more. Ordering 1 is the arbitrary denominator baseline, not an unpruned search.
Other rows use `N/A` for this column.

The final `hw06_results/` directory is intentionally not ignored by Git. Commit
its `raw.csv` and `summary.csv` files with your submission. Trial directories
whose names begin with `hw06_trial` are ignored and do not count as final
evidence.

## Extension Files

- `hw06_config.py`: the single evaluator-selection import.
- `alpha_beta_agent.py`: alpha-beta scaffold and metric type.
- `alpha_beta_ordering.py`: stable wrappers and two student ordering keys.
- `hw06_measured_minimax.py`: supplied measured exhaustive minimax.
- `hw06_positions.py`: six fixed positions using the existing HW5 engine.
- `hw06_registration.py`: registrations for the existing agent framework.
- `hw06_run_experiments.py`: Stage A and Stage B CSV runner.
- `test_hw06_framework.py`: initially passing framework/support tests.
- `test_hw06_student.py`: initially failing student requirements.
