# Battleship Bayesian Belief-Tracking Agent — Refinement & Evaluation (U0-HW-12)

Refined Sequential Bayesian Belief-Tracking Battleship agent (`a12/`) building directly on the U0-HW-11 vertical slice (`b9ad421`), with preserved baseline-to-revised comparison, 10 retained evaluation cases, and a paired 30-seed statistical benchmark.

## Setup

- **Requirements:** Python 3.10+ (verified on Python 3.14.8). Standard library only — no third-party dependencies, API keys, or `.env` files are required.
- Run all commands from the repository root (`CS-4300`).

## Exact Commands

### 1. Run Automated Tests (20 Deterministic Unit & Integration Tests)

```powershell
python -m unittest discover -s a12 -p "test_*.py" -v
```

### 2. Run One Episode (Refined vs. U0-HW-11 Baseline)

```powershell
python a12/runner.py --agent smc_info_gain --seed 4300 --budget 75 --particles 250 --show-board
python a12/runner.py --agent smc_baseline  --seed 4300 --budget 75 --particles 250 --show-board
```

### 3. Run the Full Repeatable Evaluation (10 Retained Cases + 30-Seed Paired Benchmark)

```powershell
python a12/evaluation.py --seeds 30 --seed-start 4300 --particles 250
```

Preserved outputs under `a12/results/`:
- `a12/results/evaluation_command.txt` — exact evaluation command
- `a12/results/retained_cases_comparison.csv` — baseline vs. revised metrics across all 10 retained cases
- `a12/results/multi_seed_raw.csv` — all 150 raw paired episode runs (30 seeds × 5 agents)
- `a12/results/multi_seed_summary.csv` — aggregate statistics, 95% bootstrap CIs, latency P95/peak, and MCMC effort
- `a12/results/evaluation_summary.json` — complete structured summary + paired differences
- `a12/results/eval_traces/*.json` — full step-by-step execution traces for every retained case

### 4. Regenerate the Accessible Evaluation PDF

```powershell
python a12/generate_report.py --commit <commit-sha>
```

## Key U0-HW-12 Refinements Over U0-HW-11 (`smc_baseline`)

1. **Cell-Indexed Placement Lookup Tables & Fast Unconstrained Gibbs Sampling (`optimized_mcmc=True`):** Replaces $O(|\text{domain}|)$ placement scans on hit-constrained Gibbs and two-ship block moves with $O(k)$ (`k <= 8`) bit-indexed lookups and rejection-first unconstrained sampling.
2. **ESS / Survival-Gated Adaptive MCMC Rejuvenation:** Uses 1 lightweight Gibbs sweep without two-ship coupling on high-survival routine misses (`survived >= 0.65N`, diversity `>= 0.70`), 2 sweeps under moderate pruning, and 3 sweeps under severe pruning (`survived < 0.20N`), cutting MCMC proposals by ~51% and peak step latency by ~4× while preserving 100% particle consistency.
3. **Exact One-Step Expected Percept Information Gain (`smc_info_gain`):** Computes exact multi-outcome mutual information $I(F; Y_c \mid \text{history}) = H(Y_c \mid \text{history})$ over `MISS`, `HIT`, and ship-specific `SUNK(s)` outcomes in a single pass over active particles, paired with certainty exploitation ($P(\text{occ}_c) = 1.0$) and composite search-and-destroy scoring $P(\text{occ}_c) + 0.35\,I(F; Y_c)$.
