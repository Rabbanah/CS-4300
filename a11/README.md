# Battleship Bayesian Belief-Tracking Agent — Vertical Slice (U0-HW-11)

Runnable end-to-end vertical slice of the **Sequential Bayesian Belief-Tracking Battleship Agent** proposed in U0-HW-10 (`a10/project_definition_proof_of_concept.pdf`).

## Setup

- **Requirements:** Python 3.10+ (verified on Python 3.14.8). Standard library only — no third-party packages, API keys, or `.env` files are required.
- Run all commands from the repository root (`CS-4300`):

```powershell
python --version
```

## Exact Commands

### 1. Run Automated Tests (Deterministic, Repeatable)

```powershell
python -m unittest discover -s a11 -p "test_*.py" -v
```

### 2. Run One Complete Vertical-Slice Episode (10×10 Standard Fleet)

```powershell
python a11/runner.py --seed 4300 --budget 75 --particles 250 --show-board
```

Optional trace export or baseline comparison:

```powershell
python a11/runner.py --seed 4300 --agent smc --trace-out a11/results/traces/ad_hoc.json
python a11/runner.py --seed 4300 --agent hunt_target --show-board
```

### 3. Run Retained Baseline Measurements (5 Retained Cases + 2 Reference Baselines)

```powershell
python a11/baseline_cases.py
```

Raw outputs are preserved under `a11/results/`:
- `a11/results/baseline_command.txt` — exact measurement command
- `a11/results/baseline_summary.csv` — tabular baseline results
- `a11/results/baseline_summary.json` — structured JSON summary
- `a11/results/traces/*.json` — full step-by-step execution traces for all retained and reference cases

## Architecture Overview

| Module | Responsibility |
| :--- | :--- |
| `battleship_env.py` | Deterministic Battleship simulator (`BattleshipEnv`), input validation (`validate_config`, `InvalidConfigError`), action enforcement (`InvalidActionError`), `ShotResult` percepts, and explicit `EpisodeStatus` lifecycle enum (`RUNNING`, `VICTORY`, `TIMEOUT`, `BELIEF_COLLAPSE`, `INVALID_CONFIG`, `INVALID_ACTION`). |
| `belief_agent.py` | Approved AI strategy (`ParticleFilterAgent`): bitmask exact likelihood filtering (hits, misses, sunk-ship identity + sinking coordinate), MRV hit-covering backtracking repair on particle deprivation, exact single-ship Gibbs + coupled two-ship MCMC rejuvenation, automatic exact endgame enumeration fallback (`EXACT_FALLBACK`), plus `HuntAndTargetAgent` and `RandomAgent` reference baselines. |
| `runner.py` | Application-controlled agent-environment loop (`run_episode`), per-step inspectable `StepTrace` recording (available actions, top candidate actions, belief telemetry, percepts, environment effects, latency, and board entropy reduction), and CLI entrypoint. |
| `baseline_cases.py` | Deterministic 5-case baseline measurement suite (representative supported, adversarial corner-clustered boundary, tight-budget `TIMEOUT`, unsupported `INVALID_CONFIG`, and corrupted-sensor `BELIEF_COLLAPSE`) plus paired `HuntAndTargetAgent` reference checks. |
| `test_*.py` | Automated `unittest` suite covering deterministic environment rules, exact 6×6 PoC equivalence (`2,472` hypotheses), SMC constraint invariance, particle deprivation repair, belief collapse detection, and end-to-end runner integration. |
