"""Retained baseline measurement cases for the U0-HW-11 Vertical Slice.

Executes five deterministic retained cases covering:
  1. Representative supported case (10x10 standard fleet, seed=4300, B_max=75)
  2. Adversarial boundary case (10x10 corner-clustered fleet, B_max=75)
  3. Unsuccessful outcome (10x10 standard fleet under tight shot budget B_max=30 -> TIMEOUT)
  4. Unsupported input (standard 5-ship fleet on 4x4 grid -> INVALID_CONFIG)
  5. Expected failure mode (corrupted sensor reporting 3-cell Cruiser sunk on 1st hit -> BELIEF_COLLAPSE)

Also runs paired Hunt-and-Target reference comparisons on the two supported 10x10 layouts
to contextualize baseline efficiency prior to the full 1,000-episode evaluation.

Preserves exact command metadata, CSV/JSON summaries, and full step-by-step JSON traces
under `a11/results/`.
"""

from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from battleship_env import EnvConfig, POC_FLEET, STANDARD_FLEET, ShotResult
from runner import EpisodeResult, PerceptHook, run_episode

RESULTS_DIR = _THIS_DIR / "results"
TRACES_DIR = RESULTS_DIR / "traces"

CORNER_CLUSTERED_SHIPS: dict[str, list[tuple[int, int]]] = {
    "Carrier": [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],
    "Battleship": [(9, 6), (9, 7), (9, 8), (9, 9)],
    "Cruiser": [(0, 9), (1, 9), (2, 9)],
    "Submarine": [(7, 0), (8, 0), (9, 0)],
    "Destroyer": [(9, 1), (9, 2)],
}


@dataclass(frozen=True)
class RetainedCase:
    case_id: str
    category: str
    description: str
    config: EnvConfig
    agent_type: str = "smc"
    num_particles: int = 250
    mcmc_sweeps: int = 3
    agent_seed: int = 4300
    percept_hook: PerceptHook | None = None


def _corrupted_sink_on_step_1(step: int, percept: ShotResult) -> ShotResult:
    """Fault-injection hook simulating a corrupted sensor announcing Cruiser sunk on shot 1."""
    if step == 1:
        return ShotResult(coord=percept.coord, hit=True, sunk_ship="Cruiser")
    return percept


def get_retained_cases() -> list[RetainedCase]:
    """Return the five deterministic retained baseline cases."""
    return [
        RetainedCase(
            case_id="case_01_standard_seed4300",
            category="Representative Supported",
            description="10x10 standard 5-ship fleet (seed=4300), shot budget B_max=75.",
            config=EnvConfig(
                rows=10,
                cols=10,
                fleet_spec=dict(STANDARD_FLEET),
                shot_budget=75,
                seed=4300,
            ),
        ),
        RetainedCase(
            case_id="case_02_corner_clustered_boundary",
            category="Adversarial Boundary",
            description="10x10 fixed fleet tucked into all four corners/edges, B_max=75.",
            config=EnvConfig(
                rows=10,
                cols=10,
                fleet_spec=dict(STANDARD_FLEET),
                shot_budget=75,
                seed=4300,
                fixed_ships=CORNER_CLUSTERED_SHIPS,
            ),
        ),
        RetainedCase(
            case_id="case_03_tight_budget_timeout",
            category="Unsuccessful (Timeout)",
            description="10x10 standard fleet (seed=4302) under tight shot budget B_max=30.",
            config=EnvConfig(
                rows=10,
                cols=10,
                fleet_spec=dict(STANDARD_FLEET),
                shot_budget=30,
                seed=4302,
            ),
        ),
        RetainedCase(
            case_id="case_04_unsupported_config",
            category="Unsupported Input",
            description="Standard 5-ship fleet (17 cells, max length 5) on unsupported 4x4 grid.",
            config=EnvConfig(
                rows=4,
                cols=4,
                fleet_spec=dict(STANDARD_FLEET),
                shot_budget=16,
                seed=4300,
            ),
        ),
        RetainedCase(
            case_id="case_05_corrupted_sensor_collapse",
            category="Expected Failure (Collapse)",
            description="6x6 PoC fleet with injected contradictory percept (Cruiser length 3 sunk on step 1).",
            config=EnvConfig(
                rows=6,
                cols=6,
                fleet_spec=dict(POC_FLEET),
                shot_budget=36,
                seed=4300,
            ),
            percept_hook=_corrupted_sink_on_step_1,
        ),
    ]


def get_reference_comparisons() -> list[RetainedCase]:
    """Paired Hunt-and-Target comparisons on the two supported 10x10 layouts."""
    return [
        RetainedCase(
            case_id="ref_01_hunt_target_seed4300",
            category="Reference Baseline",
            description="Hunt-and-Target parity baseline on Case 01 (10x10, seed=4300, B_max=75).",
            config=EnvConfig(
                rows=10,
                cols=10,
                fleet_spec=dict(STANDARD_FLEET),
                shot_budget=75,
                seed=4300,
            ),
            agent_type="hunt_target",
        ),
        RetainedCase(
            case_id="ref_02_hunt_target_corner_clustered",
            category="Reference Baseline",
            description="Hunt-and-Target parity baseline on Case 02 (10x10 corner-clustered, B_max=75).",
            config=EnvConfig(
                rows=10,
                cols=10,
                fleet_spec=dict(STANDARD_FLEET),
                shot_budget=75,
                seed=4300,
                fixed_ships=CORNER_CLUSTERED_SHIPS,
            ),
            agent_type="hunt_target",
        ),
    ]


def _summary_row(case: RetainedCase, res: EpisodeResult) -> dict[str, object]:
    return {
        "case_id": case.case_id,
        "category": case.category,
        "agent": res.agent_name,
        "status": res.status.value,
        "success": res.success,
        "shots_taken": res.shots_taken,
        "shot_budget": res.shot_budget,
        "hits": res.hits,
        "total_ship_cells": res.total_ship_cells,
        "hit_rate": round(res.hit_rate, 4),
        "mean_step_latency_ms": round(res.mean_step_latency_ms, 3),
        "max_step_latency_ms": round(res.max_step_latency_ms, 3),
        "mean_entropy_reduction_bits": round(res.mean_entropy_reduction_bits, 4),
        "rejuvenation_events": res.rejuvenation_events,
        "repair_events": res.repair_events,
        "exact_fallback_step": res.exact_fallback_step,
        "error_message": res.error_message or "",
    }


def run_all_baselines(save_artifacts: bool = True) -> dict[str, object]:
    """Execute all retained cases and reference baselines, optionally persisting outputs."""
    primary_cases = get_retained_cases()
    ref_cases = get_reference_comparisons()

    primary_rows: list[dict[str, object]] = []
    ref_rows: list[dict[str, object]] = []
    traces: dict[str, dict[str, object]] = {}

    for case in primary_cases:
        res = run_episode(
            config=case.config,
            agent_type=case.agent_type,
            case_id=case.case_id,
            num_particles=case.num_particles,
            mcmc_sweeps=case.mcmc_sweeps,
            agent_seed=case.agent_seed,
            percept_hook=case.percept_hook,
            verbose=False,
        )
        primary_rows.append(_summary_row(case, res))
        traces[case.case_id] = res.to_dict()

    for case in ref_cases:
        res = run_episode(
            config=case.config,
            agent_type=case.agent_type,
            case_id=case.case_id,
            agent_seed=case.agent_seed,
            verbose=False,
        )
        ref_rows.append(_summary_row(case, res))
        traces[case.case_id] = res.to_dict()

    payload = {
        "command": "python a11/baseline_cases.py",
        "retained_cases": primary_rows,
        "reference_baselines": ref_rows,
    }

    if save_artifacts:
        TRACES_DIR.mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / "baseline_command.txt").write_text(
            "python a11/baseline_cases.py\n", encoding="utf-8"
        )
        (RESULTS_DIR / "baseline_summary.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        csv_fields = list(primary_rows[0].keys())
        with (RESULTS_DIR / "baseline_summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as fh:
            writer = csv.DictWriter(fh, fieldnames=csv_fields)
            writer.writeheader()
            writer.writerows(primary_rows)
            writer.writerows(ref_rows)
        for case_id, trace_data in traces.items():
            (TRACES_DIR / f"{case_id}.json").write_text(
                json.dumps(trace_data, indent=2), encoding="utf-8"
            )

    return payload


def main() -> int:
    payload = run_all_baselines(save_artifacts=True)
    print("=== Retained Vertical-Slice Baseline Results ===")
    header = (
        f"{'Case ID':<34} | {'Status':<15} | {'Success':<7} | {'Shots':<7} | "
        f"{'HitRate':<7} | {'MeanMs':<7} | {'MaxMs':<7} | {'Rejuv':<5} | {'Rep':<3} | {'Exact'}"
    )
    print(header)
    print("-" * len(header))
    for row in payload["retained_cases"]:
        shots = f"{row['shots_taken']}/{row['shot_budget']}"
        exact = str(row["exact_fallback_step"]) if row["exact_fallback_step"] is not None else "-"
        print(
            f"{row['case_id']:<34} | {row['status']:<15} | {str(row['success']):<7} | "
            f"{shots:<7} | {row['hit_rate']:<7.3f} | {row['mean_step_latency_ms']:<7.2f} | "
            f"{row['max_step_latency_ms']:<7.2f} | {row['rejuvenation_events']:<5} | "
            f"{row['repair_events']:<3} | {exact}"
        )
    print("\n=== Paired Hunt-and-Target Reference Baselines ===")
    for row in payload["reference_baselines"]:
        shots = f"{row['shots_taken']}/{row['shot_budget']}"
        print(
            f"{row['case_id']:<34} | {row['status']:<15} | {str(row['success']):<7} | "
            f"{shots:<7} | {row['hit_rate']:<7.3f} | {row['mean_step_latency_ms']:<7.2f}"
        )
    print(f"\nSaved raw results and traces under {RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
