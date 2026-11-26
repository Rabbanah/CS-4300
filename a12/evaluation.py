"""Repeatable U0-HW-12 Evaluation Harness: Baseline-to-Revised & Multi-Seed Benchmark.

Executes:
  Part A: 10 Retained Evaluation Cases (ordinary, adversarial corner-clustered, difficult
          dense-packed block, strict-budget boundary, unsuccessful TIMEOUT, unsupported
          INVALID_CONFIG x2, and expected-failure BELIEF_COLLAPSE) comparing the preserved
          U0-HW-11 baseline (`smc_baseline`) directly against the U0-HW-12 refined agent
          (`smc_info_gain`), plus `smc_refined` and `hunt_target` references.
  Part B: 30-Seed Paired 10x10 Benchmark (`seeds 4300..4329`, B_max=75) across 5 agents
          (`random`, `hunt_target`, `smc_baseline`, `smc_refined`, `smc_info_gain`) with
          95% deterministic bootstrap confidence intervals, latency distributions (mean,
          P95, peak, and >50 ms violation count), MCMC move efficiency, and reliability
          across shot budgets B_max in {50, 60, 75}.

Preserves exact command metadata, raw CSVs, JSON summaries, and full step-by-step traces
under `a12/results/`.
"""

from __future__ import annotations

import csv
import json
import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from battleship_env import EnvConfig, POC_FLEET, STANDARD_FLEET, ShotResult
from runner import EpisodeResult, PerceptHook, run_episode

RESULTS_DIR = _THIS_DIR / "results"
EVAL_TRACES_DIR = RESULTS_DIR / "eval_traces"

CORNER_CLUSTERED_SHIPS: dict[str, list[tuple[int, int]]] = {
    "Carrier": [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],
    "Battleship": [(9, 6), (9, 7), (9, 8), (9, 9)],
    "Cruiser": [(0, 9), (1, 9), (2, 9)],
    "Submarine": [(7, 0), (8, 0), (9, 0)],
    "Destroyer": [(9, 1), (9, 2)],
}

PACKED_BLOCK_SHIPS: dict[str, list[tuple[int, int]]] = {
    "Carrier": [(3, 3), (4, 3), (5, 3), (6, 3), (7, 3)],
    "Battleship": [(3, 4), (4, 4), (5, 4), (6, 4)],
    "Cruiser": [(3, 5), (4, 5), (5, 5)],
    "Submarine": [(3, 6), (4, 6), (5, 6)],
    "Destroyer": [(6, 5), (6, 6)],
}


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    category: str
    description: str
    config: EnvConfig
    percept_hook: PerceptHook | None = None
    run_hunt_target_ref: bool = False


def _corrupted_sink_on_step_1(step: int, percept: ShotResult) -> ShotResult:
    """Fault-injection hook simulating a corrupted sensor announcing Cruiser sunk on shot 1."""
    if step == 1:
        return ShotResult(coord=percept.coord, hit=True, sunk_ship="Cruiser")
    return percept


def get_evaluation_cases() -> list[EvaluationCase]:
    """Return the 10 deterministic retained evaluation cases."""
    return [
        EvaluationCase(
            case_id="eval_01_standard_seed4300",
            category="Ordinary Supported (HW11 Carryover)",
            description="10x10 standard 5-ship fleet (seed=4300), B_max=75.",
            config=EnvConfig(
                rows=10,
                cols=10,
                fleet_spec=dict(STANDARD_FLEET),
                shot_budget=75,
                seed=4300,
            ),
            run_hunt_target_ref=True,
        ),
        EvaluationCase(
            case_id="eval_02_standard_seed4301",
            category="Ordinary Supported",
            description="10x10 standard 5-ship fleet (seed=4301), B_max=75.",
            config=EnvConfig(
                rows=10,
                cols=10,
                fleet_spec=dict(STANDARD_FLEET),
                shot_budget=75,
                seed=4301,
            ),
            run_hunt_target_ref=True,
        ),
        EvaluationCase(
            case_id="eval_03_standard_seed4308",
            category="Ordinary Supported (Latency Stress)",
            description="10x10 standard 5-ship fleet (seed=4308), B_max=75.",
            config=EnvConfig(
                rows=10,
                cols=10,
                fleet_spec=dict(STANDARD_FLEET),
                shot_budget=75,
                seed=4308,
            ),
            run_hunt_target_ref=True,
        ),
        EvaluationCase(
            case_id="eval_04_corner_clustered_boundary",
            category="Adversarial Boundary (Corners/Edges)",
            description="10x10 fixed fleet tucked into all four corners/edges, B_max=75.",
            config=EnvConfig(
                rows=10,
                cols=10,
                fleet_spec=dict(STANDARD_FLEET),
                shot_budget=75,
                seed=4300,
                fixed_ships=CORNER_CLUSTERED_SHIPS,
            ),
            run_hunt_target_ref=True,
        ),
        EvaluationCase(
            case_id="eval_05_packed_block_difficult",
            category="Difficult Boundary (Dense 5x4 Cluster)",
            description="10x10 fixed fleet packed orthogonally adjacent inside rows 3..7, cols 3..6, B_max=75.",
            config=EnvConfig(
                rows=10,
                cols=10,
                fleet_spec=dict(STANDARD_FLEET),
                shot_budget=75,
                seed=4300,
                fixed_ships=PACKED_BLOCK_SHIPS,
            ),
            run_hunt_target_ref=True,
        ),
        EvaluationCase(
            case_id="eval_06_strict_budget_b48",
            category="Tight-Budget Boundary (B_max=48)",
            description="10x10 standard fleet (seed=4300) under harsh budget ceiling B_max=48.",
            config=EnvConfig(
                rows=10,
                cols=10,
                fleet_spec=dict(STANDARD_FLEET),
                shot_budget=48,
                seed=4300,
            ),
            run_hunt_target_ref=True,
        ),
        EvaluationCase(
            case_id="eval_07_tight_budget_timeout",
            category="Unsuccessful (Budget Exhaustion)",
            description="10x10 standard fleet (seed=4302) under tight budget B_max=30 -> TIMEOUT.",
            config=EnvConfig(
                rows=10,
                cols=10,
                fleet_spec=dict(STANDARD_FLEET),
                shot_budget=30,
                seed=4302,
            ),
        ),
        EvaluationCase(
            case_id="eval_08_unsupported_grid_config",
            category="Unsupported Input (Grid Capacity)",
            description="Standard 5-ship fleet (17 cells, Carrier length 5) on 4x4 grid -> INVALID_CONFIG.",
            config=EnvConfig(
                rows=4,
                cols=4,
                fleet_spec=dict(STANDARD_FLEET),
                shot_budget=16,
                seed=4300,
            ),
        ),
        EvaluationCase(
            case_id="eval_09_unsupported_budget_config",
            category="Unsupported Input (Zero Budget)",
            description="10x10 standard fleet with shot_budget=0 -> INVALID_CONFIG.",
            config=EnvConfig(
                rows=10,
                cols=10,
                fleet_spec=dict(STANDARD_FLEET),
                shot_budget=0,
                seed=4300,
            ),
        ),
        EvaluationCase(
            case_id="eval_10_corrupted_sensor_collapse",
            category="Expected Failure (Belief Collapse)",
            description="6x6 PoC fleet with injected contradictory percept (Cruiser length 3 sunk on shot 1).",
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


def _bootstrap_ci_mean(
    values: list[float], n_resamples: int = 1000, seed: int = 4300
) -> tuple[float, float]:
    """Deterministic percentile 95% bootstrap confidence interval for the mean."""
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    n = len(values)
    means: list[float] = []
    for _ in range(n_resamples):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * n_resamples)]
    hi = means[int(0.975 * n_resamples)]
    return lo, hi


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
    return ordered[idx]


def _case_row(case: EvaluationCase, variant: str, res: EpisodeResult) -> dict[str, object]:
    over_50ms = sum(
        1
        for s in res.steps
        if (s.decision_time_ms + float((s.belief_after or {}).get("update_time_ms", 0.0)))
        > 50.0
    )
    return {
        "case_id": case.case_id,
        "category": case.category,
        "variant": variant,
        "agent_name": res.agent_name,
        "status": res.status.value,
        "success": res.success,
        "shots_taken": res.shots_taken,
        "shot_budget": res.shot_budget,
        "hits": res.hits,
        "total_ship_cells": res.total_ship_cells,
        "hit_rate": round(res.hit_rate, 4),
        "mean_step_latency_ms": round(res.mean_step_latency_ms, 3),
        "max_step_latency_ms": round(res.max_step_latency_ms, 3),
        "total_step_time_ms": round(res.total_step_time_ms, 3),
        "steps_over_50ms": over_50ms,
        "mean_entropy_reduction_bits": round(res.mean_entropy_reduction_bits, 4),
        "rejuvenation_events": res.rejuvenation_events,
        "repair_events": res.repair_events,
        "total_mcmc_moves": res.total_mcmc_moves,
        "exact_fallback_step": res.exact_fallback_step,
        "error_message": res.error_message or "",
    }


def run_evaluation(
    num_seeds: int = 30,
    seed_start: int = 4300,
    num_particles: int = 250,
    save_artifacts: bool = True,
) -> dict[str, object]:
    """Run both the 10 retained comparison cases and the paired multi-seed benchmark."""
    cases = get_evaluation_cases()
    retained_rows: list[dict[str, object]] = []
    saved_traces: dict[str, dict[str, object]] = {}

    # Part A: 10 Retained Evaluation Cases (Baseline vs. Refined Max-Prob vs. Info Gain + Hunt-and-Target ref)
    for case in cases:
        for variant in ("smc_baseline", "smc_refined", "smc_info_gain"):
            res = run_episode(
                config=case.config,
                agent_type=variant,
                case_id=f"{case.case_id}__{variant}",
                num_particles=num_particles,
                mcmc_sweeps=3,
                agent_seed=4300,
                percept_hook=case.percept_hook,
            )
            retained_rows.append(_case_row(case, variant, res))
            saved_traces[f"{case.case_id}__{variant}"] = res.to_dict()

        if case.run_hunt_target_ref:
            ref_res = run_episode(
                config=case.config,
                agent_type="hunt_target",
                case_id=f"{case.case_id}__hunt_target",
                agent_seed=4300,
            )
            retained_rows.append(_case_row(case, "hunt_target", ref_res))

    # Part B: Paired Multi-Seed Benchmark (seeds seed_start .. seed_start + num_seeds - 1)
    bench_agents = (
        "random",
        "hunt_target",
        "smc_baseline",
        "smc_refined",
        "smc_info_gain",
    )
    raw_seed_rows: list[dict[str, object]] = []
    by_agent: dict[str, list[dict[str, object]]] = {ag: [] for ag in bench_agents}
    all_step_latencies: dict[str, list[float]] = {ag: [] for ag in bench_agents}

    for seed in range(seed_start, seed_start + num_seeds):
        cfg = EnvConfig(
            rows=10,
            cols=10,
            fleet_spec=dict(STANDARD_FLEET),
            shot_budget=75,
            seed=seed,
        )
        for ag in bench_agents:
            res = run_episode(
                config=cfg,
                agent_type=ag,
                case_id=f"bench_seed{seed}__{ag}",
                num_particles=num_particles,
                mcmc_sweeps=3,
                agent_seed=4300,
            )
            step_lats = [
                s.decision_time_ms
                + float((s.belief_after or {}).get("update_time_ms", 0.0))
                for s in res.steps
            ]
            all_step_latencies[ag].extend(step_lats)
            over_50 = sum(1 for x in step_lats if x > 50.0)
            row = {
                "seed": seed,
                "agent": ag,
                "status": res.status.value,
                "success": res.success,
                "shots_taken": res.shots_taken,
                "hits": res.hits,
                "hit_rate": round(res.hit_rate, 4),
                "mean_step_latency_ms": round(res.mean_step_latency_ms, 3),
                "max_step_latency_ms": round(res.max_step_latency_ms, 3),
                "total_step_time_ms": round(res.total_step_time_ms, 3),
                "steps_over_50ms": over_50,
                "mean_entropy_reduction_bits": round(
                    res.mean_entropy_reduction_bits, 4
                ),
                "rejuvenation_events": res.rejuvenation_events,
                "repair_events": res.repair_events,
                "total_mcmc_moves": res.total_mcmc_moves,
                "exact_fallback_step": res.exact_fallback_step,
            }
            raw_seed_rows.append(row)
            by_agent[ag].append(row)

    summary_rows: list[dict[str, object]] = []
    for ag in bench_agents:
        rows = by_agent[ag]
        shots = [float(r["shots_taken"]) for r in rows]
        wins_75 = sum(1 for r in rows if r["success"])
        wins_60 = sum(1 for r in rows if r["success"] and int(r["shots_taken"]) <= 60)
        wins_50 = sum(1 for r in rows if r["success"] and int(r["shots_taken"]) <= 50)
        ci_lo, ci_hi = _bootstrap_ci_mean(shots, n_resamples=1000, seed=4300)
        step_lats = all_step_latencies[ag]
        total_steps = len(step_lats)
        over_50_total = sum(int(r["steps_over_50ms"]) for r in rows)
        summary_rows.append(
            {
                "agent": ag,
                "episodes": num_seeds,
                "win_rate_b75": round(wins_75 / num_seeds, 4),
                "win_rate_b60": round(wins_60 / num_seeds, 4),
                "win_rate_b50": round(wins_50 / num_seeds, 4),
                "mean_shots": round(statistics.fmean(shots), 2),
                "median_shots": round(statistics.median(shots), 2),
                "std_shots": round(statistics.stdev(shots), 2) if num_seeds > 1 else 0.0,
                "min_shots": int(min(shots)),
                "max_shots": int(max(shots)),
                "shots_ci95_low": round(ci_lo, 2),
                "shots_ci95_high": round(ci_hi, 2),
                "mean_hit_rate": round(
                    statistics.fmean(float(r["hit_rate"]) for r in rows), 4
                ),
                "mean_step_latency_ms": round(
                    statistics.fmean(step_lats) if step_lats else 0.0, 3
                ),
                "p95_step_latency_ms": round(_percentile(step_lats, 95.0), 3),
                "peak_step_latency_ms": round(max(step_lats) if step_lats else 0.0, 3),
                "steps_over_50ms": over_50_total,
                "total_steps": total_steps,
                "mean_episode_time_ms": round(
                    statistics.fmean(float(r["total_step_time_ms"]) for r in rows), 2
                ),
                "mean_mcmc_moves": round(
                    statistics.fmean(float(r["total_mcmc_moves"]) for r in rows), 1
                ),
                "total_repairs": sum(int(r["repair_events"]) for r in rows),
                "collapses": sum(
                    1 for r in rows if r["status"] == "BELIEF_COLLAPSE"
                ),
            }
        )

    # Paired difference analyses (refined & info_gain vs baseline & hunt_target)
    paired_diffs: dict[str, dict[str, float]] = {}
    for focal in ("smc_refined", "smc_info_gain"):
        for other in ("smc_baseline", "hunt_target"):
            diffs = [
                float(by_agent[focal][i]["shots_taken"])
                - float(by_agent[other][i]["shots_taken"])
                for i in range(num_seeds)
            ]
            ci_lo, ci_hi = _bootstrap_ci_mean(diffs, n_resamples=1000, seed=4300)
            lat_diffs = [
                float(by_agent[focal][i]["mean_step_latency_ms"])
                - float(by_agent[other][i]["mean_step_latency_ms"])
                for i in range(num_seeds)
            ]
            paired_diffs[f"{focal}_minus_{other}"] = {
                "mean_shot_diff": round(statistics.fmean(diffs), 2),
                "shot_diff_ci95_low": round(ci_lo, 2),
                "shot_diff_ci95_high": round(ci_hi, 2),
                "revised_better_count": sum(1 for d in diffs if d < 0),
                "tie_count": sum(1 for d in diffs if d == 0),
                "revised_worse_count": sum(1 for d in diffs if d > 0),
                "mean_latency_diff_ms": round(statistics.fmean(lat_diffs), 3),
            }

    command_str = (
        f"python a12/evaluation.py --seeds {num_seeds} --seed-start {seed_start} "
        f"--particles {num_particles}"
    )
    payload: dict[str, object] = {
        "command": command_str,
        "num_seeds": num_seeds,
        "seed_start": seed_start,
        "num_particles": num_particles,
        "retained_cases": retained_rows,
        "multi_seed_summary": summary_rows,
        "paired_differences": paired_diffs,
    }

    if save_artifacts:
        EVAL_TRACES_DIR.mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / "evaluation_command.txt").write_text(
            command_str + "\n", encoding="utf-8"
        )
        (RESULTS_DIR / "evaluation_summary.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        with (RESULTS_DIR / "retained_cases_comparison.csv").open(
            "w", newline="", encoding="utf-8"
        ) as fh:
            writer = csv.DictWriter(fh, fieldnames=list(retained_rows[0].keys()))
            writer.writeheader()
            writer.writerows(retained_rows)
        with (RESULTS_DIR / "multi_seed_raw.csv").open(
            "w", newline="", encoding="utf-8"
        ) as fh:
            writer = csv.DictWriter(fh, fieldnames=list(raw_seed_rows[0].keys()))
            writer.writeheader()
            writer.writerows(raw_seed_rows)
        with (RESULTS_DIR / "multi_seed_summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as fh:
            writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
        for trace_key, trace_data in saved_traces.items():
            (EVAL_TRACES_DIR / f"{trace_key}.json").write_text(
                json.dumps(trace_data, indent=2), encoding="utf-8"
            )

    return payload


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Run repeatable U0-HW-12 Battleship refinement evaluation."
    )
    parser.add_argument("--seeds", type=int, default=30, help="Paired benchmark seeds (default: 30)")
    parser.add_argument("--seed-start", type=int, default=4300, help="Starting seed (default: 4300)")
    parser.add_argument("--particles", type=int, default=250, help="SMC particle count N (default: 250)")
    args = parser.parse_args()

    payload = run_evaluation(
        num_seeds=args.seeds,
        seed_start=args.seed_start,
        num_particles=args.particles,
        save_artifacts=True,
    )

    print("=== Part A: 10 Retained Evaluation Cases (Baseline vs. Revised) ===")
    hdr = (
        f"{'Case ID':<34} | {'Variant':<13} | {'Status':<15} | {'OK':<5} | "
        f"{'Shots':<6} | {'MeanMs':<6} | {'PeakMs':<6} | {'>50ms':<5} | {'MCMC':<6}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in payload["retained_cases"]:
        shots = f"{r['shots_taken']}/{r['shot_budget']}"
        print(
            f"{r['case_id']:<34} | {r['variant']:<13} | {r['status']:<15} | "
            f"{str(r['success']):<5} | {shots:<6} | {r['mean_step_latency_ms']:<6.2f} | "
            f"{r['max_step_latency_ms']:<6.2f} | {r['steps_over_50ms']:<5} | {r['total_mcmc_moves']:<6}"
        )

    print(
        f"\n=== Part B: Paired {args.seeds}-Seed 10x10 Benchmark "
        f"(seeds {args.seed_start}..{args.seed_start + args.seeds - 1}, B_max=75) ==="
    )
    hdr2 = (
        f"{'Agent':<14} | {'Win@75':<6} | {'Win@60':<6} | {'Win@50':<6} | "
        f"{'MeanShots [95% CI]':<22} | {'Std':<5} | {'MeanMs':<6} | {'P95Ms':<6} | {'PeakMs':<6} | {'>50ms':<5} | {'MCMC'}"
    )
    print(hdr2)
    print("-" * len(hdr2))
    for r in payload["multi_seed_summary"]:
        ci_str = f"{r['mean_shots']:5.2f} [{r['shots_ci95_low']:4.1f}, {r['shots_ci95_high']:4.1f}]"
        print(
            f"{r['agent']:<14} | {r['win_rate_b75']*100:5.1f}% | "
            f"{r['win_rate_b60']*100:5.1f}% | {r['win_rate_b50']*100:5.1f}% | "
            f"{ci_str:<22} | {r['std_shots']:5.2f} | {r['mean_step_latency_ms']:6.2f} | "
            f"{r['p95_step_latency_ms']:6.2f} | {r['peak_step_latency_ms']:6.2f} | "
            f"{r['steps_over_50ms']:<5} | {r['mean_mcmc_moves']:.0f}"
        )

    print("\n=== Paired Differences (smc_info_gain minus comparator) ===")
    for k, v in payload["paired_differences"].items():
        print(f"  {k}: {v}")
    print(f"\nSaved all raw results and traces under {RESULTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
