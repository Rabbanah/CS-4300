"""Application-controlled vertical slice runner and CLI for Battleship agents.

Connects BattleshipEnv and the selected agent through a strict percept/action boundary,
recording inspectable step traces, belief telemetry, environment effects, and explicit
terminal outcomes (VICTORY, TIMEOUT, BELIEF_COLLAPSE, INVALID_CONFIG, INVALID_ACTION).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# Ensure sibling imports work when invoked as `python a11/runner.py` from repo root.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from battleship_env import (
    BattleshipEnv,
    EnvConfig,
    EpisodeStatus,
    InvalidActionError,
    InvalidConfigError,
    POC_FLEET,
    STANDARD_FLEET,
    ShotResult,
)
from belief_agent import (
    BattleshipAgent,
    BeliefCollapseError,
    HuntAndTargetAgent,
    ParticleFilterAgent,
    RandomAgent,
)

PerceptHook = Callable[[int, ShotResult], ShotResult]


@dataclass
class StepTrace:
    """Complete inspectable record of a single agent-environment interaction step."""

    step: int
    available_actions_count: int
    belief_before: dict[str, object]
    top_candidates: list[dict[str, object]]
    chosen_action: list[int]
    chosen_prob: float
    chosen_cell_entropy_bits: float
    decision_time_ms: float
    percept: dict[str, object]
    belief_after: dict[str, object] | None
    env_effect: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "step": self.step,
            "available_actions_count": self.available_actions_count,
            "belief_before": self.belief_before,
            "top_candidates": self.top_candidates,
            "chosen_action": self.chosen_action,
            "chosen_prob": round(self.chosen_prob, 6),
            "chosen_cell_entropy_bits": round(self.chosen_cell_entropy_bits, 6),
            "decision_time_ms": round(self.decision_time_ms, 3),
            "percept": self.percept,
            "belief_after": self.belief_after,
            "env_effect": self.env_effect,
        }


@dataclass
class EpisodeResult:
    """Structured terminal result and full execution evidence for one episode."""

    case_id: str
    agent_name: str
    status: EpisodeStatus
    success: bool
    shots_taken: int
    shot_budget: int
    total_ship_cells: int
    hits: int
    misses: int
    sunk_ships: list[str]
    hit_rate: float
    mean_step_latency_ms: float
    max_step_latency_ms: float
    mean_entropy_reduction_bits: float
    rejuvenation_events: int
    repair_events: int
    exact_fallback_step: int | None
    error_message: str | None = None
    config_summary: dict[str, object] = field(default_factory=dict)
    final_env_state: dict[str, object] = field(default_factory=dict)
    steps: list[StepTrace] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "agent_name": self.agent_name,
            "status": self.status.value,
            "success": self.success,
            "shots_taken": self.shots_taken,
            "shot_budget": self.shot_budget,
            "total_ship_cells": self.total_ship_cells,
            "hits": self.hits,
            "misses": self.misses,
            "sunk_ships": self.sunk_ships,
            "hit_rate": round(self.hit_rate, 4),
            "mean_step_latency_ms": round(self.mean_step_latency_ms, 3),
            "max_step_latency_ms": round(self.max_step_latency_ms, 3),
            "mean_entropy_reduction_bits": round(self.mean_entropy_reduction_bits, 4),
            "rejuvenation_events": self.rejuvenation_events,
            "repair_events": self.repair_events,
            "exact_fallback_step": self.exact_fallback_step,
            "error_message": self.error_message,
            "config_summary": self.config_summary,
            "final_env_state": self.final_env_state,
            "steps": [s.to_dict() for s in self.steps],
        }


def build_agent(
    agent_type: str,
    rows: int,
    cols: int,
    fleet_spec: dict[str, int],
    num_particles: int = 250,
    mcmc_sweeps: int = 3,
    exact_threshold: int = 4000,
    agent_seed: int = 4300,
) -> BattleshipAgent:
    """Instantiate one of the supported Battleship agents."""
    if agent_type == "smc":
        return ParticleFilterAgent(
            rows=rows,
            cols=cols,
            fleet_spec=fleet_spec,
            num_particles=num_particles,
            mcmc_sweeps=mcmc_sweeps,
            exact_threshold=exact_threshold,
            seed=agent_seed,
        )
    if agent_type == "hunt_target":
        return HuntAndTargetAgent(
            rows=rows,
            cols=cols,
            fleet_spec=fleet_spec,
            seed=agent_seed,
        )
    if agent_type == "random":
        return RandomAgent(
            rows=rows,
            cols=cols,
            fleet_spec=fleet_spec,
            seed=agent_seed,
        )
    raise InvalidConfigError(
        f"Unsupported agent_type '{agent_type}'. Expected one of: smc, hunt_target, random."
    )


def run_episode(
    config: EnvConfig,
    agent_type: str = "smc",
    case_id: str = "ad_hoc",
    num_particles: int = 250,
    mcmc_sweeps: int = 3,
    exact_threshold: int = 4000,
    agent_seed: int = 4300,
    percept_hook: PerceptHook | None = None,
    agent_override: BattleshipAgent | None = None,
    verbose: bool = False,
    show_board: bool = False,
) -> EpisodeResult:
    """Run one complete Battleship episode through the agent-environment vertical slice."""
    config_summary: dict[str, object] = {
        "rows": config.rows,
        "cols": config.cols,
        "fleet_spec": dict(config.fleet_spec),
        "shot_budget": config.shot_budget,
        "env_seed": config.seed,
        "agent_type": agent_override.name if agent_override is not None else agent_type,
        "agent_seed": agent_seed,
        "num_particles": num_particles if agent_type == "smc" and agent_override is None else None,
        "has_fixed_ships": config.fixed_ships is not None,
    }

    try:
        env = BattleshipEnv(config)
        agent = (
            agent_override
            if agent_override is not None
            else build_agent(
                agent_type=agent_type,
                rows=env.rows,
                cols=env.cols,
                fleet_spec=env.fleet_spec,
                num_particles=num_particles,
                mcmc_sweeps=mcmc_sweeps,
                exact_threshold=exact_threshold,
                agent_seed=agent_seed,
            )
        )
    except InvalidConfigError as exc:
        return EpisodeResult(
            case_id=case_id,
            agent_name=agent_type,
            status=EpisodeStatus.INVALID_CONFIG,
            success=False,
            shots_taken=0,
            shot_budget=config.shot_budget if isinstance(config.shot_budget, int) else 0,
            total_ship_cells=sum(
                v for v in config.fleet_spec.values() if isinstance(v, int) and v > 0
            )
            if isinstance(config.fleet_spec, dict)
            else 0,
            hits=0,
            misses=0,
            sunk_ships=[],
            hit_rate=0.0,
            mean_step_latency_ms=0.0,
            max_step_latency_ms=0.0,
            mean_entropy_reduction_bits=0.0,
            rejuvenation_events=0,
            repair_events=0,
            exact_fallback_step=None,
            error_message=str(exc),
            config_summary=config_summary,
            final_env_state={"status": EpisodeStatus.INVALID_CONFIG.value},
            steps=[],
        )
    except BeliefCollapseError as exc:
        return EpisodeResult(
            case_id=case_id,
            agent_name=agent_type,
            status=EpisodeStatus.BELIEF_COLLAPSE,
            success=False,
            shots_taken=0,
            shot_budget=config.shot_budget,
            total_ship_cells=sum(config.fleet_spec.values()),
            hits=0,
            misses=0,
            sunk_ships=[],
            hit_rate=0.0,
            mean_step_latency_ms=0.0,
            max_step_latency_ms=0.0,
            mean_entropy_reduction_bits=0.0,
            rejuvenation_events=0,
            repair_events=0,
            exact_fallback_step=None,
            error_message=str(exc),
            config_summary=config_summary,
            final_env_state={"status": EpisodeStatus.BELIEF_COLLAPSE.value},
            steps=[],
        )

    steps: list[StepTrace] = []
    latencies_ms: list[float] = []
    entropy_reductions: list[float] = []
    rejuvenation_events = 0
    repair_events = 0
    exact_fallback_step: int | None = (
        0 if agent.inspect_belief().get("mode") == "EXACT_FALLBACK" else None
    )
    terminal_status: EpisodeStatus = env.status
    error_message: str | None = None

    if verbose:
        init_belief = agent.inspect_belief()
        print(
            f"=== Battleship Vertical Slice [{case_id}] | "
            f"{env.rows}x{env.cols} | agent={agent.name} | budget={env.shot_budget} | "
            f"init_mode={init_belief.get('mode')} | particles={init_belief.get('particle_count', 0)} ==="
        )

    step = 0
    while env.status == EpisodeStatus.RUNNING:
        step += 1
        avail_count = len(env.available_actions())

        try:
            decision = agent.select_action()
        except BeliefCollapseError as exc:
            terminal_status = EpisodeStatus.BELIEF_COLLAPSE
            error_message = str(exc)
            break

        try:
            raw_percept = env.probe(decision.coord)
        except InvalidActionError as exc:
            terminal_status = EpisodeStatus.INVALID_ACTION
            error_message = str(exc)
            break

        percept = (
            percept_hook(step, raw_percept)
            if percept_hook is not None
            else raw_percept
        )

        try:
            update = agent.observe(percept)
        except BeliefCollapseError as exc:
            terminal_status = EpisodeStatus.BELIEF_COLLAPSE
            error_message = str(exc)
            env_snapshot = env.inspect_state(reveal_ships=False)
            steps.append(
                StepTrace(
                    step=step,
                    available_actions_count=avail_count,
                    belief_before=decision.belief_snapshot,
                    top_candidates=[c.to_dict() for c in decision.top_candidates],
                    chosen_action=[decision.coord[0], decision.coord[1]],
                    chosen_prob=decision.probability,
                    chosen_cell_entropy_bits=decision.cell_entropy_bits,
                    decision_time_ms=decision.decision_time_ms,
                    percept=percept.to_dict(),
                    belief_after={"error": str(exc), "particle_count": 0},
                    env_effect=env_snapshot,
                )
            )
            latencies_ms.append(decision.decision_time_ms)
            if verbose:
                print(
                    f"Step {step:02d} | Probe {decision.coord} | Percept={percept} | "
                    f"BELIEF_COLLAPSE: {exc}"
                )
            break

        total_step_ms = decision.decision_time_ms + update.update_time_ms
        latencies_ms.append(total_step_ms)
        entropy_reductions.append(update.entropy_reduction_bits)
        if update.rejuvenated:
            rejuvenation_events += 1
        if update.repair_triggered:
            repair_events += 1
        if exact_fallback_step is None and update.mode == "EXACT_FALLBACK":
            exact_fallback_step = step

        env_snapshot = env.inspect_state(reveal_ships=False)
        steps.append(
            StepTrace(
                step=step,
                available_actions_count=avail_count,
                belief_before=decision.belief_snapshot,
                top_candidates=[c.to_dict() for c in decision.top_candidates],
                chosen_action=[decision.coord[0], decision.coord[1]],
                chosen_prob=decision.probability,
                chosen_cell_entropy_bits=decision.cell_entropy_bits,
                decision_time_ms=decision.decision_time_ms,
                percept=percept.to_dict(),
                belief_after=update.to_dict(),
                env_effect=env_snapshot,
            )
        )

        if verbose:
            outcome_label = (
                f"HIT (SUNK {percept.sunk_ship}!)"
                if percept.sunk_ship
                else ("HIT" if percept.hit else "MISS")
            )
            flags = []
            if update.repair_triggered:
                flags.append("REPAIR")
            elif update.rejuvenated:
                flags.append("MCMC")
            if update.mode == "EXACT_FALLBACK":
                flags.append("EXACT")
            flag_str = ",".join(flags) if flags else "KEEP"
            print(
                f"Step {step:02d} | Probe ({decision.coord[0]},{decision.coord[1]}) | "
                f"P={decision.probability:.3f} | H_cell={decision.cell_entropy_bits:.3f} | "
                f"{outcome_label:<22} | surv={update.survived_particles:3d}->"
                f"{update.particle_count:4d} ({update.unique_particles:4d} uniq) | "
                f"H_board={update.board_entropy_bits:5.2f} | {total_step_ms:6.2f}ms [{flag_str}]"
            )

        terminal_status = env.status

    if verbose and show_board:
        print("\n--- Final Board (X=hit, o=miss, S=unsunk ship) ---")
        print(env.render_board(reveal_ships=True))

    shots_taken = env.shots_count
    hits = len(env.hits)
    misses = len(env.misses)
    hit_rate = (hits / shots_taken) if shots_taken > 0 else 0.0
    mean_latency = statistics.fmean(latencies_ms) if latencies_ms else 0.0
    max_latency = max(latencies_ms) if latencies_ms else 0.0
    mean_ent_red = (
        statistics.fmean(entropy_reductions) if entropy_reductions else 0.0
    )

    result = EpisodeResult(
        case_id=case_id,
        agent_name=agent.name,
        status=terminal_status,
        success=(terminal_status == EpisodeStatus.VICTORY),
        shots_taken=shots_taken,
        shot_budget=env.shot_budget,
        total_ship_cells=env.total_ship_cells,
        hits=hits,
        misses=misses,
        sunk_ships=list(env.sunk_ships),
        hit_rate=hit_rate,
        mean_step_latency_ms=mean_latency,
        max_step_latency_ms=max_latency,
        mean_entropy_reduction_bits=mean_ent_red,
        rejuvenation_events=rejuvenation_events,
        repair_events=repair_events,
        exact_fallback_step=exact_fallback_step,
        error_message=error_message,
        config_summary=config_summary,
        final_env_state=env.inspect_state(reveal_ships=True),
        steps=steps,
    )

    if verbose:
        print(
            f"\nTerminal Status: {result.status.value} | Success={result.success} | "
            f"Shots={result.shots_taken}/{result.shot_budget} | "
            f"Hits={result.hits}/{result.total_ship_cells} | "
            f"Mean Latency={result.mean_step_latency_ms:.2f} ms | "
            f"Rejuvenations={result.rejuvenation_events} (Repairs={result.repair_events}) | "
            f"Exact Fallback Step={result.exact_fallback_step}"
        )

    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Battleship Bayesian Belief-Tracking Vertical Slice."
    )
    parser.add_argument("--rows", type=int, default=10, help="Board rows (default: 10)")
    parser.add_argument("--cols", type=int, default=10, help="Board cols (default: 10)")
    parser.add_argument(
        "--fleet",
        choices=["standard", "poc"],
        default="standard",
        help="Fleet manifest: standard (5 ships, 17 cells) or poc (2 ships, 5 cells)",
    )
    parser.add_argument(
        "--budget", type=int, default=75, help="Maximum shot budget B_max (default: 75)"
    )
    parser.add_argument(
        "--seed", type=int, default=4300, help="Environment layout RNG seed (default: 4300)"
    )
    parser.add_argument(
        "--agent-seed",
        type=int,
        default=4300,
        help="Agent RNG seed for deterministic SMC/MCMC (default: 4300)",
    )
    parser.add_argument(
        "--agent",
        choices=["smc", "hunt_target", "random"],
        default="smc",
        help="Agent strategy (default: smc)",
    )
    parser.add_argument(
        "--particles",
        type=int,
        default=250,
        help="Number of SMC particles N (default: 250)",
    )
    parser.add_argument(
        "--mcmc-sweeps",
        type=int,
        default=3,
        help="Number of MCMC Gibbs sweeps per rejuvenation (default: 3)",
    )
    parser.add_argument(
        "--trace-out",
        type=Path,
        default=None,
        help="Optional path to save full JSON execution trace",
    )
    parser.add_argument(
        "--show-board",
        action="store_true",
        help="Print final ASCII board revealing ship positions",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    fleet = dict(STANDARD_FLEET) if args.fleet == "standard" else dict(POC_FLEET)
    config = EnvConfig(
        rows=args.rows,
        cols=args.cols,
        fleet_spec=fleet,
        shot_budget=args.budget,
        seed=args.seed,
    )
    result = run_episode(
        config=config,
        agent_type=args.agent,
        case_id=f"cli_{args.agent}_{args.rows}x{args.cols}_seed{args.seed}",
        num_particles=args.particles,
        mcmc_sweeps=args.mcmc_sweeps,
        agent_seed=args.agent_seed,
        verbose=True,
        show_board=args.show_board,
    )
    if args.trace_out is not None:
        args.trace_out.parent.mkdir(parents=True, exist_ok=True)
        args.trace_out.write_text(
            json.dumps(result.to_dict(), indent=2), encoding="utf-8"
        )
        print(f"Saved trace to {args.trace_out}")
    return 0 if result.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
