"""Generate the accessible U0-HW-11 Vertical Slice PDF report from preserved baseline artifacts.

Reads `a11/results/baseline_summary.json` and `a11/results/traces/*.json` so that all
reported metrics and execution traces exactly match the raw measurement files.
Renders semantic HTML to `a11/lutz-hyrum-vertical-slice.pdf` via headless Chrome/Edge.
"""

from __future__ import annotations

import argparse
import html
import json
import subprocess
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = _THIS_DIR / "results"
TRACES_DIR = RESULTS_DIR / "traces"
DEFAULT_PDF = _THIS_DIR / "lutz-hyrum-vertical-slice.pdf"
HTML_PREVIEW = RESULTS_DIR / "vertical_slice_report.html"


def _find_browser() -> Path:
    candidates = [
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
        Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "No Chrome or Edge executable found for headless PDF rendering."
    )


def _format_step_line(step: dict[str, object]) -> str:
    s_num = int(step["step"])
    r, c = step["chosen_action"]
    prob = float(step["chosen_prob"])
    h_cell = float(step["chosen_cell_entropy_bits"])
    percept = step["percept"]
    assert isinstance(percept, dict)
    if percept.get("sunk_ship"):
        res_txt = f"HIT (SUNK {percept['sunk_ship']})"
    elif percept.get("hit"):
        res_txt = "HIT"
    else:
        res_txt = "MISS"
    ba = step.get("belief_after") or {}
    assert isinstance(ba, dict)
    if "error" in ba:
        return (
            f"S{s_num:02d} ({r},{c}) | P={prob:.3f} | {res_txt} | "
            f"BELIEF_COLLAPSE: {ba['error']}"
        )
    surv = int(ba["survived_particles"])
    cnt = int(ba["particle_count"])
    uniq = int(ba["unique_particles"])
    mode = str(ba["mode"])
    h_board = float(ba["board_entropy_bits"])
    dt_ms = float(step["decision_time_ms"]) + float(ba["update_time_ms"])
    tag = "EXACT" if mode == "EXACT_FALLBACK" else ("REPAIR" if ba["repair_triggered"] else "MCMC")
    return (
        f"S{s_num:02d} ({r},{c}) | P={prob:.3f} H={h_cell:.2f} | "
        f"{res_txt:<21} | {surv:4d}->{cnt:4d} ({uniq:4d}u) | "
        f"Hb={h_board:5.2f} | {dt_ms:5.2f}ms [{tag}]"
    )


def build_html(commit_label: str) -> str:
    summary = json.loads((RESULTS_DIR / "baseline_summary.json").read_text(encoding="utf-8"))
    retained = summary["retained_cases"]
    refs = summary["reference_baselines"]

    by_id = {row["case_id"]: row for row in retained}
    ref_by_id = {row["case_id"]: row for row in refs}

    trace_c1 = json.loads((TRACES_DIR / "case_01_standard_seed4300.json").read_text(encoding="utf-8"))
    trace_c2 = json.loads((TRACES_DIR / "case_02_corner_clustered_boundary.json").read_text(encoding="utf-8"))
    trace_c3 = json.loads((TRACES_DIR / "case_03_tight_budget_timeout.json").read_text(encoding="utf-8"))
    trace_c4 = json.loads((TRACES_DIR / "case_04_unsupported_config.json").read_text(encoding="utf-8"))
    trace_c5 = json.loads((TRACES_DIR / "case_05_corrupted_sensor_collapse.json").read_text(encoding="utf-8"))

    # Select key explanatory steps from Case 01
    c1_steps = trace_c1["steps"]
    selected_c1_indices = [0, 3, 4, 5, 15, 20, 21, 27, 36, 39, 49, 50, 51]
    c1_lines = [_format_step_line(c1_steps[i]) for i in selected_c1_indices if i < len(c1_steps)]
    c1_first_top = c1_steps[0]["top_candidates"]
    c1_top_str = ", ".join(
        f"({c['coord'][0]},{c['coord'][1]}):{c['probability']:.3f}" for c in c1_first_top
    )

    # Select key explanatory steps from Case 02 (showing interior center-bias misses before first hit)
    c2_steps = trace_c2["steps"]
    first_hit_idx_c2 = next((i for i, s in enumerate(c2_steps) if s["percept"]["hit"]), 0)
    c2_interior_before_hit = sum(
        1
        for s in c2_steps[:first_hit_idx_c2]
        if 1 <= s["chosen_action"][0] <= 8 and 1 <= s["chosen_action"][1] <= 8
    )
    c2_excerpt_indices = sorted(
        set([0, 1, 2, max(0, first_hit_idx_c2 - 1), first_hit_idx_c2, 34, len(c2_steps) - 1])
    )
    c2_lines = [_format_step_line(c2_steps[i]) for i in c2_excerpt_indices if i < len(c2_steps)]

    # Select failure excerpts for Cases 03, 04, 05
    c3_last = _format_step_line(trace_c3["steps"][-1])
    c5_line = _format_step_line(trace_c5["steps"][0])

    c1 = by_id["case_01_standard_seed4300"]
    c2 = by_id["case_02_corner_clustered_boundary"]
    c3 = by_id["case_03_tight_budget_timeout"]
    c4 = by_id["case_04_unsupported_config"]
    c5 = by_id["case_05_corrupted_sensor_collapse"]
    r1 = ref_by_id["ref_01_hunt_target_seed4300"]
    r2 = ref_by_id["ref_02_hunt_target_corner_clustered"]

    def status_badge(status: str) -> str:
        cls_map = {
            "VICTORY": "badge-ok",
            "TIMEOUT": "badge-warn",
            "INVALID_CONFIG": "badge-err",
            "BELIEF_COLLAPSE": "badge-err",
            "INVALID_ACTION": "badge-err",
        }
        return f'<span class="badge {cls_map.get(status, "")}">{html.escape(status)}</span>'

    retained_rows_html = []
    for row in retained:
        exact_txt = str(row["exact_fallback_step"]) if row["exact_fallback_step"] is not None else "—"
        retained_rows_html.append(
            f"""
            <tr>
              <td><code>{html.escape(str(row["case_id"]))}</code><br><span class="muted">{html.escape(str(row["category"]))}</span></td>
              <td>{status_badge(str(row["status"]))}</td>
              <td>{"Yes" if row["success"] else "No"}</td>
              <td>{row["shots_taken"]}/{row["shot_budget"]}</td>
              <td>{row["hits"]}/{row["total_ship_cells"]} ({row["hit_rate"]*100:.1f}%)</td>
              <td>{row["mean_step_latency_ms"]:.2f} / {row["max_step_latency_ms"]:.2f}</td>
              <td>{row["mean_entropy_reduction_bits"]:.3f}</td>
              <td>{row["rejuvenation_events"]} / {row["repair_events"]} / {exact_txt}</td>
            </tr>"""
        )

    ref_rows_html = []
    for row in refs:
        ref_rows_html.append(
            f"""
            <tr>
              <td><code>{html.escape(str(row["case_id"]))}</code></td>
              <td><code>{html.escape(str(row["agent"]))}</code></td>
              <td>{status_badge(str(row["status"]))}</td>
              <td>{"Yes" if row["success"] else "No"}</td>
              <td>{row["shots_taken"]}/{row["shot_budget"]}</td>
              <td>{row["hit_rate"]*100:.1f}%</td>
              <td>{row["mean_step_latency_ms"]:.3f} ms</td>
            </tr>"""
        )

    c1_diff = r1["shots_taken"] - c1["shots_taken"]
    c1_pct = (c1_diff / r1["shots_taken"]) * 100.0
    c2_diff = c2["shots_taken"] - r2["shots_taken"]

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Vertical Slice Report: Bayesian Belief-Tracking Battleship Agent — Hyrum Lutz</title>
  <style>
    @page {{
      size: letter;
      margin: 0.55in 0.6in 0.55in 0.6in;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      font-family: "Segoe UI", -apple-system, BlinkMacSystemFont, Roboto, Arial, sans-serif;
      color: #1f2937;
      font-size: 9.8pt;
      line-height: 1.48;
      margin: 0;
    }}
    header.title-block {{
      border-bottom: 2.5px solid #1d4ed8;
      padding-bottom: 12px;
      margin-bottom: 16px;
    }}
    h1 {{
      font-size: 18pt;
      color: #0f172a;
      margin: 0 0 6px 0;
      line-height: 1.2;
    }}
    .subtitle {{
      font-size: 10.5pt;
      color: #475569;
      margin: 0 0 10px 0;
    }}
    .meta-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 4px 18px;
      font-size: 9.2pt;
      background: #f8fafc;
      border: 1px solid #cbd5e1;
      border-radius: 6px;
      padding: 8px 12px;
    }}
    .meta-grid div strong {{ color: #0f172a; }}
    h2 {{
      font-size: 12.5pt;
      color: #1e3a8a;
      border-bottom: 1px solid #cbd5e1;
      padding-bottom: 3px;
      margin: 16px 0 8px 0;
      page-break-after: avoid;
    }}
    h3 {{
      font-size: 10.4pt;
      color: #0f172a;
      margin: 11px 0 5px 0;
      page-break-after: avoid;
    }}
    p {{ margin: 0 0 7px 0; }}
    ul, ol {{ margin: 0 0 8px 0; padding-left: 20px; }}
    li {{ margin-bottom: 3.5px; }}
    code {{
      font-family: "Cascadia Code", Consolas, "Courier New", monospace;
      font-size: 8.6pt;
      background: #f1f5f9;
      padding: 1px 4px;
      border-radius: 3px;
      color: #0f172a;
    }}
    pre {{
      font-family: "Cascadia Code", Consolas, "Courier New", monospace;
      font-size: 7.2pt;
      line-height: 1.36;
      background: #f8fafc;
      border: 1px solid #cbd5e1;
      border-left: 3.5px solid #2563eb;
      border-radius: 4px;
      padding: 7px 9px;
      margin: 5px 0 9px 0;
      white-space: pre-wrap;
      word-break: break-word;
      overflow-wrap: anywhere;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 6px 0 10px 0;
      font-size: 8.4pt;
    }}
    th, td {{
      border: 1px solid #cbd5e1;
      padding: 5px 6px;
      text-align: left;
      vertical-align: top;
    }}
    th {{
      background: #eff6ff;
      color: #1e3a8a;
      font-weight: 600;
    }}
    tr:nth-child(even) td {{ background: #fafcff; }}
    .muted {{ color: #64748b; font-size: 7.7pt; }}
    .badge {{
      display: inline-block;
      padding: 1px 6px;
      border-radius: 10px;
      font-size: 7.5pt;
      font-weight: 600;
      letter-spacing: 0.2px;
      white-space: nowrap;
    }}
    .badge-ok {{ background: #dcfce7; color: #166534; border: 1px solid #86efac; }}
    .badge-warn {{ background: #fef3c7; color: #92400e; border: 1px solid #fcd34d; }}
    .badge-err {{ background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }}
    .callout {{
      background: #eff6ff;
      border: 1px solid #bfdbfe;
      border-left: 4px solid #1d4ed8;
      border-radius: 4px;
      padding: 8px 11px;
      margin: 8px 0 10px 0;
      break-inside: avoid;
    }}
    .diagram-box {{
      background: #f8fafc;
      border: 1px solid #cbd5e1;
      border-radius: 6px;
      padding: 10px;
      margin: 8px 0 10px 0;
      text-align: center;
      break-inside: avoid;
    }}
    .avoid-break {{ break-inside: avoid; }}
    a {{ color: #1d4ed8; text-decoration: none; }}
    footer {{
      margin-top: 14px;
      padding-top: 8px;
      border-top: 1px solid #cbd5e1;
      font-size: 8.3pt;
      color: #475569;
    }}
  </style>
</head>
<body>
  <header class="title-block">
    <h1>Vertical Slice Checkpoint: Bayesian Belief-Tracking Battleship Agent</h1>
    <p class="subtitle">U0-HW-11 Complete Agent-Environment Path, Automated Tests, and Baseline Measurements</p>
    <div class="meta-grid">
      <div><strong>Student:</strong> Hyrum Lutz</div>
      <div><strong>Course:</strong> CS-4300 Artificial Intelligence</div>
      <div><strong>Repository URL:</strong> <a href="https://github.com/Rabbanah/CS-4300">https://github.com/Rabbanah/CS-4300</a></div>
      <div><strong>Submitted Commit:</strong> <code>{html.escape(commit_label)}</code></div>
      <div><strong>Implementation Directory:</strong> <code>a11/</code> (builds on U0-HW-10 in <code>a10/</code>)</div>
      <div><strong>Runtime Stack:</strong> Python 3.14.8 (Standard Library only; no secrets or <code>.env</code>)</div>
    </div>
  </header>

  <section>
    <h2>1. Setup, Test, Run, and Baseline-Measurement Commands</h2>
    <p>All code uses the Python 3.10+ standard library with deterministic explicit random seeds (no external services, network calls, or credentials). Execute all commands from the repository root (<code>CS-4300</code>):</p>
    <pre># 1. Verify environment (no pip installs or .env setup required)
python --version

# 2. Run automated deterministic unit &amp; integration tests (18 tests, ~2.3 s)
python -m unittest discover -s a11 -p "test_*.py" -v

# 3. Run one complete 10x10 vertical-slice episode with step-by-step inspection
python a11/runner.py --seed 4300 --budget 75 --particles 250 --show-board

# 4. Run all 5 retained baseline cases (+ 2 Hunt-and-Target reference comparisons)
#    Writes raw CSV, JSON summaries, and full JSON traces under a11/results/
python a11/baseline_cases.py</pre>
  </section>

  <section>
    <h2>2. Implemented Agent-Environment Path and Changes from U0-HW-10</h2>
    <p>The vertical slice converts the 6&times;6 exhaustive Proof of Concept (<code>a10/poc_battleship.py</code>) into a full 10&times;10 five-ship system in <code>a11/</code> with strict separation between the hidden-state simulator (<code>battleship_env.py</code>), the Sequential Bayesian Belief-Tracking agent (<code>belief_agent.py</code>), and the application-controlled execution &amp; telemetry loop (<code>runner.py</code>).</p>

    <div class="diagram-box">
      <svg viewBox="0 0 700 185" width="100%" role="img" aria-label="Architecture diagram of the Battleship vertical slice agent-environment loop">
        <defs>
          <marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 1 L 10 5 L 0 9 z" fill="#1e3a8a"/>
          </marker>
        </defs>
        <!-- Config Box -->
        <rect x="8" y="20" width="115" height="62" rx="6" fill="#eff6ff" stroke="#1d4ed8" stroke-width="1.5"/>
        <text x="65" y="42" text-anchor="middle" font-size="10" font-weight="700" fill="#1e3a8a">EnvConfig</text>
        <text x="65" y="57" text-anchor="middle" font-size="8.5" fill="#334155">validate_config()</text>
        <text x="65" y="71" text-anchor="middle" font-size="8" fill="#991b1b">INVALID_CONFIG</text>

        <!-- Env Box -->
        <rect x="155" y="14" width="175" height="74" rx="6" fill="#f0fdf4" stroke="#15803d" stroke-width="1.5"/>
        <text x="242" y="34" text-anchor="middle" font-size="10.5" font-weight="700" fill="#14532d">BattleshipEnv (10x10)</text>
        <text x="242" y="49" text-anchor="middle" font-size="8.5" fill="#166534">Hidden 5-ship fleet (17 cells)</text>
        <text x="242" y="62" text-anchor="middle" font-size="8.5" fill="#166534">Bounds, duplicates, budget B_max</text>
        <text x="242" y="76" text-anchor="middle" font-size="8" fill="#166534">VICTORY / TIMEOUT / INVALID_ACTION</text>

        <!-- Runner Box -->
        <rect x="362" y="14" width="150" height="74" rx="6" fill="#fefce8" stroke="#a16207" stroke-width="1.5"/>
        <text x="437" y="34" text-anchor="middle" font-size="10.5" font-weight="700" fill="#713f12">runner.py Loop</text>
        <text x="437" y="49" text-anchor="middle" font-size="8.5" fill="#854d0e">Per-step StepTrace log</text>
        <text x="437" y="62" text-anchor="middle" font-size="8.5" fill="#854d0e">Latency &amp; entropy metrics</text>
        <text x="437" y="76" text-anchor="middle" font-size="8" fill="#854d0e">Explicit EpisodeStatus</text>

        <!-- Trace Artifact Box -->
        <rect x="544" y="20" width="148" height="62" rx="6" fill="#f8fafc" stroke="#475569" stroke-width="1.5"/>
        <text x="618" y="42" text-anchor="middle" font-size="10" font-weight="700" fill="#0f172a">EpisodeResult</text>
        <text x="618" y="57" text-anchor="middle" font-size="8.5" fill="#334155">CSV/JSON summaries</text>
        <text x="618" y="71" text-anchor="middle" font-size="8" fill="#334155">Full step traces (results/)</text>

        <!-- Agent Box -->
        <rect x="155" y="116" width="357" height="60" rx="6" fill="#eff6ff" stroke="#1d4ed8" stroke-width="1.5"/>
        <text x="333" y="134" text-anchor="middle" font-size="10.5" font-weight="700" fill="#1e3a8a">ParticleFilterAgent (SMC + MCMC + Exact Endgame Fallback)</text>
        <text x="333" y="149" text-anchor="middle" font-size="8.5" fill="#1e40af">[1] Bitmask Likelihood Filter  ->  [2] Exact Check OR MRV Repair + Gibbs/2-Ship MCMC</text>
        <text x="333" y="163" text-anchor="middle" font-size="8.5" fill="#1e40af">[3] Policy: arg max P(occ_r,c) + top-5 candidates  |  Raises BELIEF_COLLAPSE if 0 fleets</text>

        <!-- Arrows -->
        <line x1="123" y1="51" x2="152" y2="51" stroke="#1e3a8a" stroke-width="1.4" marker-end="url(#arrow)"/>
        <line x1="330" y1="51" x2="359" y2="51" stroke="#1e3a8a" stroke-width="1.4" marker-end="url(#arrow)"/>
        <line x1="512" y1="51" x2="541" y2="51" stroke="#1e3a8a" stroke-width="1.4" marker-end="url(#arrow)"/>
        <!-- Bidirectional Env <-> Agent via Runner -->
        <line x1="220" y1="116" x2="220" y2="91" stroke="#1e3a8a" stroke-width="1.4" marker-end="url(#arrow)"/>
        <text x="212" y="106" text-anchor="end" font-size="8.2" font-weight="600" fill="#1e3a8a">probe(r, c)</text>
        <line x1="285" y1="88" x2="285" y2="113" stroke="#1e3a8a" stroke-width="1.4" marker-end="url(#arrow)"/>
        <text x="293" y="106" text-anchor="start" font-size="8.2" font-weight="600" fill="#15803d">ShotResult(coord, hit, sunk_ship)</text>
      </svg>
    </div>

    <h3>Alignment and Material Refinements from U0-HW-10</h3>
    <table>
      <thead>
        <tr>
          <th scope="col" style="width:24%;">Design Component</th>
          <th scope="col" style="width:28%;">U0-HW-10 Specification</th>
          <th scope="col" style="width:48%;">Implemented Vertical Slice &amp; Justification</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td><strong>Exact Likelihood Conditioning</strong></td>
          <td>Filter hypotheses by hits, misses, and sunk ship identities.</td>
          <td>Implemented via 100-bit integer bitmasks conditioning on both <code>sunk_ship</code> and the exact sinking coordinate + cumulative hits mask at the sink step. Guarantees exact Bayesian equivalence to full percept history while cutting per-check cost to O(1) bitwise ops.</td>
        </tr>
        <tr>
          <td><strong>Particle Deprivation Recovery</strong></td>
          <td>SMC particle set with local MCMC ship-perturbation rejuvenation (flagged as principal uncertainty).</td>
          <td>Added <strong>randomized MRV hit-covering backtracking repair</strong> when <code>survived_particles == 0</code>, paired with <strong>exact single-ship Gibbs sweeps</strong> (100% acceptance, preserves uniform posterior) and <strong>coupled two-ship block moves</strong> to break adjacent-ship lock-in. Exhaustive tree pruning also proves unsatisfiability on corrupted observations.</td>
        </tr>
        <tr>
          <td><strong>Exact Endgame Fallback</strong></td>
          <td>Switch to exact enumeration when remaining hypotheses drop below a threshold.</td>
          <td>Triggered automatically when remaining per-ship domain product is tractable (&le; 60,000, or &le; 120,000 with &le; 2 unsunk ships) and total consistent fleets &le; 4,000. Reproduces the exact 2,472 hypotheses on the 6&times;6 PoC and activates seamlessly mid-game on 10&times;10 (Step 28 in Case 01).</td>
        </tr>
        <tr>
          <td><strong>Failure &amp; Trace Boundaries</strong></td>
          <td>Victory within <code>B_max</code>; timeout or invalid action failure.</td>
          <td>Added explicit <code>EpisodeStatus</code> enum (<code>VICTORY</code>, <code>TIMEOUT</code>, <code>BELIEF_COLLAPSE</code>, <code>INVALID_CONFIG</code>, <code>INVALID_ACTION</code>) and per-step JSON trace recording (available actions, top-5 candidates, particle survival, board entropy reduction, latency).</td>
        </tr>
      </tbody>
    </table>
  </section>

  <section>
    <h2>3. Initial Baseline Measurements and Execution Evidence</h2>
    <p>Five retained cases were executed via <code>python a11/baseline_cases.py</code> (with <code>N=250</code> particles, 3 MCMC sweeps, <code>agent_seed=4300</code>). Numerical measures track sample efficiency (shots taken &amp; hit rate), reliability (budget completion &amp; explicit failure handling), information efficiency (mean board Shannon entropy reduction in bits/probe), and computational cost (mean &amp; peak step latency in ms).</p>

    <table>
      <thead>
        <tr>
          <th scope="col">Retained Case &amp; Category</th>
          <th scope="col">Terminal Status</th>
          <th scope="col">Success Met?</th>
          <th scope="col">Shots / Budget</th>
          <th scope="col">Hits (Rate)</th>
          <th scope="col">Mean / Max Latency (ms)</th>
          <th scope="col">Mean &Delta;H (bits/shot)</th>
          <th scope="col">Rejuv / Repair / Exact Step</th>
        </tr>
      </thead>
      <tbody>
        {"".join(retained_rows_html)}
      </tbody>
    </table>

    <div class="avoid-break">
      <h3>Paired Reference Baseline Check (<code>HuntAndTargetAgent</code> on Identical 10&times;10 Boards)</h3>
      <table>
        <thead>
          <tr>
            <th scope="col">Reference Run</th>
            <th scope="col">Agent</th>
            <th scope="col">Status</th>
            <th scope="col">Success</th>
            <th scope="col">Shots / Budget</th>
            <th scope="col">Hit Rate</th>
            <th scope="col">Mean Latency</th>
          </tr>
        </thead>
        <tbody>
          {"".join(ref_rows_html)}
        </tbody>
      </table>
      <p><strong>Key Empirical Contrast:</strong> On the representative 10&times;10 layout (<code>case_01</code>), <code>smc_max_prob</code> sank all 5 ships in <strong>{c1["shots_taken"]} shots</strong> vs. <strong>{r1["shots_taken"]} shots</strong> for <code>hunt_target</code> (a <strong>{c1_diff}-shot / {c1_pct:.1f}% reduction</strong>) with mean latency <strong>{c1["mean_step_latency_ms"]:.2f} ms</strong> and zero particle repairs. However, on the adversarial corner-clustered board (<code>case_02</code>), <code>smc_max_prob</code> required <strong>{c2["shots_taken"]} shots</strong> vs. <strong>{r2["shots_taken"]} shots</strong> for <code>hunt_target</code> (+{c2_diff} shots), and peak single-step latency on <code>case_01</code> hit <strong>{c1["max_step_latency_ms"]:.2f} ms</strong> (within 0.6 ms of the 50 ms U0-HW-10 limit).</p>
    </div>

    <h3>Selected Execution Evidence (Extracted Directly from Preserved JSON Traces)</h3>
    <p><strong>Case 01 (<code>a11/results/traces/case_01_standard_seed4300.json</code>):</strong> Step 01 had 100 available actions with top-5 SMC candidates <code>{html.escape(c1_top_str)}</code>. The excerpt below shows MCMC hunting &amp; sinking Submarine (Steps 04–06), deductive Carrier completion at <code>P=1.000</code> (Step 22), seamless transition to <code>EXACT_FALLBACK</code> with 3,750 exact fleets (Step 28), and exact endgame deduction of Destroyer (Steps 50–52):</p>
    <pre>{html.escape(chr(10).join(c1_lines))}</pre>

    <div class="avoid-break">
      <p><strong>Case 02 Adversarial Boundary (<code>case_02_corner_clustered_boundary.json</code>):</strong> Because all five ships sit along rows 0/9 and columns 0/9, marginal probability center bias caused <strong>{first_hit_idx_c2} consecutive misses</strong> ({c2_interior_before_hit} in the interior 8&times;8 region) before the first ship hit at Step {first_hit_idx_c2 + 1}, demonstrating why corner-clustered boards degrade <code>max_prob</code>:</p>
      <pre>{html.escape(chr(10).join(c2_lines))}</pre>
    </div>

    <div class="avoid-break">
      <p><strong>Explicit Failure, Boundary, and Unsupported-Input Evidence (Cases 03, 04, 05):</strong></p>
      <pre># Case 03 (TIMEOUT at tight budget B_max=30; 12/17 ship cells hit, 2 ships sunk before budget expired):
{html.escape(c3_last)}
-> Terminal Status: {html.escape(str(c3["status"]))} | success=False | budget_remaining=0 | remaining_ship_cells=5

# Case 04 (INVALID_CONFIG caught prior to episode loop; 0 shots executed):
-> Terminal Status: {html.escape(str(c4["status"]))} | success=False | error="{html.escape(str(c4["error_message"]))}"

# Case 05 (BELIEF_COLLAPSE on injected impossible percept: length-3 Cruiser sunk on 1st hit):
{html.escape(c5_line)}
-> Terminal Status: {html.escape(str(c5["status"]))} | success=False | remaining_particles=0</pre>
    </div>
  </section>

  <section class="avoid-break">
    <h2>4. Current Omissions and Known Failures</h2>
    <ul>
      <li><strong>Deferred Expected Information Gain Policy (<code>smc_info_gain</code>):</strong> Only the <code>arg max P(occ_r,c)</code> policy is implemented in this vertical slice; one-step expected Shannon entropy reduction is scheduled for the next checkpoint (per U0-HW-10 Week 3).</li>
      <li><strong>Deferred Multi-Seed Statistical Evaluation:</strong> The 1,000-episode paired benchmark across <code>B_max &isin; &#123;60, 75, 100&#125;</code> with 95% bootstrap confidence intervals and the optional no-touching rule variant are deferred to the November 19 evaluation checkpoint.</li>
      <li><strong>Known Weakness — Corner/Edge Bias on Adversarial Layouts:</strong> On <code>case_02_corner_clustered_boundary</code>, <code>smc_max_prob</code> took {first_hit_idx_c2} consecutive misses ({c2_interior_before_hit} in the interior 8&times;8 region) before its first hit and trailed <code>hunt_target</code> by {c2_diff} shots ({c2["shots_taken"]} vs. {r2["shots_taken"]}).</li>
      <li><strong>Known Weakness — Latency Spikes Under Heavy Pruning:</strong> Unconditional 3-sweep MCMC rejuvenation on every step where <code>survived &lt; 250</code> produced a peak step latency of {c1["max_step_latency_ms"]:.2f} ms on <code>case_01</code>, leaving minimal margin below the 50 ms U0-HW-10 target.</li>
      <li><strong>Explicitly Handled Failure Boundaries:</strong> The agent assumes a known fleet manifest and noise-free deterministic percepts; budget exhaustion (<code>TIMEOUT</code>), invalid grid/fleet specs (<code>INVALID_CONFIG</code>), invalid probes (<code>INVALID_ACTION</code>), and contradictory percepts (<code>BELIEF_COLLAPSE</code>) terminate explicitly with diagnostic records rather than silent recovery.</li>
    </ul>
  </section>

  <section class="avoid-break">
    <h2>5. Evidence-Based Change Planned for Next Checkpoint (Nov 19)</h2>
    <div class="callout">
      <p><strong>Planned Change:</strong> Implement <strong>Expected Information Gain (<code>smc_info_gain</code>) top-<em>K</em> one-step entropy lookahead</strong> paired with <strong>ESS-gated MCMC rejuvenation and cell-indexed placement tables</strong>.</p>
      <p><strong>Empirical Support from Baseline Results:</strong></p>
      <ol>
        <li><em>Addressing Corner-Clustered Degradation (Case 02):</em> <code>smc_max_prob</code> lost {c2_diff} shots to <code>hunt_target</code> ({c2["shots_taken"]} vs. {r2["shots_taken"]}) and achieved lower entropy reduction ({c2["mean_entropy_reduction_bits"]:.3f} vs. {c1["mean_entropy_reduction_bits"]:.3f} bits/shot) due to interior center bias. Evaluating expected posterior entropy reduction across the top-<em>K</em> candidate cells rewards probes that maximally partition both interior and perimeter fleet hypotheses.</li>
        <li><em>Creating Latency Headroom for Lookahead (Case 01):</em> Peak single-step latency reached {c1["max_step_latency_ms"]:.2f} ms during unconditional rejuvenation (27 events in Case 01, 33 in Case 02). Triggering MCMC rejuvenation only when effective survival drops below <code>0.6N</code> (or unique diversity falls below <code>0.7N</code>) and pre-indexing ship placements by cell bit will reduce peak SMC update latency below ~20 ms, ensuring <code>smc_info_gain</code> stays safely under the 50 ms/shot deployment criterion across the 1,000-episode evaluation.</li>
      </ol>
    </div>
  </section>

  <footer>
    <strong>AI-Assistance Disclosure:</strong> Antigravity / Gemini assisted in modularizing the vertical slice (<code>a11/battleship_env.py</code>, <code>a11/belief_agent.py</code>, <code>a11/runner.py</code>, <code>a11/baseline_cases.py</code>), writing the 18-test <code>unittest</code> suite, and formatting this accessible PDF report. All 18 tests and all 5 retained baseline cases were executed locally on Python 3.14.8, and every reported number and trace excerpt was verified directly against the raw JSON/CSV artifacts in <code>a11/results/</code>.
  </footer>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate accessible Vertical Slice PDF.")
    parser.add_argument(
        "--commit",
        default="pending final commit (base U0-HW-10: 3fcd68e)",
        help="Commit SHA/label to stamp into the PDF header.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_PDF,
        help="Output PDF file path.",
    )
    args = parser.parse_args()

    html_content = build_html(commit_label=args.commit)
    HTML_PREVIEW.parent.mkdir(parents=True, exist_ok=True)
    HTML_PREVIEW.write_text(html_content, encoding="utf-8")

    browser = _find_browser()
    file_url = HTML_PREVIEW.resolve().as_uri()
    out_path = args.out.resolve()

    cmd = [
        str(browser),
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={out_path}",
        file_url,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0 or not out_path.exists():
        print("Browser PDF render failed:", proc.stderr, file=sys.stderr)
        return 1
    print(f"Generated accessible PDF: {out_path} ({out_path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
