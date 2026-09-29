"""Tests for the non-interactive experiment runner."""

import csv
import subprocess
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import run_experiments


SUCCESS_OUTPUT = """progress text
result=ORANGE reason=max_plies plies=40 orange_utility=0 black_utility=0 orange_time=1.250000s black_time=0.500000s
agent=orange name='Orange Test Agent' generated=12 evaluated=10 maximum_depth=2 model_calls=2 fallback_calls=1 cache_hits=3
agent=black name='Black Test Agent' generated=8 evaluated=6 maximum_depth=1 model_calls=4 fallback_calls=2 cache_hits=1
"""


class ExperimentRunnerTests(unittest.TestCase):
    def result_row(self, match, **changes):
        row = {field: "" for field in run_experiments.FIELDNAMES}
        row.update(match)
        row.update(status="ok", result="DRAW")
        row.update(changes)
        return row

    def test_a1_matrix_has_three_evaluators_and_both_colors(self) -> None:
        matrix = run_experiments.build_matrix("a1")

        self.assertEqual(len(matrix), 12)
        self.assertEqual(
            {
                (row["focal_agent"], row["orange_seed"], row["focal_color"])
                for row in matrix
            },
            {
                (agent, seed, color)
                for agent in ("minimax_1", "minimax_2", "minimax_3")
                for seed in (0, 1)
                for color in ("orange", "black")
            },
        )
        self.assertTrue(all(row["max_plies"] == 100 for row in matrix))
        self.assertTrue(
            all(row["orange_seed"] == row["black_seed"] for row in matrix)
        )
        self.assertTrue(
            all(row["orange_depth"] == row["black_depth"] == 1 for row in matrix)
        )

    def test_a2_runs_only_depth_two_matches(self) -> None:
        matrix = run_experiments.build_matrix("a2", "minimax_2")

        self.assertEqual(len(matrix), 4)
        self.assertEqual(
            {(row["focal_color"], row["orange_seed"]) for row in matrix},
            {(color, seed) for color in ("orange", "black") for seed in (0, 1)},
        )
        for row in matrix:
            self.assertEqual(row[f'{row["focal_color"]}_depth'], 2)
            opponent_color = "black" if row["focal_color"] == "orange" else "orange"
            self.assertEqual(row[f"{opponent_color}_agent"], "random")
            self.assertEqual(row[f"{opponent_color}_depth"], 1)
            self.assertEqual(row["orange_seed"], row["black_seed"])
            self.assertEqual(row["max_plies"], 100)

    def test_b_matrix_limits_and_live_flags(self) -> None:
        matrix = run_experiments.build_matrix("b", "minimax_3", 2)

        self.assertEqual(len(matrix), 12)
        deterministic = [row for row in matrix if row["focal_agent"] == "minimax_3"]
        evaluator = [row for row in matrix if row["focal_agent"] == "llm_evaluator"]
        direct = [row for row in matrix if row["focal_agent"] == "llm_direct"]
        self.assertTrue(all(not row["live"] for row in deterministic))
        self.assertTrue(
            all(row["live"] and row["max_llm_calls"] == 100 for row in evaluator)
        )
        self.assertTrue(all(row["live"] for row in direct))
        self.assertTrue(all(row["max_plies"] == 60 for row in matrix))
        self.assertTrue(all(row["max_think_seconds"] == 3600.0 for row in matrix))
        self.assertTrue(
            all(row["orange_seed"] == row["black_seed"] for row in matrix)
        )
        self.assertEqual(
            {
                (row["focal_agent"], row["focal_color"], row["orange_seed"])
                for row in matrix
            },
            {
                (agent, color, seed)
                for agent in ("minimax_3", "llm_evaluator", "llm_direct")
                for color in ("orange", "black")
                for seed in (0, 1)
            },
        )
        self.assertTrue(
            all(
                row[f'{row["focal_color"]}_depth']
                == (2 if row["focal_agent"] == "minimax_3" else 1)
                for row in matrix
            )
        )

    def test_c_places_direct_llm_in_both_colors(self) -> None:
        matrix = run_experiments.build_matrix("c", "minimax_1", 2)

        self.assertEqual(len(matrix), 4)
        self.assertEqual({row["focal_color"] for row in matrix}, {"orange", "black"})
        self.assertTrue(all(row["live"] for row in matrix))
        self.assertTrue(all(row["max_plies"] == 60 for row in matrix))
        self.assertEqual(
            {
                (row["focal_color"], row[f'{row["focal_color"]}_seed'])
                for row in matrix
            },
            {(color, seed) for color in ("orange", "black") for seed in (0, 1)},
        )
        self.assertTrue(
            all(
                {row["orange_agent"], row["black_agent"]}
                == {"minimax_1", "llm_direct"}
                for row in matrix
            )
        )
        for row in matrix:
            self.assertEqual(row[f'{row["focal_color"]}_depth'], 1)
            opponent_color = "black" if row["focal_color"] == "orange" else "orange"
            self.assertEqual(row[f"{opponent_color}_depth"], 2)
            self.assertEqual(row[f"{opponent_color}_seed"], 0)

    @patch("run_experiments.subprocess.run")
    def test_run_match_uses_argument_list_and_parses_metrics(self, mock_run) -> None:
        mock_run.return_value = subprocess.CompletedProcess([], 0, SUCCESS_OUTPUT, "")
        match = run_experiments.build_matrix("b", "minimax_1", 1, 7200)[4]

        row = run_experiments.run_match(match, Path("/tmp/play.py"))

        command = mock_run.call_args.args[0]
        self.assertIsInstance(command, list)
        self.assertIn("--live", command)
        self.assertEqual(command[command.index("--max-llm-calls") + 1], "100")
        self.assertEqual(command[command.index("--max-think-seconds") + 1], "7200")
        self.assertEqual(command[command.index("--max-plies") + 1], "60")
        self.assertNotIn("shell", mock_run.call_args.kwargs)
        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["orange_time"], "1.250000")
        self.assertEqual(row["orange_generated"], "12")
        self.assertEqual(row["orange_model_calls"], "2")
        self.assertEqual(row["black_generated"], "8")
        self.assertEqual(row["black_fallback_calls"], "2")

    @patch("run_experiments.subprocess.run")
    def test_failed_run_retains_output_and_available_metrics(self, mock_run) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            [],
            2,
            "agent=orange name='Partial' generated=7 maximum_depth=1\n",
            "configuration failed\n",
        )
        match = run_experiments.build_matrix("a1")[0]

        row = run_experiments.run_match(match, Path("/tmp/play.py"))

        self.assertEqual(row["status"], "failed")
        self.assertEqual(row["exit_status"], 2)
        self.assertEqual(row["error"], "configuration failed")
        self.assertEqual(row["orange_generated"], "7")
        self.assertEqual(row["orange_evaluated"], "")
        self.assertEqual(row["stderr"], "configuration failed\n")

    @patch("run_experiments.subprocess.run", side_effect=OSError("cannot execute"))
    def test_launch_failure_still_returns_a_row(self, mock_run) -> None:
        match = run_experiments.build_matrix("a2", "minimax_1")[0]

        row = run_experiments.run_match(match, Path("/tmp/play.py"))

        self.assertEqual(row["status"], "failed")
        self.assertIn("cannot execute", row["error"])
        self.assertEqual(row["result"], "")

    def test_summary_uses_focal_agents_color_perspective(self) -> None:
        orange, black = run_experiments.build_matrix("a2", "minimax_1")[:2]
        rows = [
            self.result_row(
                orange,
                result="ORANGE",
                orange_utility="1",
                black_utility="-1",
                orange_time="2",
                black_time="20",
            ),
            self.result_row(
                black,
                result="ORANGE",
                orange_utility="1",
                black_utility="-1",
                orange_time="30",
                black_time="4",
            ),
        ]

        summary = run_experiments.summarize_rows(rows)[0]

        self.assertEqual(summary["wins"], 1)
        self.assertEqual(summary["losses"], 1)
        self.assertEqual(summary["mean_utility"], 0.0)
        self.assertEqual(summary["mean_think_time"], 3.0)

    def test_stage_c_summarizes_both_nonrandom_agents(self) -> None:
        orange, black = run_experiments.build_matrix("c", "minimax_2", 2)[:2]
        rows = [
            self.result_row(orange, result="ORANGE"),
            self.result_row(black, result="BLACK"),
        ]

        summaries = {
            row["agent"]: row for row in run_experiments.summarize_rows(rows)
        }

        self.assertEqual(set(summaries), {"llm_direct", "minimax_2"})
        self.assertEqual(summaries["llm_direct"]["configured_depth"], "1")
        self.assertEqual(summaries["minimax_2"]["configured_depth"], "2")
        self.assertEqual(summaries["llm_direct"]["wins"], 2)
        self.assertEqual(summaries["minimax_2"]["losses"], 2)

    def test_summary_excludes_missing_and_nonfinite_metrics(self) -> None:
        orange, black = run_experiments.build_matrix("a2", "minimax_1")[:2]
        rows = [
            self.result_row(orange, orange_generated="12", orange_evaluated="nan"),
            self.result_row(
                black,
                status="failed",
                result="",
                black_generated="",
                black_evaluated="8",
            ),
        ]

        summary = run_experiments.summarize_rows(rows)[0]

        self.assertEqual(summary["games_attempted"], 2)
        self.assertEqual(summary["games_with_results"], 1)
        self.assertEqual(summary["failures"], 1)
        self.assertEqual(summary["mean_generated"], 12.0)
        self.assertEqual(summary["generated_denominator"], 1)
        self.assertEqual(summary["mean_evaluated"], 8.0)
        self.assertEqual(summary["evaluated_denominator"], 1)
        self.assertEqual(summary["utility_denominator"], 0)

    @patch("run_experiments.subprocess.run")
    def test_main_writes_one_row_per_scheduled_match(self, mock_run) -> None:
        mock_run.return_value = subprocess.CompletedProcess([], 0, SUCCESS_OUTPUT, "")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "a2.csv"
            announcements = StringIO()
            with redirect_stdout(announcements):
                run_experiments.main(
                    ["a2", "--evaluator", "minimax_2", "--output", str(output)]
                )
            with output.open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            summary_path = Path(directory) / "a2-summary.csv"
            with summary_path.open(newline="", encoding="utf-8") as source:
                summaries = list(csv.DictReader(source))

        self.assertEqual(len(rows), 4)
        self.assertEqual(len(summaries), 1)
        self.assertIn(str(output), announcements.getvalue())
        self.assertIn(str(summary_path), announcements.getvalue())
        self.assertEqual(mock_run.call_count, 4)

    def test_main_checkpoints_raw_and_summary_before_next_match(self) -> None:
        matrix = run_experiments.build_matrix("a2", "minimax_2")
        first_row = self.result_row(matrix[0], result="ORANGE")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "a2.csv"
            summary_path = Path(directory) / "a2-summary.csv"
            with patch(
                "run_experiments.run_match",
                side_effect=(first_row, KeyboardInterrupt()),
            ):
                with self.assertRaises(KeyboardInterrupt):
                    run_experiments.main(
                        ["a2", "--evaluator", "minimax_2", "--output", str(output)]
                    )
            with output.open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            with summary_path.open(newline="", encoding="utf-8") as source:
                summaries = list(csv.DictReader(source))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "ok")
        self.assertEqual(len(summaries), 1)
        self.assertEqual(summaries[0]["games_attempted"], "1")

    @patch("run_experiments.subprocess.run")
    def test_resume_skips_success_and_replaces_failure(self, mock_run) -> None:
        mock_run.return_value = subprocess.CompletedProcess([], 0, SUCCESS_OUTPUT, "")
        matrix = run_experiments.build_matrix("a2", "minimax_2")
        successful = self.result_row(matrix[0], result="BLACK")
        failed = self.result_row(
            matrix[1], status="failed", result="", error="temporary failure"
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "a2.csv"
            run_experiments.write_csv(
                output, run_experiments.FIELDNAMES, [successful, failed]
            )
            run_experiments.main(
                [
                    "a2",
                    "--evaluator",
                    "minimax_2",
                    "--output",
                    str(output),
                    "--resume",
                ]
            )
            with output.open(newline="", encoding="utf-8") as source:
                rows = list(csv.DictReader(source))
            with run_experiments.summary_path_for(output).open(
                newline="", encoding="utf-8"
            ) as source:
                summaries = list(csv.DictReader(source))

        self.assertEqual(mock_run.call_count, 3)
        self.assertEqual(len(rows), 4)
        self.assertEqual(
            len({run_experiments.configuration_key(row) for row in rows}), 4
        )
        self.assertEqual(sum(row["status"] == "ok" for row in rows), 4)
        self.assertEqual(sum(row["error"] == "temporary failure" for row in rows), 0)
        self.assertEqual(summaries[0]["games_attempted"], "4")
        self.assertEqual(summaries[0]["failures"], "0")


if __name__ == "__main__":
    unittest.main()
