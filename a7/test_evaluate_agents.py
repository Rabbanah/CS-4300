"""Tests for reproducible generated-suite evaluation."""

import csv
import unittest
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from evaluate_agents import (
    _bootstrap_interval,
    evaluate,
    main,
    print_summary,
    write_csv,
)


class EvaluateAgentsTests(unittest.TestCase):
    def test_evaluation_is_paired_and_reproducible(self) -> None:
        arguments = (["stop", "random"], 4300, 4, (10, 11, 12), 20, "test")

        first = evaluate(*arguments)
        second = evaluate(*arguments)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 24)
        pairs_by_agent = {
            agent: {(row.scenario, row.seed) for row in first if row.agent == agent}
            for agent in ("stop", "random")
        }
        self.assertEqual(pairs_by_agent["stop"], pairs_by_agent["random"])

    def test_csv_contains_one_row_per_episode(self) -> None:
        records = evaluate(["stop"], 4300, 3, (7, 8), 20, "abc123")
        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "results.csv"
            write_csv(records, output_path)
            with output_path.open(encoding="utf-8", newline="") as input_file:
                rows = list(csv.DictReader(input_file))

        self.assertEqual(len(rows), 6)
        self.assertEqual(rows[0]["code_version"], "abc123")
        self.assertEqual(rows[0]["status"], "stopped")
        self.assertEqual(rows[0]["unprotected_slips"], "0")

    def test_evaluate_rejects_invalid_pairing_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "agent_names"):
            evaluate([], 4300, 3, (1,), 20, "test")
        with self.assertRaisesRegex(ValueError, "must be unique"):
            evaluate(["stop", "stop"], 4300, 3, (1,), 20, "test")
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            evaluate(["stop"], 4300, 3, (), 20, "test")
        with self.assertRaisesRegex(ValueError, "must be unique"):
            evaluate(["stop"], 4300, 3, (1, 1), 20, "test")

    def test_bootstrap_interval_is_reproducible(self) -> None:
        first = _bootstrap_interval((1.0, 2.0, 3.0, 4.0), resamples=500)
        second = _bootstrap_interval((1.0, 2.0, 3.0, 4.0), resamples=500)

        self.assertEqual(first, second)
        self.assertEqual(first[0], 2.5)
        self.assertLessEqual(first[1], first[0])
        self.assertGreaterEqual(first[2], first[0])

    def test_summary_rejects_incomplete_or_duplicate_groups(self) -> None:
        records = evaluate(["stop"], 4300, 2, (7,), 20, "test")

        with self.assertRaisesRegex(ValueError, "no evaluation records"):
            print_summary(records, ["stop", "random"])
        with self.assertRaisesRegex(ValueError, "must be unique"):
            print_summary(records, ["stop", "stop"])

    def test_summary_includes_statuses_and_paired_difference(self) -> None:
        records = evaluate(["stop", "random"], 4300, 3, (7, 8), 20, "test")
        output = StringIO()

        with patch("sys.stdout", output):
            print_summary(records, ["stop", "random"])

        text = output.getvalue()
        self.assertIn("status:", text)
        self.assertIn("unprotected_slips", text)
        self.assertIn("paired scenario score differences", text)
        self.assertIn("random:", text)

    def test_main_writes_requested_output(self) -> None:
        with TemporaryDirectory() as directory:
            output_path = Path(directory) / "evaluation.csv"
            with (
                patch("evaluate_agents.repository_version", return_value="test"),
                patch("sys.stdout", StringIO()),
            ):
                status = main(
                    [
                        "stop",
                        "--scenarios",
                        "2",
                        "--agent-seeds",
                        "2",
                        "--output",
                        str(output_path),
                    ]
                )
            self.assertTrue(output_path.exists())

        self.assertEqual(status, 0)


if __name__ == "__main__":
    unittest.main()
