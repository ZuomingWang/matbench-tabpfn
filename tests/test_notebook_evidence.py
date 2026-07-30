"""Regression tests for deterministic notebook-evidence extraction."""

from __future__ import annotations

import csv
import io
import tempfile
import unittest
from pathlib import Path

from scripts.extract_colab_notebook_evidence import (
    DEFAULT_ACTIVE_NOTEBOOK,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_SMALL_NOTEBOOK,
    SECRET_PATTERN,
    render_artifacts,
    verify_artifacts,
)

SMALL_NOTEBOOK_EVIDENCE = (
    DEFAULT_OUTPUT_ROOT / "small_data" / "tables" / "notebook_summary.csv"
)
ACTIVE_NOTEBOOK_EVIDENCE = (
    DEFAULT_OUTPUT_ROOT / "active_learning" / "tables" / "notebook_summary.csv"
)


class NotebookEvidenceTests(unittest.TestCase):
    def test_committed_artifacts_match_notebooks(self) -> None:
        self.assertEqual(verify_artifacts(), [])

    def test_generated_artifacts_do_not_copy_secret_like_tokens(self) -> None:
        for content in render_artifacts().values():
            self.assertIsNone(SECRET_PATTERN.search(content))

    def test_expected_row_counts_and_variants(self) -> None:
        artifacts = render_artifacts()
        small_rows = list(
            csv.DictReader(io.StringIO(artifacts[SMALL_NOTEBOOK_EVIDENCE]))
        )
        active_rows = list(
            csv.DictReader(io.StringIO(artifacts[ACTIVE_NOTEBOOK_EVIDENCE]))
        )
        self.assertEqual(len(small_rows), 26)
        self.assertEqual(len(active_rows), 56)
        variants = [row["variant"] for row in active_rows]
        self.assertEqual(variants.count("standard_top5pct"), 40)
        self.assertEqual(variants.count("stress_top1pct"), 16)
        self.assertEqual(
            {row["unit"] for row in small_rows if row["task"] == "matbench_jdft2d"},
            {"meV/atom", "%", "dimensionless"},
        )
        self.assertEqual(
            {row["unit"] for row in active_rows},
            {"meV/atom", "cm^-1"},
        )
        standard_rows = [
            row for row in active_rows if row["variant"] == "standard_top5pct"
        ]
        self.assertEqual(
            sum(bool(row["found_global_best_rate"]) for row in standard_rows),
            15,
        )

    def test_check_detects_modified_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_root = Path(temp_dir)
            artifacts = render_artifacts(output_root=output_root)
            for path, content in artifacts.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
            self.assertEqual(
                verify_artifacts(
                    small_notebook=DEFAULT_SMALL_NOTEBOOK,
                    active_notebook=DEFAULT_ACTIVE_NOTEBOOK,
                    output_root=output_root,
                ),
                [],
            )
            target = (
                output_root
                / "small_data"
                / "tables"
                / "notebook_summary.csv"
            )
            target.write_text(
                target.read_text(encoding="utf-8").replace("41.53", "99.99", 1),
                encoding="utf-8",
            )
            mismatches = verify_artifacts(
                small_notebook=DEFAULT_SMALL_NOTEBOOK,
                active_notebook=DEFAULT_ACTIVE_NOTEBOOK,
                output_root=output_root,
            )
            self.assertEqual(len(mismatches), 1)
            self.assertIn("out of date", mismatches[0])


if __name__ == "__main__":
    unittest.main()
