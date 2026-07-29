"""Regression tests for presentation-value verification semantics."""

from __future__ import annotations

import argparse
import csv
import io
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import verify_presentation_results as verifier


AULC_FIELDS = ["task", "model", "feature_set", "aulc_mean_mae"]
HIGH_FIELDS = [
    "task",
    "model",
    "mae_pct_change_structure_minus_composition",
]
ACTIVE_FIELDS = [
    "task",
    "strategy",
    "feature_set",
    "initial_fraction",
    "variant",
    "top_fraction",
    "final_top_fraction_hit_rate_mean",
    "final_objective_regret_fraction_mean",
]

AULC_ROWS = [
    ("matbench_jdft2d", "tabpfn", "magpie_structure_all", "41.5"),
    ("matbench_jdft2d", "extra_trees", "magpie_structure_all", "47.0"),
    ("matbench_jdft2d", "tabpfn", "magpie", "51.6"),
    ("matbench_jdft2d", "extra_trees", "magpie", "55.3"),
    ("matbench_phonons", "tabpfn", "magpie_structure_all", "48.5"),
    ("matbench_phonons", "extra_trees", "magpie_structure_all", "57.0"),
    ("matbench_phonons", "tabpfn", "magpie", "69.3"),
    ("matbench_phonons", "extra_trees", "magpie", "75.8"),
]
HIGH_ROWS = [
    ("matbench_jdft2d", "extra_trees", "-15.9"),
    ("matbench_jdft2d", "tabpfn", "-18.3"),
    ("matbench_phonons", "extra_trees", "-11.2"),
    ("matbench_phonons", "tabpfn", "-17.1"),
]
ACTIVE_VALUES = {
    ("matbench_jdft2d", "random"): ("0.289744", "0.243519"),
    ("matbench_jdft2d", "extra_trees_greedy"): ("0.838462", "0.212022"),
    ("matbench_jdft2d", "extra_trees_ucb"): ("0.858974", "0.262902"),
    ("matbench_jdft2d", "tabpfn_greedy"): ("0.871795", "0.282542"),
    ("matbench_jdft2d", "tabpfn_disagreement_ucb"): (
        "0.861538",
        "0.262902",
    ),
    ("matbench_phonons", "random"): ("0.207843", "0.105576"),
    ("matbench_phonons", "extra_trees_greedy"): ("0.977778", "0"),
    ("matbench_phonons", "extra_trees_ucb"): ("0.979085", "0"),
    ("matbench_phonons", "tabpfn_greedy"): ("0.973856", "0"),
    ("matbench_phonons", "tabpfn_disagreement_ucb"): ("0.983007", "0"),
}


def _write_csv(path: Path, fields: list[str], rows: list[tuple[str, ...]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(fields)
        writer.writerows(rows)


def _active_rows() -> list[dict[str, str]]:
    return [
        {
            "task": task,
            "strategy": strategy,
            "feature_set": "magpie_structure_all",
            "initial_fraction": "0.1",
            "variant": "presentation_standard",
            "top_fraction": "0.05",
            "final_top_fraction_hit_rate_mean": hit_rate,
            "final_objective_regret_fraction_mean": regret,
        }
        for (task, strategy), (hit_rate, regret) in ACTIVE_VALUES.items()
    ]


class VerifyPresentationResultsTests(unittest.TestCase):
    def test_full_run_reduction_requires_negative_source_change(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir)
            _write_csv(
                run_dir / "tables" / "aulc_data_efficiency_summary.csv",
                AULC_FIELDS,
                AULC_ROWS,
            )
            _write_csv(
                run_dir / "tables" / "high_target_structure_improvement.csv",
                HIGH_FIELDS,
                HIGH_ROWS,
            )
            self.assertTrue(all(check.passed for check in verifier._small_data_checks(run_dir)))

            wrong_sign = list(HIGH_ROWS)
            wrong_sign[0] = ("matbench_jdft2d", "extra_trees", "15.9")
            _write_csv(
                run_dir / "tables" / "high_target_structure_improvement.csv",
                HIGH_FIELDS,
                wrong_sign,
            )
            checks = verifier._small_data_checks(run_dir)
            target = next(
                check
                for check in checks
                if check.label == "matbench_jdft2d extra_trees high-target MAE reduction"
            )
            self.assertFalse(target.passed)

    def test_duplicate_active_condition_is_rejected(self) -> None:
        rows = _active_rows()
        checks = verifier._active_learning_row_checks(
            rows,
            source=Path("fixture.csv"),
            feature_set="magpie_structure_all",
            initial_fraction=0.1,
            variant="presentation_standard",
            top_fraction=0.05,
        )
        self.assertEqual(len(checks), 20)
        self.assertTrue(all(check.passed for check in checks))

        rows.append(dict(rows[0]))
        with self.assertRaisesRegex(ValueError, "found 2"):
            verifier._active_learning_row_checks(
                rows,
                source=Path("fixture.csv"),
                feature_set="magpie_structure_all",
                initial_fraction=0.1,
                variant="presentation_standard",
                top_fraction=0.05,
            )

    def test_notebook_rounding_tolerances(self) -> None:
        rows = _active_rows()
        rows[1]["final_top_fraction_hit_rate_mean"] = "0.8450"
        rows[2]["final_objective_regret_fraction_mean"] = "0.26349"
        checks = verifier._active_learning_row_checks(
            rows,
            source=Path("fixture.csv"),
            feature_set="magpie_structure_all",
            initial_fraction=0.1,
            hit_tolerance=0.0051,
            regret_tolerance=0.00051,
        )
        self.assertTrue(
            next(
                check
                for check in checks
                if "extra_trees_greedy final top-candidate" in check.label
                and "jdft2d" in check.label
            ).passed
        )
        self.assertTrue(
            next(
                check
                for check in checks
                if "extra_trees_ucb final regret" in check.label
                and "jdft2d" in check.label
            ).passed
        )

        rows[1]["final_top_fraction_hit_rate_mean"] = "0.8452"
        rows[2]["final_objective_regret_fraction_mean"] = "0.26352"
        checks = verifier._active_learning_row_checks(
            rows,
            source=Path("fixture.csv"),
            feature_set="magpie_structure_all",
            initial_fraction=0.1,
            hit_tolerance=0.0051,
            regret_tolerance=0.00051,
        )
        failing = [
            check
            for check in checks
            if (
                "jdft2d extra_trees_greedy final top-candidate" in check.label
                or "jdft2d extra_trees_ucb final regret" in check.label
            )
        ]
        self.assertEqual([check.tolerance for check in failing], [0.0051, 0.00051])
        self.assertTrue(all(not check.passed for check in failing))

    def test_non_presentation_active_selectors_are_rejected(self) -> None:
        with mock.patch.object(
            sys,
            "argv",
            ["verify", "--active-feature-set", "magpie"],
        ), mock.patch("sys.stderr", new_callable=io.StringIO):
            with self.assertRaises(SystemExit) as caught:
                verifier._parse_args()
        self.assertEqual(caught.exception.code, 2)

        with mock.patch.object(
            sys,
            "argv",
            ["verify", "--active-initial-fraction", "0.2"],
        ), mock.patch("sys.stderr", new_callable=io.StringIO):
            with self.assertRaises(SystemExit) as caught:
                verifier._parse_args()
        self.assertEqual(caught.exception.code, 2)

    def test_module_help_imports(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.verify_presentation_results",
                "--help",
            ],
            cwd=verifier.PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--require-extensions", completed.stdout)

    def test_main_exit_precedence_and_strict_missing(self) -> None:
        default_args = argparse.Namespace(
            small_data_run=None,
            active_learning_run=[],
            active_feature_set="magpie_structure_all",
            active_initial_fraction=0.1,
            require_extensions=False,
        )
        passing = verifier.Check("fixture", 1.0, 1.0, 0.0)
        failing = verifier.Check("fixture", 1.0, 2.0, 0.0)
        common_patches = (
            mock.patch.object(verifier, "_parse_args", return_value=default_args),
            mock.patch.object(verifier, "_committed_checks", return_value=[passing]),
            mock.patch.object(verifier, "verify_artifacts", return_value=[]),
            mock.patch.object(
                verifier,
                "_small_notebook_evidence_checks",
                return_value=[],
            ),
            mock.patch.object(
                verifier,
                "_active_notebook_evidence_checks",
                return_value=[],
            ),
        )
        with (
            common_patches[0],
            common_patches[1],
            common_patches[2],
            common_patches[3],
            common_patches[4],
        ):
            with mock.patch("sys.stdout", new_callable=io.StringIO) as output:
                verifier.main()
        self.assertIn("SKIP", output.getvalue())

        strict_args = argparse.Namespace(**vars(default_args))
        strict_args.require_extensions = True
        with (
            mock.patch.object(verifier, "_parse_args", return_value=strict_args),
            mock.patch.object(verifier, "_committed_checks", return_value=[passing]),
            mock.patch.object(verifier, "verify_artifacts", return_value=[]),
            mock.patch.object(
                verifier,
                "_small_notebook_evidence_checks",
                return_value=[],
            ),
            mock.patch.object(
                verifier,
                "_active_notebook_evidence_checks",
                return_value=[],
            ),
            mock.patch.object(verifier, "_latest_run", return_value=None),
            mock.patch.object(Path, "glob", return_value=iter(())),
            mock.patch("sys.stdout", new_callable=io.StringIO),
        ):
            with self.assertRaises(SystemExit) as caught:
                verifier.main()
        self.assertEqual(caught.exception.code, 2)

        explicit_args = argparse.Namespace(
            small_data_run=Path("small-run"),
            active_learning_run=[Path("active-run")],
            active_feature_set="magpie_structure_all",
            active_initial_fraction=0.1,
            require_extensions=True,
        )
        with (
            mock.patch.object(verifier, "_parse_args", return_value=explicit_args),
            mock.patch.object(verifier, "_committed_checks", return_value=[passing]),
            mock.patch.object(verifier, "verify_artifacts", return_value=[]),
            mock.patch.object(
                verifier,
                "_small_notebook_evidence_checks",
                return_value=[],
            ),
            mock.patch.object(
                verifier,
                "_active_notebook_evidence_checks",
                return_value=[],
            ),
            mock.patch.object(verifier, "_small_data_checks", return_value=[passing]),
            mock.patch.object(
                verifier,
                "_active_learning_checks",
                return_value=[passing],
            ),
            mock.patch("sys.stdout", new_callable=io.StringIO) as output,
        ):
            verifier.main()
        self.assertNotIn("SKIP", output.getvalue())

        with (
            mock.patch.object(verifier, "_parse_args", return_value=default_args),
            mock.patch.object(verifier, "_committed_checks", return_value=[failing]),
            mock.patch.object(verifier, "verify_artifacts", return_value=[]),
            mock.patch.object(
                verifier,
                "_small_notebook_evidence_checks",
                return_value=[],
            ),
            mock.patch.object(
                verifier,
                "_active_notebook_evidence_checks",
                return_value=[],
            ),
            mock.patch("sys.stdout", new_callable=io.StringIO),
        ):
            with self.assertRaises(SystemExit) as caught:
                verifier.main()
        self.assertEqual(caught.exception.code, 1)

        with (
            mock.patch.object(verifier, "_parse_args", return_value=default_args),
            mock.patch.object(verifier, "_committed_checks", return_value=[]),
            mock.patch.object(
                verifier,
                "verify_artifacts",
                return_value=["out of date: fixture"],
            ),
            mock.patch("sys.stdout", new_callable=io.StringIO),
        ):
            with self.assertRaises(SystemExit) as caught:
                verifier.main()
        self.assertEqual(caught.exception.code, 1)


if __name__ == "__main__":
    unittest.main()
