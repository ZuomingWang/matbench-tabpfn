"""Verify reported results against committed repository evidence.

The script uses only the Python standard library so it can audit committed CSV
files before the scientific environment is installed. Primary, MLP, and
notebook-derived extension checks always run. Full small-data and
active-learning source summary-table checks run when their directories are
supplied, or are discovered in strict mode.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

try:
    from .extract_colab_notebook_evidence import verify_artifacts
except ImportError:
    from extract_colab_notebook_evidence import verify_artifacts


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SMALL_NOTEBOOK_EVIDENCE = (
    PROJECT_ROOT
    / "results"
    / "diagnostics"
    / "small_data"
    / "tables"
    / "notebook_summary.csv"
)
ACTIVE_NOTEBOOK_EVIDENCE = (
    PROJECT_ROOT
    / "results"
    / "diagnostics"
    / "active_learning"
    / "tables"
    / "notebook_summary.csv"
)


@dataclass(frozen=True)
class Check:
    label: str
    expected: float
    actual: float
    tolerance: float

    @property
    def passed(self) -> bool:
        return math.isclose(
            self.actual,
            self.expected,
            rel_tol=0.0,
            abs_tol=self.tolerance,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare reported values with committed result tables."
    )
    parser.add_argument(
        "--small-data-run",
        type=Path,
        default=None,
        help="Path to one results/small_data_diagnostics/<run_id> directory.",
    )
    parser.add_argument(
        "--active-learning-run",
        type=Path,
        action="append",
        default=[],
        help=(
            "Path to an active-learning run directory. Repeat for the four "
            "task/representation runs used by the reference result set."
        ),
    )
    parser.add_argument(
        "--active-feature-set",
        default="magpie_structure_all",
        help=(
            "Reference feature set (currently fixed to magpie_structure_all "
            "because the expected values are condition-specific)."
        ),
    )
    parser.add_argument(
        "--active-initial-fraction",
        type=float,
        default=0.1,
        help=(
            "Reference initial labeled fraction (currently fixed to 0.1 "
            "because the expected values are condition-specific)."
        ),
    )
    parser.add_argument(
        "--require-extensions",
        action="store_true",
        help=(
            "Fail if the small-data source summary tables or active-learning "
            "aggregate source tables are unavailable. Committed notebook-display "
            "evidence is not a substitute for this stricter summary-table check."
        ),
    )
    args = parser.parse_args()
    if args.active_feature_set != "magpie_structure_all":
        parser.error(
            "--active-feature-set must be magpie_structure_all for the "
            "reference result check."
        )
    if not math.isclose(
        args.active_initial_fraction,
        0.1,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        parser.error(
            "--active-initial-fraction must be 0.1 for the "
            "reference result check."
        )
    return args


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _select_one(
    rows: Iterable[dict[str, str]],
    *,
    source: Path,
    **criteria: object,
) -> dict[str, str]:
    matches = [
        row
        for row in rows
        if all(str(row.get(key)) == str(value) for key, value in criteria.items())
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one row in {source} for {criteria}, found {len(matches)}."
        )
    return matches[0]


def _check(
    checks: list[Check],
    label: str,
    row: dict[str, str],
    column: str,
    expected: float,
    tolerance: float,
    transform: Callable[[float], float] | None = None,
) -> None:
    actual = float(row[column])
    if transform is not None:
        actual = transform(actual)
    checks.append(Check(label, expected, actual, tolerance))


def _committed_checks() -> list[Check]:
    checks: list[Check] = []
    primary_path = PROJECT_ROOT / "results" / "metrics" / "model_summary.csv"
    primary = _read_rows(primary_path)

    primary_expected = [
        ("Steels TabPFN MAE", "matbench_steels", "magpie", "tabpfn", 87.6508603676),
        ("Steels ExtraTrees MAE", "matbench_steels", "magpie", "extra_trees", 90.7758334972),
        ("Band-gap TabPFN MAE", "matbench_expt_gap", "magpie", "tabpfn", 0.3688042396),
        ("Band-gap ExtraTrees MAE", "matbench_expt_gap", "magpie", "extra_trees", 0.4083439984),
        (
            "JDFT2D structure TabPFN MAE",
            "matbench_jdft2d",
            "magpie_structure_all",
            "tabpfn",
            34.3959220379,
        ),
        (
            "JDFT2D structure ExtraTrees MAE",
            "matbench_jdft2d",
            "magpie_structure_all",
            "extra_trees",
            40.5434730835,
        ),
        (
            "Phonons structure TabPFN MAE",
            "matbench_phonons",
            "magpie_structure_all",
            "tabpfn",
            30.6309780631,
        ),
        (
            "Phonons best ExtraTrees MAE",
            "matbench_phonons",
            "magpie_density_packing",
            "extra_trees",
            44.2270274289,
        ),
        (
            "JDFT2D composition-proxy TabPFN MAE",
            "matbench_jdft2d",
            "magpie",
            "tabpfn",
            46.1301388444,
        ),
        (
            "Phonons composition-proxy TabPFN MAE",
            "matbench_phonons",
            "magpie",
            "tabpfn",
            36.8531542230,
        ),
    ]
    for label, task, feature_set, model, expected in primary_expected:
        row = _select_one(
            primary,
            source=primary_path,
            task=task,
            feature_set=feature_set,
            model=model,
        )
        _check(checks, label, row, "mean_mae", expected, 1e-8)

    mlp_dir = PROJECT_ROOT / "results" / "diagnostics" / "mlp" / "tables"
    mlp_specs = [
        ("mlp_summary.csv", "mlp", "MLP"),
        ("tuned_mlp_summary.csv", "tuned_mlp", "Tuned MLP"),
    ]
    expected_mlp = {
        ("mlp", "matbench_jdft2d", "magpie"): 63.3266141246,
        ("mlp", "matbench_jdft2d", "magpie_structure_all"): 55.3605199479,
        ("mlp", "matbench_phonons", "magpie"): 96.5438880017,
        ("mlp", "matbench_phonons", "magpie_structure_all"): 77.3539563709,
        ("tuned_mlp", "matbench_jdft2d", "magpie"): 64.0055257564,
        ("tuned_mlp", "matbench_jdft2d", "magpie_structure_all"): 55.1944164127,
        ("tuned_mlp", "matbench_phonons", "magpie"): 101.1934002951,
        ("tuned_mlp", "matbench_phonons", "magpie_structure_all"): 73.9934761983,
    }
    for filename, model, display in mlp_specs:
        path = mlp_dir / filename
        rows = _read_rows(path)
        for (expected_model, task, feature_set), expected in expected_mlp.items():
            if expected_model != model:
                continue
            row = _select_one(
                rows,
                source=path,
                task=task,
                feature_set=feature_set,
                model=model,
            )
            _check(
                checks,
                f"{task} {display} {feature_set} MAE",
                row,
                "mean_mae",
                expected,
                1e-8,
            )
    return checks


def _latest_run(base: Path) -> Path | None:
    candidates = sorted(path for path in base.glob("*") if path.is_dir())
    return candidates[-1] if candidates else None


def _small_notebook_evidence_checks(path: Path) -> list[Check]:
    rows = _read_rows(path)
    checks: list[Check] = []
    expected_aulc = {
        "matbench_jdft2d": 41.53,
        "matbench_phonons": 48.49,
    }
    for task, expected in expected_aulc.items():
        row = _select_one(
            rows,
            source=path,
            claim_scope="data_efficiency",
            task=task,
            model="tabpfn",
            feature_set="magpie_structure_all",
            metric="aulc_mean_mae",
            condition="fractions_0.1_to_1.0",
        )
        _check(
            checks,
            f"{task} notebook-displayed TabPFN structure AULC",
            row,
            "value",
            expected,
            0.0051,
        )

    expected_high = {
        ("matbench_jdft2d", "extra_trees"): -15.91,
        ("matbench_jdft2d", "tabpfn"): -18.25,
        ("matbench_phonons", "extra_trees"): -11.21,
        ("matbench_phonons", "tabpfn"): -17.11,
    }
    for (task, model), expected in expected_high.items():
        row = _select_one(
            rows,
            source=path,
            claim_scope="target_regime",
            task=task,
            model=model,
            feature_set="structure_minus_composition",
            metric="mae_pct_change",
            condition="high_target",
        )
        _check(
            checks,
            f"{task} {model} notebook-displayed high-target MAE change",
            row,
            "value",
            expected,
            0.0051,
        )
    return checks


def _small_data_checks(run_dir: Path) -> list[Check]:
    checks: list[Check] = []
    aulc_path = run_dir / "tables" / "aulc_data_efficiency_summary.csv"
    high_path = run_dir / "tables" / "high_target_structure_improvement.csv"
    aulc_rows = _read_rows(aulc_path)
    high_rows = _read_rows(high_path)

    expected_aulc = {
        ("matbench_jdft2d", "tabpfn", "magpie_structure_all"): 41.5,
        ("matbench_jdft2d", "extra_trees", "magpie_structure_all"): 47.0,
        ("matbench_jdft2d", "tabpfn", "magpie"): 51.6,
        ("matbench_jdft2d", "extra_trees", "magpie"): 55.3,
        ("matbench_phonons", "tabpfn", "magpie_structure_all"): 48.5,
        ("matbench_phonons", "extra_trees", "magpie_structure_all"): 57.0,
        ("matbench_phonons", "tabpfn", "magpie"): 69.3,
        ("matbench_phonons", "extra_trees", "magpie"): 75.8,
    }
    for (task, model, feature_set), expected in expected_aulc.items():
        row = _select_one(
            aulc_rows,
            source=aulc_path,
            task=task,
            model=model,
            feature_set=feature_set,
        )
        _check(
            checks,
            f"{task} {model} {feature_set} AULC",
            row,
            "aulc_mean_mae",
            expected,
            0.15,
        )

    expected_high = {
        ("matbench_jdft2d", "extra_trees"): 15.9,
        ("matbench_jdft2d", "tabpfn"): 18.3,
        ("matbench_phonons", "extra_trees"): 11.2,
        ("matbench_phonons", "tabpfn"): 17.1,
    }
    for (task, model), expected in expected_high.items():
        row = _select_one(
            high_rows,
            source=high_path,
            task=task,
            model=model,
        )
        _check(
            checks,
            f"{task} {model} high-target MAE reduction",
            row,
            "mae_pct_change_structure_minus_composition",
            expected,
            0.15,
            transform=lambda value: -value,
        )
    return checks


def _active_learning_row_checks(
    rows: list[dict[str, str]],
    *,
    source: Path,
    feature_set: str,
    initial_fraction: float,
    variant: str | None = None,
    top_fraction: float | None = None,
    label_prefix: str = "",
    hit_tolerance: float = 0.015,
    regret_tolerance: float = 0.0015,
) -> list[Check]:
    checks: list[Check] = []
    expected = {
        ("matbench_jdft2d", "random"): (0.29, 0.244),
        ("matbench_jdft2d", "extra_trees_greedy"): (0.84, 0.212),
        ("matbench_jdft2d", "extra_trees_ucb"): (0.86, 0.263),
        ("matbench_jdft2d", "tabpfn_greedy"): (0.87, 0.283),
        ("matbench_jdft2d", "tabpfn_disagreement_ucb"): (0.86, 0.263),
        ("matbench_phonons", "random"): (0.21, 0.106),
        ("matbench_phonons", "extra_trees_greedy"): (0.98, 0.000),
        ("matbench_phonons", "extra_trees_ucb"): (0.98, 0.000),
        ("matbench_phonons", "tabpfn_greedy"): (0.97, 0.000),
        ("matbench_phonons", "tabpfn_disagreement_ucb"): (0.98, 0.000),
    }
    for (task, strategy), (hit_rate, regret) in expected.items():
        matching = [
            row
            for row in rows
            if row.get("task") == task
            and row.get("strategy") == strategy
            and row.get("feature_set") == feature_set
            and math.isclose(
                float(row.get("initial_fraction", "nan")),
                initial_fraction,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            and (variant is None or row.get("variant") == variant)
            and (
                top_fraction is None
                or math.isclose(
                    float(row.get("top_fraction", "nan")),
                    top_fraction,
                    rel_tol=0.0,
                    abs_tol=1e-12,
                )
            )
        ]
        if len(matching) != 1:
            raise ValueError(
                "Expected exactly one active-learning row for "
                f"{task}, {strategy}, {feature_set}, init={initial_fraction}; "
                f"found {len(matching)} in {source}."
            )
        row = matching[0]
        _check(
            checks,
            f"{label_prefix}{task} {strategy} final top-candidate hit rate",
            row,
            "final_top_fraction_hit_rate_mean",
            hit_rate,
            hit_tolerance,
        )
        _check(
            checks,
            f"{label_prefix}{task} {strategy} final regret",
            row,
            "final_objective_regret_fraction_mean",
            regret,
            regret_tolerance,
        )
    return checks


def _active_learning_checks(
    run_dirs: list[Path],
    *,
    feature_set: str,
    initial_fraction: float,
) -> list[Check]:
    rows: list[dict[str, str]] = []
    sources: list[Path] = []
    for run_dir in run_dirs:
        path = run_dir / "tables" / "active_learning_aggregate_summary.csv"
        rows.extend(_read_rows(path))
        sources.append(path)
    source = Path(", ".join(str(path) for path in sources))
    return _active_learning_row_checks(
        rows,
        source=source,
        feature_set=feature_set,
        initial_fraction=initial_fraction,
        label_prefix="full-run ",
    )


def _active_notebook_evidence_checks(path: Path) -> list[Check]:
    return _active_learning_row_checks(
        _read_rows(path),
        source=path,
        feature_set="magpie_structure_all",
        initial_fraction=0.1,
        variant="standard_top5pct",
        top_fraction=0.05,
        label_prefix="notebook-displayed ",
        hit_tolerance=0.0051,
        regret_tolerance=0.00051,
    )


def _print_checks(checks: Iterable[Check]) -> int:
    failures = 0
    for check in checks:
        status = "PASS" if check.passed else "FAIL"
        print(
            f"{status:4}  {check.label}: actual={check.actual:.12g}, "
            f"expected={check.expected:.12g}, tolerance={check.tolerance:g}"
        )
        failures += not check.passed
    return failures


def main() -> None:
    args = _parse_args()
    checks = _committed_checks()
    evidence_mismatches = verify_artifacts()
    if evidence_mismatches:
        for mismatch in evidence_mismatches:
            print(f"FAIL  Notebook evidence {mismatch}")
        raise SystemExit(1)
    print("PASS  Committed notebook evidence matches the executed notebook outputs.")
    checks.extend(_small_notebook_evidence_checks(SMALL_NOTEBOOK_EVIDENCE))
    checks.extend(_active_notebook_evidence_checks(ACTIVE_NOTEBOOK_EVIDENCE))
    missing_extensions: list[str] = []

    small_run = args.small_data_run
    if small_run is None and args.require_extensions:
        small_run = _latest_run(PROJECT_ROOT / "results" / "small_data_diagnostics")
    if small_run is None:
        missing_extensions.append("small-data")
    else:
        checks.extend(_small_data_checks(small_run.resolve()))

    active_runs = [path.resolve() for path in args.active_learning_run]
    if not active_runs and args.require_extensions:
        active_base = PROJECT_ROOT / "results" / "active_learning_screening"
        discovered = sorted(path for path in active_base.glob("*") if path.is_dir())
        active_runs = discovered
    if not active_runs:
        missing_extensions.append("active-learning")
    else:
        checks.extend(
            _active_learning_checks(
                active_runs,
                feature_set=args.active_feature_set,
                initial_fraction=args.active_initial_fraction,
            )
        )

    failures = _print_checks(checks)
    if missing_extensions:
        if args.require_extensions:
            message = "Missing extension source summary tables"
        else:
            message = "Extension source summary tables not supplied or checked"
        print(
            f"SKIP  {message}: "
            + ", ".join(missing_extensions)
            + ". Notebook-display evidence was checked separately."
        )
    print(
        f"\nChecked {len(checks)} values: "
        f"{len(checks) - failures} passed, {failures} failed."
    )

    if failures:
        raise SystemExit(1)
    if missing_extensions and args.require_extensions:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
