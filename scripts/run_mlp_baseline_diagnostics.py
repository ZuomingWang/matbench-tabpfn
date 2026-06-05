"""Dense neural-network baseline for the Matbench TabPFN project.

This is a separate, optional extension from the small-data TabPFN diagnostic.
It connects the final project to the Lab3 neural-network theme by testing a
plain dense MLP on the same Magpie and structure-aware descriptor matrices.

Outputs go under:

    results/mlp_baseline_diagnostics/<run_id>/

The script imports the project's existing Matbench loading and featurization
helpers, but it does not modify the main benchmark code or result folders.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


DEFAULT_TASKS = ("matbench_jdft2d", "matbench_phonons")
DEFAULT_FEATURE_SETS = ("magpie", "magpie_structure_all")
DEFAULT_RANDOM_SEED = 42


@dataclass(frozen=True)
class MLPDiagnosticConfig:
    tasks: tuple[str, ...]
    feature_sets: tuple[str, ...]
    random_seed: int
    hidden_layers: tuple[int, ...]
    alpha: float
    learning_rate_init: float
    max_iter: int
    early_stopping: bool
    validation_fraction: float
    n_iter_no_change: int
    feature_n_jobs: int
    use_feature_cache: bool
    show_feature_progress: bool


def _parse_args() -> MLPDiagnosticConfig:
    parser = argparse.ArgumentParser(
        description="Run a standalone dense MLP baseline on selected Matbench tasks."
    )
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--feature-sets", nargs="+", default=list(DEFAULT_FEATURE_SETS))
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument(
        "--hidden-layers",
        nargs="+",
        type=int,
        default=[128, 64],
        help="Hidden layer sizes for sklearn.neural_network.MLPRegressor.",
    )
    parser.add_argument("--alpha", type=float, default=1e-4)
    parser.add_argument("--learning-rate-init", type=float, default=1e-3)
    parser.add_argument("--max-iter", type=int, default=600)
    parser.add_argument("--no-early-stopping", action="store_true")
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--n-iter-no-change", type=int, default=30)
    parser.add_argument("--feature-n-jobs", type=int, default=1)
    parser.add_argument("--no-feature-cache", action="store_true")
    parser.add_argument("--show-feature-progress", action="store_true")
    args = parser.parse_args()

    return MLPDiagnosticConfig(
        tasks=tuple(args.tasks),
        feature_sets=tuple(args.feature_sets),
        random_seed=args.random_seed,
        hidden_layers=tuple(args.hidden_layers),
        alpha=args.alpha,
        learning_rate_init=args.learning_rate_init,
        max_iter=args.max_iter,
        early_stopping=not args.no_early_stopping,
        validation_fraction=args.validation_fraction,
        n_iter_no_change=args.n_iter_no_change,
        feature_n_jobs=args.feature_n_jobs,
        use_feature_cache=not args.no_feature_cache,
        show_feature_progress=args.show_feature_progress,
    )


def _build_mlp(config: MLPDiagnosticConfig):
    from sklearn.impute import SimpleImputer
    from sklearn.neural_network import MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                MLPRegressor(
                    hidden_layer_sizes=config.hidden_layers,
                    activation="relu",
                    solver="adam",
                    alpha=config.alpha,
                    learning_rate_init=config.learning_rate_init,
                    max_iter=config.max_iter,
                    early_stopping=config.early_stopping,
                    validation_fraction=config.validation_fraction,
                    n_iter_no_change=config.n_iter_no_change,
                    random_state=config.random_seed,
                ),
            ),
        ]
    )


def _sem(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if len(values) <= 1:
        return float("nan")
    return float(values.std(ddof=1) / np.sqrt(len(values)))


def _summarize(metrics_df: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics_df.groupby(
            ["task", "feature_set", "feature_set_display", "model", "model_display"],
            dropna=False,
        )
        .agg(
            folds_completed=("fold", "nunique"),
            mean_mae=("mae", "mean"),
            std_mae=("mae", "std"),
            sem_mae=("mae", _sem),
            mean_r2=("r2", "mean"),
            std_r2=("r2", "std"),
            n_features=("n_features", "mean"),
            train_size_mean=("train_size", "mean"),
            test_size_sum=("test_size", "sum"),
            fit_seconds_sum=("fit_seconds", "sum"),
            predict_seconds_sum=("predict_seconds", "sum"),
        )
        .reset_index()
        .sort_values(["task", "mean_mae"])
    )


def _write_summary_text(summary_df: pd.DataFrame, path: Path) -> str:
    lines = [
        "# MLP Baseline Diagnostic Summary",
        "",
        "This optional extension tests whether a plain dense neural network can "
        "compete with the TabPFN/ExtraTrees workflow on the same descriptor "
        "matrices. It is mainly a Lab3 connection and a model-complexity check.",
        "",
    ]
    for task, task_df in summary_df.groupby("task", sort=False):
        best = task_df.sort_values("mean_mae").iloc[0]
        lines.append(
            "- `{}`: best MLP branch is {}, MAE={:.4g}, R2={:.4g}.".format(
                task,
                best["feature_set_display"],
                best["mean_mae"],
                best["mean_r2"],
            )
        )
    lines.append("")
    lines.append(
        "Interpretation guide: if the MLP underperforms TabPFN or tree baselines, "
        "that supports the course theme that dense neural networks are not "
        "automatically better on small, high-dimensional tabular materials data."
    )
    text = "\n".join(lines).strip() + "\n"
    path.write_text(text, encoding="utf-8")
    return text


def _plot_mlp_summary(summary_df: pd.DataFrame, figures_dir: Path) -> Path | None:
    if summary_df.empty:
        return None
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="paper")
    tasks = list(summary_df["task"].drop_duplicates())
    fig, axes = plt.subplots(len(tasks), 1, figsize=(7.8, 3.4 * len(tasks)), squeeze=False)
    for ax, task in zip(axes.reshape(-1), tasks):
        task_df = summary_df.query("task == @task").sort_values("mean_mae")
        ax.barh(
            task_df["feature_set_display"],
            task_df["mean_mae"],
            xerr=task_df["sem_mae"].fillna(0),
            color="#6b7fa8",
            alpha=0.9,
        )
        ax.invert_yaxis()
        ax.set_title(f"{task}: dense MLP baseline")
        ax.set_xlabel("Official-fold MAE")
        for idx, value in enumerate(task_df["mean_mae"]):
            ax.text(value, idx, f" {value:.3g}", va="center", fontsize=8)
    fig.tight_layout()
    path = figures_dir / "mlp_baseline_mae.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path


def run_mlp_diagnostics(config: MLPDiagnosticConfig) -> None:
    from sklearn.metrics import mean_absolute_error, r2_score

    from matbench_tabpfn.features import build_task_features, load_matbench_task
    from matbench_tabpfn.paths import collect_environment_manifest, create_run_paths, write_json

    paths = create_run_paths(
        PROJECT_ROOT,
        run_id=None,
        base_dir="results/mlp_baseline_diagnostics",
        update_latest=False,
    )
    write_json(paths.logs / "config.json", asdict(config))
    write_json(paths.logs / "environment.json", collect_environment_manifest(asdict(config)))

    metrics_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []

    for task_name in config.tasks:
        task = load_matbench_task(task_name)
        target_col = task.metadata["target"]
        unit = task.metadata.get("unit")
        input_type = task.metadata["input_type"]
        for feature_set in config.feature_sets:
            print(f"Building features: {task_name} | {feature_set}")
            try:
                features, feature_source, feature_set_display = build_task_features(
                    task_name,
                    task,
                    feature_set,
                    paths.features,
                    use_cache=config.use_feature_cache,
                    show_progress=config.show_feature_progress,
                    n_jobs=config.feature_n_jobs,
                )
            except ValueError as exc:
                print(f"Skipping {task_name} | {feature_set}: {exc}")
                continue

            for fold in task.folds_nums:
                print(f"{task_name} | {feature_set} | mlp | fold {fold}")
                X_train_raw, y_train = task.get_train_and_val_data(fold)
                X_test_raw, y_test = task.get_test_data(fold, include_target=True)
                X_train = features.loc[X_train_raw.index]
                X_test = features.loc[X_test_raw.index]

                model = _build_mlp(config)
                fit_start = time.time()
                model.fit(X_train, y_train)
                fit_seconds = time.time() - fit_start

                predict_start = time.time()
                y_pred = model.predict(X_test)
                predict_seconds = time.time() - predict_start

                metrics_rows.append(
                    {
                        "task": task_name,
                        "target": target_col,
                        "unit": unit,
                        "matbench_input_type": input_type,
                        "feature_source": feature_source,
                        "feature_set": feature_set,
                        "feature_set_display": feature_set_display,
                        "model": "mlp",
                        "model_display": "MLP",
                        "fold": fold,
                        "train_size": len(X_train),
                        "test_size": len(X_test),
                        "n_features": features.shape[1],
                        "mae": mean_absolute_error(y_test, y_pred),
                        "r2": r2_score(y_test, y_pred),
                        "fit_seconds": fit_seconds,
                        "predict_seconds": predict_seconds,
                        "random_seed": config.random_seed,
                    }
                )
                prediction_frames.append(
                    pd.DataFrame(
                        {
                            "task": task_name,
                            "target": target_col,
                            "unit": unit,
                            "feature_source": feature_source,
                            "feature_set": feature_set,
                            "feature_set_display": feature_set_display,
                            "model": "mlp",
                            "model_display": "MLP",
                            "fold": fold,
                            "mbid": y_test.index,
                            "y_true": y_test.to_numpy(),
                            "y_pred": y_pred,
                            "absolute_error": np.abs(y_test.to_numpy() - y_pred),
                        }
                    )
                )

    metrics_df = pd.DataFrame(metrics_rows)
    predictions_df = (
        pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    )
    summary_df = _summarize(metrics_df)

    metrics_df.to_csv(paths.metrics / "mlp_fold_metrics.csv", index=False)
    predictions_df.to_csv(paths.predictions / "mlp_predictions.csv", index=False)
    summary_df.to_csv(paths.tables / "mlp_summary.csv", index=False)
    _plot_mlp_summary(summary_df, paths.figures)
    summary_text = _write_summary_text(
        summary_df,
        paths.tables / "mlp_baseline_summary.md",
    )

    print("")
    print(summary_text)
    print(f"Run root: {paths.root}")


def main() -> None:
    run_mlp_diagnostics(_parse_args())


if __name__ == "__main__":
    main()
