"""Leakage-free tuned dense-MLP baseline.

This is a separate extension from `run_mlp_baseline_diagnostics.py`. For each
official Matbench fold, it tunes MLP hyperparameters only inside the official
training split, then retrains on the full official training split and evaluates
once on the official test split.

Outputs go under:

    results/tuned_mlp_baseline_diagnostics/<run_id>/
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
class TunedMLPConfig:
    tasks: tuple[str, ...]
    feature_sets: tuple[str, ...]
    random_seed: int
    tune_hidden_layers: tuple[tuple[int, ...], ...]
    tune_alphas: tuple[float, ...]
    tune_learning_rates: tuple[float, ...]
    max_iter: int
    early_stopping: bool
    validation_fraction: float
    n_iter_no_change: int
    feature_n_jobs: int
    use_feature_cache: bool
    show_feature_progress: bool


def _parse_layer_spec(spec: str) -> tuple[int, ...]:
    layers = tuple(int(item) for item in spec.split(",") if item.strip())
    if not layers:
        raise ValueError(f"Invalid layer spec: {spec!r}")
    return layers


def _parse_args() -> TunedMLPConfig:
    parser = argparse.ArgumentParser(
        description="Run an inner-validation tuned dense MLP baseline."
    )
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--feature-sets", nargs="+", default=list(DEFAULT_FEATURE_SETS))
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument(
        "--tune-hidden-layers",
        nargs="+",
        default=["64", "128", "128,64", "256,128"],
        help="Candidate hidden-layer specs, e.g. 64 128,64 256,128.",
    )
    parser.add_argument("--tune-alphas", nargs="+", type=float, default=[1e-4, 1e-3, 1e-2])
    parser.add_argument(
        "--tune-learning-rates",
        nargs="+",
        type=float,
        default=[1e-3, 3e-4],
    )
    parser.add_argument("--max-iter", type=int, default=600)
    parser.add_argument("--no-early-stopping", action="store_true")
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--n-iter-no-change", type=int, default=30)
    parser.add_argument("--feature-n-jobs", type=int, default=1)
    parser.add_argument("--no-feature-cache", action="store_true")
    parser.add_argument("--show-feature-progress", action="store_true")
    args = parser.parse_args()

    return TunedMLPConfig(
        tasks=tuple(args.tasks),
        feature_sets=tuple(args.feature_sets),
        random_seed=args.random_seed,
        tune_hidden_layers=tuple(_parse_layer_spec(spec) for spec in args.tune_hidden_layers),
        tune_alphas=tuple(args.tune_alphas),
        tune_learning_rates=tuple(args.tune_learning_rates),
        max_iter=args.max_iter,
        early_stopping=not args.no_early_stopping,
        validation_fraction=args.validation_fraction,
        n_iter_no_change=args.n_iter_no_change,
        feature_n_jobs=args.feature_n_jobs,
        use_feature_cache=not args.no_feature_cache,
        show_feature_progress=args.show_feature_progress,
    )


def _layer_label(layers: tuple[int, ...]) -> str:
    return ",".join(str(layer) for layer in layers)


def _build_mlp(
    config: TunedMLPConfig,
    *,
    hidden_layers: tuple[int, ...],
    alpha: float,
    learning_rate_init: float,
):
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
                    hidden_layer_sizes=hidden_layers,
                    activation="relu",
                    solver="adam",
                    alpha=alpha,
                    learning_rate_init=learning_rate_init,
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


def _tune_one_fold(
    *,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: TunedMLPConfig,
    split_seed: int,
    metadata: dict,
) -> tuple[dict, list[dict]]:
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split

    inner_train_idx, inner_val_idx = train_test_split(
        list(X_train.index),
        test_size=config.validation_fraction,
        random_state=split_seed,
        shuffle=True,
    )
    X_inner_train = X_train.loc[inner_train_idx]
    y_inner_train = y_train.loc[inner_train_idx]
    X_inner_val = X_train.loc[inner_val_idx]
    y_inner_val = y_train.loc[inner_val_idx]

    rows = []
    for layers in config.tune_hidden_layers:
        for alpha in config.tune_alphas:
            for lr in config.tune_learning_rates:
                model = _build_mlp(
                    config,
                    hidden_layers=layers,
                    alpha=alpha,
                    learning_rate_init=lr,
                )
                fit_start = time.time()
                model.fit(X_inner_train, y_inner_train)
                fit_seconds = time.time() - fit_start
                y_val_pred = model.predict(X_inner_val)
                rows.append(
                    {
                        **metadata,
                        "inner_train_size": len(X_inner_train),
                        "inner_val_size": len(X_inner_val),
                        "hidden_layers": _layer_label(layers),
                        "alpha": alpha,
                        "learning_rate_init": lr,
                        "n_hidden_units": sum(layers),
                        "inner_val_mae": mean_absolute_error(y_inner_val, y_val_pred),
                        "inner_val_r2": r2_score(y_inner_val, y_val_pred),
                        "inner_fit_seconds": fit_seconds,
                    }
                )

    validation_df = pd.DataFrame(rows)
    best = validation_df.sort_values(
        ["inner_val_mae", "n_hidden_units", "alpha", "learning_rate_init"]
    ).iloc[0]
    selected = {
        "hidden_layers": tuple(int(item) for item in str(best["hidden_layers"]).split(",")),
        "alpha": float(best["alpha"]),
        "learning_rate_init": float(best["learning_rate_init"]),
        "inner_val_mae": float(best["inner_val_mae"]),
        "inner_val_r2": float(best["inner_val_r2"]),
    }
    return selected, rows


def _write_summary_text(summary_df: pd.DataFrame, path: Path) -> str:
    lines = [
        "# Tuned MLP Baseline Summary",
        "",
        "The MLP hyperparameters were selected by inner validation within each official "
        "Matbench training fold. The official test folds were used only for final "
        "evaluation.",
        "",
    ]
    for task, task_df in summary_df.groupby("task", sort=False):
        best = task_df.sort_values("mean_mae").iloc[0]
        lines.append(
            "- `{}`: best tuned MLP branch is {}, MAE={:.4g}, R2={:.4g}.".format(
                task,
                best["feature_set_display"],
                best["mean_mae"],
                best["mean_r2"],
            )
        )
    lines.append("")
    lines.append(
        "This model-complexity check compares a tuned dense neural network with "
        "TabPFN and tree baselines on the same official folds."
    )
    text = "\n".join(lines).strip() + "\n"
    path.write_text(text, encoding="utf-8")
    return text


def _plot_summary(summary_df: pd.DataFrame, figures_dir: Path) -> Path | None:
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
            color="#7d6ba8",
            alpha=0.9,
        )
        ax.invert_yaxis()
        ax.set_title(f"{task}: inner-validation tuned MLP")
        ax.set_xlabel("Official-fold MAE")
        for idx, value in enumerate(task_df["mean_mae"]):
            ax.text(value, idx, f" {value:.3g}", va="center", fontsize=8)
    fig.tight_layout()
    path = figures_dir / "tuned_mlp_baseline_mae.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path


def run_tuned_mlp(config: TunedMLPConfig) -> None:
    from sklearn.metrics import mean_absolute_error, r2_score

    from matbench_tabpfn.features import build_task_features, load_matbench_task
    from matbench_tabpfn.paths import collect_environment_manifest, create_run_paths, write_json

    paths = create_run_paths(
        PROJECT_ROOT,
        run_id=None,
        base_dir="results/tuned_mlp_baseline_diagnostics",
        update_latest=False,
    )
    write_json(paths.logs / "config.json", asdict(config))
    write_json(paths.logs / "environment.json", collect_environment_manifest(asdict(config)))

    metrics_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    validation_rows: list[dict] = []

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
                print(f"{task_name} | {feature_set} | tuned_mlp | fold {fold}")
                X_train_raw, y_train = task.get_train_and_val_data(fold)
                X_test_raw, y_test = task.get_test_data(fold, include_target=True)
                X_train = features.loc[X_train_raw.index]
                X_test = features.loc[X_test_raw.index]

                selected, fold_validation_rows = _tune_one_fold(
                    X_train=X_train,
                    y_train=y_train,
                    config=config,
                    split_seed=config.random_seed + int(fold),
                    metadata={
                        "task": task_name,
                        "feature_set": feature_set,
                        "feature_set_display": feature_set_display,
                        "fold": fold,
                        "random_seed": config.random_seed,
                    },
                )
                validation_rows.extend(fold_validation_rows)
                print(
                    "  selected hidden_layers={} alpha={} lr={} inner_val_mae={:.4g}".format(
                        _layer_label(selected["hidden_layers"]),
                        selected["alpha"],
                        selected["learning_rate_init"],
                        selected["inner_val_mae"],
                    )
                )

                model = _build_mlp(
                    config,
                    hidden_layers=selected["hidden_layers"],
                    alpha=selected["alpha"],
                    learning_rate_init=selected["learning_rate_init"],
                )
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
                        "model": "tuned_mlp",
                        "model_display": "Tuned MLP",
                        "fold": fold,
                        "train_size": len(X_train),
                        "test_size": len(X_test),
                        "n_features": features.shape[1],
                        "mae": mean_absolute_error(y_test, y_pred),
                        "r2": r2_score(y_test, y_pred),
                        "fit_seconds": fit_seconds,
                        "predict_seconds": predict_seconds,
                        "selected_hidden_layers": _layer_label(selected["hidden_layers"]),
                        "selected_alpha": selected["alpha"],
                        "selected_learning_rate_init": selected["learning_rate_init"],
                        "selected_inner_val_mae": selected["inner_val_mae"],
                        "selected_inner_val_r2": selected["inner_val_r2"],
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
                            "model": "tuned_mlp",
                            "model_display": "Tuned MLP",
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
    validation_df = pd.DataFrame(validation_rows)
    summary_df = _summarize(metrics_df)

    metrics_df.to_csv(paths.metrics / "tuned_mlp_fold_metrics.csv", index=False)
    predictions_df.to_csv(paths.predictions / "tuned_mlp_predictions.csv", index=False)
    validation_df.to_csv(paths.tables / "tuned_mlp_inner_validation_results.csv", index=False)
    summary_df.to_csv(paths.tables / "tuned_mlp_summary.csv", index=False)
    _plot_summary(summary_df, paths.figures)
    summary_text = _write_summary_text(
        summary_df,
        paths.tables / "tuned_mlp_baseline_summary.md",
    )

    print("")
    print(summary_text)
    print(f"Run root: {paths.root}")


def main() -> None:
    run_tuned_mlp(_parse_args())


if __name__ == "__main__":
    main()
