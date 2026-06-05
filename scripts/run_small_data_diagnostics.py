"""Small-data learning curve and target-regime diagnostics.

This script is intentionally standalone: it imports the project's existing
feature/model helpers, but it does not modify the main notebook, source modules,
or synchronized result tables. Outputs go under:

    results/small_data_diagnostics/<run_id>/

Example:

    python scripts/run_small_data_diagnostics.py \
        --models extra_trees tabpfn \
        --fractions 0.1 0.2 0.4 0.6 0.8 1.0 \
        --repeats 1 \
        --device cuda
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


DEFAULT_TASKS = ("matbench_jdft2d", "matbench_phonons")
DEFAULT_FEATURE_SETS = ("magpie", "magpie_structure_all")
DEFAULT_MODELS = ("extra_trees", "tabpfn")
DEFAULT_FRACTIONS = (0.1, 0.2, 0.4, 0.6, 0.8, 1.0)
DEFAULT_RANDOM_SEED = 42


@dataclass(frozen=True)
class DiagnosticConfig:
    tasks: tuple[str, ...]
    feature_sets: tuple[str, ...]
    models: tuple[str, ...]
    fractions: tuple[float, ...]
    repeats: int
    folds: tuple[int, ...] | None
    random_seed: int
    n_estimators: int
    tabpfn_n_estimators: int
    device: str
    tabpfn_predict_batch_size: int | None
    feature_n_jobs: int
    use_feature_cache: bool
    show_feature_progress: bool
    n_stratification_bins: int
    n_target_regime_bins: int


def _parse_args() -> DiagnosticConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Run learning-curve and target-regime diagnostics without touching "
            "the main benchmark outputs."
        )
    )
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--feature-sets", nargs="+", default=list(DEFAULT_FEATURE_SETS))
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument(
        "--fractions",
        nargs="+",
        type=float,
        default=list(DEFAULT_FRACTIONS),
        help="Fractions of each official train fold to use.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Deterministic resampling repeats per fraction and fold.",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=None,
        help="Optional subset of Matbench fold numbers. Default: all official folds.",
    )
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--tabpfn-n-estimators", type=int, default=8)
    parser.add_argument("--device", default="cuda", help="TabPFN device, e.g. cuda or cpu.")
    parser.add_argument("--tabpfn-predict-batch-size", type=int, default=128)
    parser.add_argument("--feature-n-jobs", type=int, default=1)
    parser.add_argument("--no-feature-cache", action="store_true")
    parser.add_argument("--show-feature-progress", action="store_true")
    parser.add_argument("--n-stratification-bins", type=int, default=5)
    parser.add_argument("--n-target-regime-bins", type=int, default=3)
    args = parser.parse_args()

    fractions = tuple(sorted(set(float(value) for value in args.fractions)))
    if not fractions or any(value <= 0 or value > 1 for value in fractions):
        raise ValueError("All fractions must be in the interval (0, 1].")
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1.")
    if args.n_stratification_bins < 1:
        raise ValueError("--n-stratification-bins must be at least 1.")
    if args.n_target_regime_bins < 2:
        raise ValueError("--n-target-regime-bins must be at least 2.")

    return DiagnosticConfig(
        tasks=tuple(args.tasks),
        feature_sets=tuple(args.feature_sets),
        models=tuple(args.models),
        fractions=fractions,
        repeats=args.repeats,
        folds=tuple(args.folds) if args.folds is not None else None,
        random_seed=args.random_seed,
        n_estimators=args.n_estimators,
        tabpfn_n_estimators=args.tabpfn_n_estimators,
        device=args.device,
        tabpfn_predict_batch_size=args.tabpfn_predict_batch_size,
        feature_n_jobs=args.feature_n_jobs,
        use_feature_cache=not args.no_feature_cache,
        show_feature_progress=args.show_feature_progress,
        n_stratification_bins=args.n_stratification_bins,
        n_target_regime_bins=args.n_target_regime_bins,
    )


def _stratified_regression_subset(
    y_train: pd.Series,
    *,
    fraction: float,
    random_seed: int,
    n_bins: int,
) -> list:
    """Choose a deterministic target-stratified subset from one train fold."""

    if fraction >= 1:
        return list(y_train.index)

    n_total = len(y_train)
    n_keep = max(2, int(round(fraction * n_total)))
    if n_keep >= n_total:
        return list(y_train.index)

    rng = np.random.default_rng(random_seed)
    y_numeric = pd.to_numeric(y_train, errors="coerce")
    n_bins = min(n_bins, y_numeric.nunique(), n_keep)
    if n_bins <= 1:
        return list(rng.choice(y_train.index.to_numpy(), size=n_keep, replace=False))

    try:
        binned = pd.qcut(y_numeric, q=n_bins, labels=False, duplicates="drop").dropna()
    except ValueError:
        return list(rng.choice(y_train.index.to_numpy(), size=n_keep, replace=False))

    selected = []
    for _, indices in binned.groupby(binned).groups.items():
        candidates = np.array(list(indices), dtype=object)
        group_size = max(1, int(round(fraction * len(candidates))))
        group_size = min(group_size, len(candidates))
        selected.extend(rng.choice(candidates, size=group_size, replace=False).tolist())

    if len(selected) > n_keep:
        selected = rng.choice(np.array(selected, dtype=object), size=n_keep, replace=False).tolist()
    elif len(selected) < n_keep:
        remaining = np.array([idx for idx in y_train.index if idx not in set(selected)], dtype=object)
        if len(remaining):
            add_n = min(n_keep - len(selected), len(remaining))
            selected.extend(rng.choice(remaining, size=add_n, replace=False).tolist())

    return selected


def _target_regime_labels(y_test: pd.Series, n_bins: int) -> pd.Series:
    """Assign low/mid/high-style target regimes within an official test fold."""

    y_numeric = pd.to_numeric(y_test, errors="coerce")
    n_bins = min(n_bins, y_numeric.nunique())
    if n_bins <= 1:
        return pd.Series("all", index=y_test.index)

    try:
        codes = pd.qcut(y_numeric, q=n_bins, labels=False, duplicates="drop")
    except ValueError:
        return pd.Series("all", index=y_test.index)

    clean_codes = codes.dropna()
    if clean_codes.empty:
        return pd.Series("all", index=y_test.index)
    n_actual = int(clean_codes.max()) + 1
    if n_actual == 2:
        labels = {0: "low", 1: "high"}
    elif n_actual == 3:
        labels = {0: "low", 1: "middle", 2: "high"}
    else:
        labels = {idx: f"q{idx + 1}" for idx in range(n_actual)}
    return codes.map(labels).fillna("unknown").astype(str)


def _sem(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    if len(values) <= 1:
        return float("nan")
    return float(values.std(ddof=1) / np.sqrt(len(values)))


def _summarize(metrics_df: pd.DataFrame, regime_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = (
        metrics_df.query("status == 'ok'")
        .groupby(
            [
                "task",
                "feature_set",
                "feature_set_display",
                "model",
                "model_display",
                "train_fraction",
            ],
            dropna=False,
        )
        .agg(
            runs=("fold", "count"),
            mean_mae=("mae", "mean"),
            std_mae=("mae", "std"),
            sem_mae=("mae", _sem),
            mean_r2=("r2", "mean"),
            train_size_mean=("train_subset_size", "mean"),
            fit_seconds_sum=("fit_seconds", "sum"),
            predict_seconds_sum=("predict_seconds", "sum"),
        )
        .reset_index()
        .sort_values(["task", "feature_set", "model", "train_fraction"])
    )

    regime_summary = (
        regime_df.groupby(
            [
                "task",
                "feature_set",
                "feature_set_display",
                "model",
                "model_display",
                "train_fraction",
                "target_regime",
            ],
            dropna=False,
        )
        .agg(
            runs=("fold", "count"),
            test_points=("n_samples", "sum"),
            mean_regime_mae=("regime_mae", "mean"),
            std_regime_mae=("regime_mae", "std"),
            sem_regime_mae=("regime_mae", _sem),
            mean_target=("target_mean", "mean"),
        )
        .reset_index()
        .sort_values(["task", "feature_set", "model", "train_fraction", "target_regime"])
    )
    return summary, regime_summary


def _plot_learning_curve(summary_df: pd.DataFrame, figures_dir: Path) -> Path | None:
    if summary_df.empty:
        return None

    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="paper")
    tasks = list(summary_df["task"].drop_duplicates())
    fig, axes = plt.subplots(len(tasks), 1, figsize=(8.5, 3.8 * len(tasks)), squeeze=False)

    for ax, task_name in zip(axes.reshape(-1), tasks):
        task_df = summary_df.query("task == @task_name").copy()
        task_df["curve_label"] = (
            task_df["model_display"].astype(str)
            + " | "
            + task_df["feature_set_display"].astype(str)
        )
        for label, curve in task_df.groupby("curve_label", sort=False):
            curve = curve.sort_values("train_fraction")
            ax.errorbar(
                curve["train_fraction"],
                curve["mean_mae"],
                yerr=curve["sem_mae"].fillna(0),
                marker="o",
                linewidth=1.7,
                capsize=3,
                label=label,
            )
        ax.set_title(task_name)
        ax.set_xlabel("Fraction of official train fold used")
        ax.set_ylabel("Official test-fold MAE")
        ax.set_xlim(0, 1.03)
        ax.legend(fontsize=8, ncols=2)

    fig.tight_layout()
    path = figures_dir / "small_data_learning_curve.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_target_regimes(regime_summary_df: pd.DataFrame, figures_dir: Path) -> Path | None:
    if regime_summary_df.empty:
        return None

    import matplotlib.pyplot as plt
    import seaborn as sns

    full = regime_summary_df.query("train_fraction == 1.0").copy()
    if full.empty:
        full = (
            regime_summary_df.sort_values("train_fraction")
            .groupby(["task", "feature_set", "model", "target_regime"], as_index=False)
            .tail(1)
        )
    full["model_feature"] = (
        full["model_display"].astype(str) + "\n" + full["feature_set_display"].astype(str)
    )

    sns.set_theme(style="whitegrid", context="paper")
    tasks = list(full["task"].drop_duplicates())
    fig, axes = plt.subplots(len(tasks), 1, figsize=(9, 3.8 * len(tasks)), squeeze=False)

    for ax, task_name in zip(axes.reshape(-1), tasks):
        task_df = full.query("task == @task_name").copy()
        sns.barplot(
            data=task_df,
            x="model_feature",
            y="mean_regime_mae",
            hue="target_regime",
            ax=ax,
        )
        ax.set_title(f"{task_name}: target-regime MAE at full train size")
        ax.set_xlabel("")
        ax.set_ylabel("MAE within target regime")
        ax.tick_params(axis="x", labelrotation=25)
        ax.legend(title="Target regime", fontsize=8, title_fontsize=8)

    fig.tight_layout()
    path = figures_dir / "target_regime_error_diagnosis.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path


def _write_interpretation(summary_df: pd.DataFrame, regime_summary_df: pd.DataFrame, path: Path) -> str:
    lines = [
        "# Small-Data Diagnostic Summary",
        "",
        "This diagnostic asks whether the existing structure-aware TabPFN result is "
        "only a full-training-set benchmark result, or whether it remains useful "
        "when the official train folds are artificially reduced.",
        "",
        "## Best Full-Train Results",
        "",
    ]

    full = summary_df.query("train_fraction == 1.0").copy()
    if full.empty and not summary_df.empty:
        full = (
            summary_df.sort_values("train_fraction")
            .groupby(["task", "feature_set", "model"], as_index=False)
            .tail(1)
        )

    for task_name, task_df in full.groupby("task", sort=False):
        best = task_df.sort_values("mean_mae").iloc[0]
        lines.append(
            "- `{}`: best full-train diagnostic model is {} with {}, MAE={:.4g}.".format(
                task_name,
                best["model_display"],
                best["feature_set_display"],
                best["mean_mae"],
            )
        )

    lines.extend(["", "## Representation Gain At Full Train Size", ""])
    for (task_name, model_name), task_model in full.groupby(["task", "model"], sort=False):
        proxy = task_model.query("feature_set == 'magpie'")
        structure = task_model.query("feature_set == 'magpie_structure_all'")
        if proxy.empty or structure.empty:
            continue
        proxy_mae = float(proxy.iloc[0]["mean_mae"])
        structure_mae = float(structure.iloc[0]["mean_mae"])
        pct = 100.0 * (structure_mae - proxy_mae) / proxy_mae
        direction = "improves over" if pct < 0 else "does not improve over"
        lines.append(
            "- `{}` / `{}`: all-structure {} composition proxy by {:+.2f}% MAE.".format(
                task_name, model_name, direction, pct
            )
        )

    lines.extend(["", "## Target-Regime Notes", ""])
    if not regime_summary_df.empty:
        full_regime = regime_summary_df.query("train_fraction == 1.0").copy()
        for task_name, task_df in full_regime.groupby("task", sort=False):
            hardest = task_df.sort_values("mean_regime_mae", ascending=False).iloc[0]
            lines.append(
                "- `{}`: hardest full-train target regime in this diagnostic is `{}` "
                "for {} with {}, MAE={:.4g}.".format(
                    task_name,
                    hardest["target_regime"],
                    hardest["model_display"],
                    hardest["feature_set_display"],
                    hardest["mean_regime_mae"],
                )
            )

    text = "\n".join(lines).strip() + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


def _iter_requested_folds(task, folds: Iterable[int] | None) -> list[int]:
    official = list(task.folds_nums)
    if folds is None:
        return official
    requested = list(folds)
    unknown = sorted(set(requested).difference(official))
    if unknown:
        raise ValueError(f"Requested folds are not official Matbench folds: {unknown}")
    return requested


def run_diagnostics(config: DiagnosticConfig) -> None:
    from sklearn.metrics import mean_absolute_error, r2_score

    from matbench_tabpfn.features import build_task_features, load_matbench_task
    from matbench_tabpfn.models import build_model, predict_in_batches
    from matbench_tabpfn.paths import (
        collect_environment_manifest,
        create_run_paths,
        write_json,
    )
    from matbench_tabpfn.settings import MODEL_DISPLAY_NAMES

    paths = create_run_paths(
        PROJECT_ROOT,
        run_id=None,
        base_dir="results/small_data_diagnostics",
        update_latest=False,
    )
    write_json(paths.logs / "config.json", asdict(config))
    write_json(paths.logs / "environment.json", collect_environment_manifest(asdict(config)))

    metrics_rows: list[dict] = []
    prediction_frames: list[pd.DataFrame] = []
    regime_rows: list[dict] = []

    for task_name in config.tasks:
        task = load_matbench_task(task_name)
        target_col = task.metadata["target"]
        unit = task.metadata.get("unit")
        input_type = task.metadata["input_type"]
        folds = _iter_requested_folds(task, config.folds)

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

            for fold in folds:
                X_train_raw, y_train = task.get_train_and_val_data(fold)
                X_test_raw, y_test = task.get_test_data(fold, include_target=True)
                X_test = features.loc[X_test_raw.index]
                target_regime = _target_regime_labels(y_test, config.n_target_regime_bins)

                for fraction in config.fractions:
                    repeat_count = 1 if fraction >= 1 else config.repeats
                    for repeat in range(repeat_count):
                        subset_seed = (
                            config.random_seed
                            + 10_000 * int(fold)
                            + 1_000 * repeat
                            + int(round(fraction * 1_000))
                        )
                        subset_idx = _stratified_regression_subset(
                            y_train,
                            fraction=fraction,
                            random_seed=subset_seed,
                            n_bins=config.n_stratification_bins,
                        )
                        X_train = features.loc[subset_idx]
                        y_subset = y_train.loc[subset_idx]

                        for model_name in config.models:
                            print(
                                f"{task_name} | {feature_set} | {model_name} | "
                                f"fold {fold} | fraction {fraction:.2f} | repeat {repeat}"
                            )
                            row_base = {
                                "task": task_name,
                                "target": target_col,
                                "unit": unit,
                                "matbench_input_type": input_type,
                                "feature_source": feature_source,
                                "feature_set": feature_set,
                                "feature_set_display": feature_set_display,
                                "model": model_name,
                                "model_display": MODEL_DISPLAY_NAMES.get(model_name, model_name),
                                "fold": fold,
                                "train_fraction": fraction,
                                "repeat": repeat,
                                "train_full_size": len(y_train),
                                "train_subset_size": len(y_subset),
                                "test_size": len(y_test),
                                "n_features": features.shape[1],
                                "random_seed": config.random_seed,
                                "subset_seed": subset_seed,
                            }
                            try:
                                model = build_model(
                                    model_name,
                                    random_seed=config.random_seed,
                                    n_estimators=config.n_estimators,
                                    tabpfn_n_estimators=config.tabpfn_n_estimators,
                                    tabpfn_device=config.device,
                                )
                                fit_start = time.time()
                                model.fit(X_train, y_subset)
                                fit_seconds = time.time() - fit_start

                                predict_start = time.time()
                                batch_size = (
                                    config.tabpfn_predict_batch_size
                                    if model_name == "tabpfn"
                                    else None
                                )
                                y_pred = predict_in_batches(model, X_test, batch_size)
                                predict_seconds = time.time() - predict_start

                                abs_error = np.abs(y_test.to_numpy() - y_pred)
                                mae = mean_absolute_error(y_test, y_pred)
                                r2 = r2_score(y_test, y_pred)
                                metrics_rows.append(
                                    {
                                        **row_base,
                                        "status": "ok",
                                        "mae": mae,
                                        "r2": r2,
                                        "fit_seconds": fit_seconds,
                                        "predict_seconds": predict_seconds,
                                    }
                                )
                                fold_predictions = pd.DataFrame(
                                    {
                                        **{
                                            key: value
                                            for key, value in row_base.items()
                                            if key
                                            in {
                                                "task",
                                                "target",
                                                "unit",
                                                "feature_source",
                                                "feature_set",
                                                "feature_set_display",
                                                "model",
                                                "model_display",
                                                "fold",
                                                "train_fraction",
                                                "repeat",
                                                "train_subset_size",
                                            }
                                        },
                                        "mbid": y_test.index,
                                        "y_true": y_test.to_numpy(),
                                        "y_pred": y_pred,
                                        "absolute_error": abs_error,
                                        "target_regime": target_regime.loc[y_test.index].to_numpy(),
                                    }
                                )
                                prediction_frames.append(fold_predictions)

                                for regime, regime_df in fold_predictions.groupby("target_regime"):
                                    regime_rows.append(
                                        {
                                            **row_base,
                                            "target_regime": regime,
                                            "n_samples": len(regime_df),
                                            "target_mean": regime_df["y_true"].mean(),
                                            "regime_mae": mean_absolute_error(
                                                regime_df["y_true"], regime_df["y_pred"]
                                            ),
                                            "regime_r2": (
                                                r2_score(regime_df["y_true"], regime_df["y_pred"])
                                                if len(regime_df) > 1
                                                else np.nan
                                            ),
                                        }
                                    )
                            except Exception as exc:
                                metrics_rows.append(
                                    {
                                        **row_base,
                                        "status": "error",
                                        "mae": np.nan,
                                        "r2": np.nan,
                                        "fit_seconds": np.nan,
                                        "predict_seconds": np.nan,
                                        "error_type": type(exc).__name__,
                                        "error_message": str(exc),
                                    }
                                )
                                raise

    metrics_df = pd.DataFrame(metrics_rows)
    predictions_df = (
        pd.concat(prediction_frames, ignore_index=True) if prediction_frames else pd.DataFrame()
    )
    regime_df = pd.DataFrame(regime_rows)
    summary_df, regime_summary_df = _summarize(metrics_df, regime_df)

    metrics_df.to_csv(paths.metrics / "small_data_fold_metrics.csv", index=False)
    predictions_df.to_csv(paths.predictions / "small_data_predictions.csv", index=False)
    regime_df.to_csv(paths.metrics / "target_regime_fold_metrics.csv", index=False)
    summary_df.to_csv(paths.tables / "small_data_learning_curve_summary.csv", index=False)
    regime_summary_df.to_csv(paths.tables / "target_regime_summary.csv", index=False)

    learning_curve_path = _plot_learning_curve(summary_df, paths.figures)
    regime_path = _plot_target_regimes(regime_summary_df, paths.figures)
    summary_text = _write_interpretation(
        summary_df,
        regime_summary_df,
        paths.tables / "small_data_diagnostic_summary.md",
    )

    print("")
    print(summary_text)
    print(f"Run root: {paths.root}")
    if learning_curve_path:
        print(f"Learning curve figure: {learning_curve_path}")
    if regime_path:
        print(f"Target-regime figure: {regime_path}")


def main() -> None:
    config = _parse_args()
    run_diagnostics(config)


if __name__ == "__main__":
    main()
