"""Official-fold evaluation utilities."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from .features import (
    build_task_features,
    feature_set_requires_structure,
    load_matbench_task,
)
from .models import build_model, predict_in_batches
from .paths import RunPaths, write_json
from .settings import MODEL_DISPLAY_NAMES, RANDOM_SEED, TABPFN_FAMILY_MODELS


FIXED_WEIGHT_ENSEMBLE_MODEL = "tabpfn_extra_trees_ensemble"
FIXED_WEIGHT_ENSEMBLE_DISPLAY = MODEL_DISPLAY_NAMES[FIXED_WEIGHT_ENSEMBLE_MODEL]
FIXED_WEIGHT_ENSEMBLE_COMPONENTS = ("tabpfn", "extra_trees")
FIXED_WEIGHT_ENSEMBLE_TABPFN_WEIGHT = 0.5
INNER_TUNED_ENSEMBLE_MODEL = "tabpfn_extra_trees_inner_tuned"
INNER_TUNED_ENSEMBLE_DISPLAY = MODEL_DISPLAY_NAMES[INNER_TUNED_ENSEMBLE_MODEL]


def summarize_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    ok = metrics_df.query("status == 'ok'").copy()
    if ok.empty:
        return pd.DataFrame()

    summary = (
        ok.groupby(
            [
                "task",
                "target",
                "unit",
                "matbench_input_type",
                "feature_source",
                "feature_set",
                "feature_set_display",
                "model",
                "model_display",
            ],
            dropna=False,
        )
        .agg(
            folds_completed=("fold", "nunique"),
            mean_mae=("mae", "mean"),
            std_mae=("mae", "std"),
            sem_mae=("mae", lambda x: x.std(ddof=1) / np.sqrt(len(x))),
            mean_r2=("r2", "mean"),
            std_r2=("r2", "std"),
            n_features=("n_features", "mean"),
            train_size_mean=("train_size", "mean"),
            test_size_sum=("test_size", "sum"),
            fit_seconds_sum=("fit_seconds", "sum"),
            predict_seconds_sum=("predict_seconds", "sum"),
        )
        .reset_index()
    )
    summary["rank_by_mae"] = summary.groupby("task")["mean_mae"].rank(
        method="dense", ascending=True
    ).astype(int)
    return summary.sort_values(["task", "rank_by_mae", "model"]).reset_index(drop=True)


def compare_against_best_baseline(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for task_name, task_summary in summary_df.groupby("task", sort=False):
        baseline_pool = task_summary[
            ~task_summary["model"].isin([*TABPFN_FAMILY_MODELS, "dummy_mean"])
        ].copy()
        if baseline_pool.empty:
            continue
        best = baseline_pool.sort_values("mean_mae").iloc[0]
        for _, row in task_summary.iterrows():
            delta = row["mean_mae"] - best["mean_mae"]
            pct = 100.0 * delta / best["mean_mae"]
            rows.append(
                {
                    "task": task_name,
                    "feature_source": row["feature_source"],
                    "feature_set": row["feature_set"],
                    "feature_set_display": row["feature_set_display"],
                    "model": row["model"],
                    "model_display": row["model_display"],
                    "mean_mae": row["mean_mae"],
                    "best_baseline_model": best["model"],
                    "best_baseline_display": best["model_display"],
                    "best_baseline_feature_set": best["feature_set"],
                    "best_baseline_feature_set_display": best["feature_set_display"],
                    "best_baseline_mean_mae": best["mean_mae"],
                    "mae_delta_vs_best_baseline": delta,
                    "mae_pct_change_vs_best_baseline": pct,
                }
            )
    return pd.DataFrame(rows)


def paired_fold_comparisons(metrics_df: pd.DataFrame) -> pd.DataFrame:
    ok = metrics_df.query("status == 'ok'").copy()
    if ok.empty or "tabpfn" not in set(ok["model"]):
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    tabpfn_rows = ok.query("model == 'tabpfn'").copy()
    for (task_name, reference_feature_set), tabpfn_metrics in tabpfn_rows.groupby(
        ["task", "feature_set"], sort=False
    ):
        task_metrics = ok.query("task == @task_name").copy()
        tabpfn = tabpfn_metrics[["fold", "mae"]].rename(columns={"mae": "tabpfn_mae"})
        baseline_pool = task_metrics[
            ~task_metrics["model"].isin([*TABPFN_FAMILY_MODELS, "dummy_mean"])
        ].copy()
        if tabpfn.empty or baseline_pool.empty:
            continue

        baseline_summary = (
            baseline_pool.groupby(["model", "feature_set"])["mae"]
            .mean()
            .sort_values()
            .reset_index()
        )
        best_baseline_model = baseline_summary.iloc[0]["model"]
        best_baseline_feature_set = baseline_summary.iloc[0]["feature_set"]
        baseline = baseline_pool.query(
            "model == @best_baseline_model and feature_set == @best_baseline_feature_set"
        )[["fold", "mae"]].rename(columns={"mae": "baseline_mae"})
        merged = tabpfn.merge(baseline, on="fold", how="inner")
        if merged.empty:
            continue

        merged["delta_tabpfn_minus_baseline"] = (
            merged["tabpfn_mae"] - merged["baseline_mae"]
        )
        rows.append(
            {
                "task": task_name,
                "reference_model": "tabpfn",
                "reference_feature_set": reference_feature_set,
                "best_baseline_model": best_baseline_model,
                "best_baseline_feature_set": best_baseline_feature_set,
                "n_paired_folds": len(merged),
                "mean_delta_tabpfn_minus_baseline": merged[
                    "delta_tabpfn_minus_baseline"
                ].mean(),
                "median_delta_tabpfn_minus_baseline": merged[
                    "delta_tabpfn_minus_baseline"
                ].median(),
                "tabpfn_win_folds": int((merged["tabpfn_mae"] < merged["baseline_mae"]).sum()),
            }
        )
    return pd.DataFrame(rows)


def _save_combined_outputs(
    paths: RunPaths,
    metrics: list[dict[str, Any]],
    predictions: list[pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics_df = pd.DataFrame(metrics)
    predictions_df = (
        pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    )
    return _write_evaluation_outputs(paths, metrics_df, predictions_df)


def _write_evaluation_outputs(
    paths: RunPaths,
    metrics_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_df = summarize_metrics(metrics_df)
    baseline_comparison_df = compare_against_best_baseline(summary_df)
    paired_comparison_df = paired_fold_comparisons(metrics_df)

    metrics_df.to_csv(paths.metrics / "fold_metrics.csv", index=False)
    predictions_df.to_csv(paths.predictions / "all_predictions.csv", index=False)
    summary_df.to_csv(paths.metrics / "model_summary.csv", index=False)
    baseline_comparison_df.to_csv(
        paths.metrics / "best_baseline_comparison.csv", index=False
    )
    paired_comparison_df.to_csv(paths.metrics / "paired_fold_comparisons.csv", index=False)
    return (
        metrics_df,
        predictions_df,
        summary_df,
        baseline_comparison_df,
        paired_comparison_df,
    )


def build_fixed_weight_tabpfn_extra_trees_ensemble(
    metrics_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    *,
    tabpfn_weight: float = FIXED_WEIGHT_ENSEMBLE_TABPFN_WEIGHT,
    model_name: str = FIXED_WEIGHT_ENSEMBLE_MODEL,
    model_display: str = FIXED_WEIGHT_ENSEMBLE_DISPLAY,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a post-hoc TabPFN/ExtraTrees ensemble from saved predictions."""

    if metrics_df.empty or predictions_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    required = {
        "task",
        "target",
        "unit",
        "feature_source",
        "feature_set",
        "feature_set_display",
        "model",
        "fold",
        "mbid",
        "y_true",
        "y_pred",
    }
    missing = required.difference(predictions_df.columns)
    if missing:
        raise ValueError(f"Predictions are missing required columns: {sorted(missing)}")
    if not 0 <= tabpfn_weight <= 1:
        raise ValueError(f"Expected tabpfn_weight in [0, 1], got {tabpfn_weight}.")

    tabpfn_model, extra_trees_model = FIXED_WEIGHT_ENSEMBLE_COMPONENTS
    extra_trees_weight = 1.0 - tabpfn_weight
    clean_predictions = predictions_df.query("model != @model_name").copy()
    clean_metrics = metrics_df.query("model != @model_name").copy()
    tabpfn_predictions = clean_predictions.query("model == @tabpfn_model").copy()
    extra_trees_predictions = clean_predictions.query(
        "model == @extra_trees_model"
    ).copy()
    if tabpfn_predictions.empty or extra_trees_predictions.empty:
        return pd.DataFrame(), pd.DataFrame()

    join_keys = ["task", "feature_set", "fold", "mbid"]
    keep_cols = [
        "task",
        "target",
        "unit",
        "feature_source",
        "feature_set",
        "feature_set_display",
        "fold",
        "mbid",
        "y_true",
        "y_pred",
    ]
    merged = tabpfn_predictions[keep_cols].merge(
        extra_trees_predictions[join_keys + ["y_true", "y_pred"]],
        on=join_keys,
        how="inner",
        suffixes=("_tabpfn", "_extra_trees"),
        validate="one_to_one",
    )
    if merged.empty:
        return pd.DataFrame(), pd.DataFrame()

    y_true_delta = (
        merged["y_true_tabpfn"].to_numpy() - merged["y_true_extra_trees"].to_numpy()
    )
    if not np.allclose(y_true_delta, 0.0, rtol=0.0, atol=1e-10):
        raise ValueError("TabPFN and ExtraTrees predictions have mismatched y_true values.")

    merged["y_pred"] = (
        tabpfn_weight * merged["y_pred_tabpfn"]
        + extra_trees_weight * merged["y_pred_extra_trees"]
    )
    merged["y_true"] = merged["y_true_tabpfn"]
    merged["absolute_error"] = np.abs(merged["y_true"] - merged["y_pred"])
    merged["model"] = model_name
    merged["model_display"] = model_display
    merged["ensemble_tabpfn_weight"] = tabpfn_weight
    merged["ensemble_extra_trees_weight"] = extra_trees_weight
    ensemble_predictions = merged[
        [
            "task",
            "target",
            "unit",
            "feature_source",
            "feature_set",
            "feature_set_display",
            "model",
            "model_display",
            "fold",
            "mbid",
            "y_true",
            "y_pred",
            "absolute_error",
            "ensemble_tabpfn_weight",
            "ensemble_extra_trees_weight",
        ]
    ].copy()

    metric_rows: list[dict[str, Any]] = []
    for (task_name, feature_set, fold), fold_predictions in ensemble_predictions.groupby(
        ["task", "feature_set", "fold"], sort=False
    ):
        template_pool = clean_metrics.query(
            "task == @task_name and feature_set == @feature_set and "
            "fold == @fold and model == @tabpfn_model and status == 'ok'"
        )
        if template_pool.empty:
            continue
        template = template_pool.iloc[0].to_dict()
        y_true = fold_predictions["y_true"].to_numpy()
        y_pred = fold_predictions["y_pred"].to_numpy()
        row = {
            **template,
            "model": model_name,
            "model_display": model_display,
            "mae": mean_absolute_error(y_true, y_pred),
            "r2": r2_score(y_true, y_pred),
            "fit_seconds": 0.0,
            "predict_seconds": 0.0,
            "ensemble_tabpfn_weight": tabpfn_weight,
            "ensemble_extra_trees_weight": extra_trees_weight,
        }
        metric_rows.append(row)

    return pd.DataFrame(metric_rows), ensemble_predictions


def add_fixed_weight_ensemble_outputs(
    *,
    metrics_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    paths: RunPaths,
    tabpfn_weight: float = FIXED_WEIGHT_ENSEMBLE_TABPFN_WEIGHT,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Append the fixed TabPFN/ExtraTrees ensemble and rewrite derived outputs."""

    model_name = FIXED_WEIGHT_ENSEMBLE_MODEL
    base_metrics = metrics_df.query("model != @model_name").copy()
    base_predictions = predictions_df.query("model != @model_name").copy()
    ensemble_metrics, ensemble_predictions = (
        build_fixed_weight_tabpfn_extra_trees_ensemble(
            base_metrics,
            base_predictions,
            tabpfn_weight=tabpfn_weight,
        )
    )
    if not ensemble_metrics.empty:
        base_metrics = pd.concat([base_metrics, ensemble_metrics], ignore_index=True)
    if not ensemble_predictions.empty:
        base_predictions = pd.concat(
            [base_predictions, ensemble_predictions], ignore_index=True
        )
        ensemble_predictions.to_csv(
            paths.predictions / f"{model_name}_predictions.csv", index=False
        )

    return _write_evaluation_outputs(paths, base_metrics, base_predictions)


def _select_ensemble_weight_by_mae(
    y_true: pd.Series | np.ndarray,
    tabpfn_pred: np.ndarray,
    extra_trees_pred: np.ndarray,
    weights: Sequence[float],
    *,
    tie_tolerance: float = 1e-12,
) -> tuple[float, pd.DataFrame]:
    rows = []
    y_true_array = np.asarray(y_true, dtype=float)
    for weight in weights:
        weight = float(weight)
        y_pred = weight * tabpfn_pred + (1.0 - weight) * extra_trees_pred
        rows.append(
            {
                "tabpfn_weight": weight,
                "extra_trees_weight": 1.0 - weight,
                "inner_val_mae": mean_absolute_error(y_true_array, y_pred),
                "inner_val_r2": r2_score(y_true_array, y_pred),
            }
        )
    scan = pd.DataFrame(rows).sort_values(["inner_val_mae", "tabpfn_weight"])
    best_mae = scan["inner_val_mae"].iloc[0]
    tied = scan.query("inner_val_mae <= @best_mae + @tie_tolerance")
    # Prefer the simpler mostly-TabPFN solution when the validation MAE tie is exact.
    best = tied.sort_values("tabpfn_weight", ascending=False).iloc[0]
    return float(best["tabpfn_weight"]), scan.reset_index(drop=True)


def add_inner_validation_tuned_ensemble_outputs(
    *,
    metrics_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    paths: RunPaths,
    targets: Sequence[tuple[str, str]],
    weights: Sequence[float] | None = None,
    inner_val_fraction: float = 0.2,
    random_seed: int = RANDOM_SEED,
    n_estimators: int = 500,
    tabpfn_n_estimators: int = 8,
    tabpfn_device: str = "cuda",
    tabpfn_predict_batch_size: int | None = 128,
    use_feature_cache: bool = True,
    show_feature_progress: bool = True,
    feature_n_jobs: int = 1,
    tabpfn_kwargs: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Tune TabPFN/ExtraTrees ensemble weights inside each official train fold."""

    if predictions_df.empty:
        raise ValueError("predictions_df is required for official test-fold components.")
    if not targets:
        return _write_evaluation_outputs(paths, metrics_df, predictions_df)
    if not 0 < inner_val_fraction < 1:
        raise ValueError(
            f"Expected inner_val_fraction in (0, 1), got {inner_val_fraction}."
        )

    weights = list(weights if weights is not None else np.linspace(0.0, 1.0, 21))
    model_name = INNER_TUNED_ENSEMBLE_MODEL
    base_metrics = metrics_df.query("model != @model_name").copy()
    base_predictions = predictions_df.query("model != @model_name").copy()
    ensemble_metrics: list[dict[str, Any]] = []
    ensemble_predictions: list[pd.DataFrame] = []
    validation_rows: list[dict[str, Any]] = []

    for task_name, feature_set in targets:
        task = load_matbench_task(task_name)
        target_col = task.metadata["target"]
        unit = task.metadata.get("unit")
        input_type = task.metadata["input_type"]
        if feature_set_requires_structure(feature_set) and input_type != "structure":
            print(f"{task_name} | {feature_set} | skipped; task is not structure input")
            continue

        features, feature_source, feature_set_display = build_task_features(
            task_name,
            task,
            feature_set,
            paths.features,
            use_cache=use_feature_cache,
            show_progress=show_feature_progress,
            n_jobs=feature_n_jobs,
        )

        for fold in task.folds_nums:
            print(f"{task_name} | {feature_set} | inner-tuned ensemble | fold {fold}")
            X_train_raw, y_train = task.get_train_and_val_data(fold)
            X_test_raw, y_test = task.get_test_data(fold, include_target=True)
            train_indices = list(X_train_raw.index)
            inner_train_idx, inner_val_idx = train_test_split(
                train_indices,
                test_size=inner_val_fraction,
                random_state=random_seed + int(fold),
                shuffle=True,
            )
            X_inner_train = features.loc[inner_train_idx]
            y_inner_train = y_train.loc[inner_train_idx]
            X_inner_val = features.loc[inner_val_idx]
            y_inner_val = y_train.loc[inner_val_idx]

            inner_start = time.time()
            tabpfn_model = build_model(
                "tabpfn",
                random_seed=random_seed,
                n_estimators=n_estimators,
                tabpfn_n_estimators=tabpfn_n_estimators,
                tabpfn_device=tabpfn_device,
                tabpfn_kwargs=tabpfn_kwargs,
            )
            extra_trees_model = build_model(
                "extra_trees",
                random_seed=random_seed,
                n_estimators=n_estimators,
                tabpfn_n_estimators=tabpfn_n_estimators,
                tabpfn_device=tabpfn_device,
                tabpfn_kwargs=tabpfn_kwargs,
            )
            tabpfn_model.fit(X_inner_train, y_inner_train)
            extra_trees_model.fit(X_inner_train, y_inner_train)
            fit_seconds = time.time() - inner_start

            predict_start = time.time()
            tabpfn_val_pred = predict_in_batches(
                tabpfn_model, X_inner_val, tabpfn_predict_batch_size
            )
            extra_trees_val_pred = predict_in_batches(
                extra_trees_model, X_inner_val, None
            )
            predict_seconds = time.time() - predict_start
            selected_weight, validation_scan = _select_ensemble_weight_by_mae(
                y_inner_val,
                tabpfn_val_pred,
                extra_trees_val_pred,
                weights,
            )
            selected_row = validation_scan.query(
                "tabpfn_weight == @selected_weight"
            ).iloc[0]
            validation_rows.append(
                {
                    "task": task_name,
                    "feature_set": feature_set,
                    "feature_set_display": feature_set_display,
                    "fold": fold,
                    "inner_train_size": len(X_inner_train),
                    "inner_val_size": len(X_inner_val),
                    "selected_tabpfn_weight": selected_weight,
                    "selected_extra_trees_weight": 1.0 - selected_weight,
                    "selected_inner_val_mae": selected_row["inner_val_mae"],
                    "selected_inner_val_r2": selected_row["inner_val_r2"],
                    "tabpfn_inner_val_mae": validation_scan.query(
                        "tabpfn_weight == 1.0"
                    )["inner_val_mae"].iloc[0],
                    "extra_trees_inner_val_mae": validation_scan.query(
                        "tabpfn_weight == 0.0"
                    )["inner_val_mae"].iloc[0],
                }
            )

            fold_tabpfn = base_predictions.query(
                "task == @task_name and feature_set == @feature_set and "
                "fold == @fold and model == 'tabpfn'"
            ).copy()
            fold_extra_trees = base_predictions.query(
                "task == @task_name and feature_set == @feature_set and "
                "fold == @fold and model == 'extra_trees'"
            ).copy()
            if fold_tabpfn.empty or fold_extra_trees.empty:
                raise ValueError(
                    f"Missing saved official test predictions for {task_name} "
                    f"{feature_set} fold {fold}."
                )
            join_keys = ["task", "feature_set", "fold", "mbid"]
            merged = fold_tabpfn[
                [
                    "task",
                    "target",
                    "unit",
                    "feature_source",
                    "feature_set",
                    "feature_set_display",
                    "fold",
                    "mbid",
                    "y_true",
                    "y_pred",
                ]
            ].merge(
                fold_extra_trees[join_keys + ["y_true", "y_pred"]],
                on=join_keys,
                how="inner",
                suffixes=("_tabpfn", "_extra_trees"),
                validate="one_to_one",
            )
            y_true_delta = (
                merged["y_true_tabpfn"].to_numpy()
                - merged["y_true_extra_trees"].to_numpy()
            )
            if not np.allclose(y_true_delta, 0.0, rtol=0.0, atol=1e-10):
                raise ValueError(
                    f"Saved component predictions have mismatched y_true values for "
                    f"{task_name} {feature_set} fold {fold}."
                )

            y_pred = (
                selected_weight * merged["y_pred_tabpfn"]
                + (1.0 - selected_weight) * merged["y_pred_extra_trees"]
            )
            y_true = merged["y_true_tabpfn"]
            fold_predictions = pd.DataFrame(
                {
                    "task": task_name,
                    "target": target_col,
                    "unit": unit,
                    "feature_source": feature_source,
                    "feature_set": feature_set,
                    "feature_set_display": feature_set_display,
                    "model": model_name,
                    "model_display": INNER_TUNED_ENSEMBLE_DISPLAY,
                    "fold": fold,
                    "mbid": merged["mbid"],
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "absolute_error": np.abs(y_true - y_pred),
                    "inner_selected_tabpfn_weight": selected_weight,
                    "inner_selected_extra_trees_weight": 1.0 - selected_weight,
                }
            )
            ensemble_predictions.append(fold_predictions)
            ensemble_metrics.append(
                {
                    "status": "ok",
                    "task": task_name,
                    "target": target_col,
                    "unit": unit,
                    "matbench_input_type": input_type,
                    "feature_source": feature_source,
                    "feature_set": feature_set,
                    "feature_set_display": feature_set_display,
                    "model": model_name,
                    "model_display": INNER_TUNED_ENSEMBLE_DISPLAY,
                    "fold": fold,
                    "train_size": len(X_train_raw),
                    "test_size": len(y_test),
                    "n_features": features.shape[1],
                    "mae": mean_absolute_error(y_true, y_pred),
                    "r2": r2_score(y_true, y_pred),
                    "fit_seconds": fit_seconds,
                    "predict_seconds": predict_seconds,
                    "random_seed": random_seed,
                    "inner_val_fraction": inner_val_fraction,
                    "inner_selected_tabpfn_weight": selected_weight,
                    "inner_selected_extra_trees_weight": 1.0 - selected_weight,
                    "inner_selected_val_mae": selected_row["inner_val_mae"],
                }
            )

    if validation_rows:
        pd.DataFrame(validation_rows).to_csv(
            paths.tables / f"{model_name}_inner_validation_weights.csv",
            index=False,
        )
    if ensemble_predictions:
        ensemble_predictions_df = pd.concat(ensemble_predictions, ignore_index=True)
        base_predictions = pd.concat(
            [base_predictions, ensemble_predictions_df], ignore_index=True
        )
        ensemble_predictions_df.to_csv(
            paths.predictions / f"{model_name}_predictions.csv", index=False
        )
    if ensemble_metrics:
        ensemble_metrics_df = pd.DataFrame(ensemble_metrics)
        base_metrics = pd.concat([base_metrics, ensemble_metrics_df], ignore_index=True)
        ensemble_metrics_df.to_csv(
            paths.metrics / f"{model_name}_fold_metrics.csv", index=False
        )

    return _write_evaluation_outputs(paths, base_metrics, base_predictions)


def run_official_fold_experiment(
    *,
    tasks: Sequence[str],
    models: Sequence[str],
    paths: RunPaths,
    feature_sets: Sequence[str] | None = None,
    random_seed: int = RANDOM_SEED,
    n_estimators: int = 500,
    tabpfn_n_estimators: int = 8,
    tabpfn_device: str = "cuda",
    tabpfn_predict_batch_size: int | None = 128,
    use_feature_cache: bool = True,
    show_feature_progress: bool = True,
    feature_n_jobs: int = 1,
    continue_on_error: bool = False,
    tabpfn_kwargs: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run selected models on official Matbench folds and save tidy outputs."""

    metrics: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    feature_sets = list(feature_sets or ["magpie"])

    for task_name in tasks:
        task_start = time.time()
        task = load_matbench_task(task_name)
        target_col = task.metadata["target"]
        unit = task.metadata.get("unit")
        input_type = task.metadata["input_type"]

        for feature_set in feature_sets:
            if feature_set_requires_structure(feature_set) and input_type != "structure":
                print(f"{task_name} | {feature_set} | skipped; task is not structure input")
                continue

            feature_start = time.time()
            features, feature_source, feature_set_display = build_task_features(
                task_name,
                task,
                feature_set,
                paths.features,
                use_cache=use_feature_cache,
                show_progress=show_feature_progress,
                n_jobs=feature_n_jobs,
            )

            write_json(
                paths.logs / f"{task_name}_{feature_set}_metadata.json",
                {
                    "task": task_name,
                    "target": target_col,
                    "unit": unit,
                    "matbench_input_type": input_type,
                    "feature_source": feature_source,
                    "feature_set": feature_set,
                    "feature_set_display": feature_set_display,
                    "n_samples": len(task.df),
                    "n_features": features.shape[1],
                    "folds": list(task.folds_nums),
                },
            )

            for model_name in models:
                model_predictions: list[pd.DataFrame] = []
                model_metrics: list[dict[str, Any]] = []

                for fold in task.folds_nums:
                    print(f"{task_name} | {feature_set} | {model_name} | fold {fold}")
                    X_train_raw, y_train = task.get_train_and_val_data(fold)
                    X_test_raw, y_test = task.get_test_data(fold, include_target=True)
                    X_train = features.loc[X_train_raw.index]
                    X_test = features.loc[X_test_raw.index]

                    try:
                        model = build_model(
                            model_name,
                            random_seed=random_seed,
                            n_estimators=n_estimators,
                            tabpfn_n_estimators=tabpfn_n_estimators,
                            tabpfn_device=tabpfn_device,
                            tabpfn_kwargs=tabpfn_kwargs,
                        )
                        fit_start = time.time()
                        model.fit(X_train, y_train)
                        fit_seconds = time.time() - fit_start

                        predict_start = time.time()
                        batch_size = (
                            tabpfn_predict_batch_size if model_name == "tabpfn" else None
                        )
                        y_pred = predict_in_batches(model, X_test, batch_size)
                        predict_seconds = time.time() - predict_start

                        fold_mae = mean_absolute_error(y_test, y_pred)
                        fold_r2 = r2_score(y_test, y_pred)

                        row = {
                            "status": "ok",
                            "task": task_name,
                            "target": target_col,
                            "unit": unit,
                            "matbench_input_type": input_type,
                            "feature_source": feature_source,
                            "feature_set": feature_set,
                            "feature_set_display": feature_set_display,
                            "model": model_name,
                            "model_display": MODEL_DISPLAY_NAMES.get(
                                model_name, model_name
                            ),
                            "fold": fold,
                            "train_size": len(X_train),
                            "test_size": len(X_test),
                            "n_features": features.shape[1],
                            "mae": fold_mae,
                            "r2": fold_r2,
                            "fit_seconds": fit_seconds,
                            "predict_seconds": predict_seconds,
                            "random_seed": random_seed,
                        }
                        model_metrics.append(row)
                        metrics.append(row)

                        fold_predictions = pd.DataFrame(
                            {
                                "task": task_name,
                                "target": target_col,
                                "unit": unit,
                                "feature_source": feature_source,
                                "feature_set": feature_set,
                                "feature_set_display": feature_set_display,
                                "model": model_name,
                                "model_display": MODEL_DISPLAY_NAMES.get(
                                    model_name, model_name
                                ),
                                "fold": fold,
                                "mbid": y_test.index,
                                "y_true": y_test.to_numpy(),
                                "y_pred": y_pred,
                                "absolute_error": np.abs(y_test.to_numpy() - y_pred),
                            }
                        )
                        model_predictions.append(fold_predictions)
                        predictions.append(fold_predictions)

                    except Exception as exc:
                        error_row = {
                            "status": "error",
                            "task": task_name,
                            "target": target_col,
                            "unit": unit,
                            "matbench_input_type": input_type,
                            "feature_source": feature_source,
                            "feature_set": feature_set,
                            "feature_set_display": feature_set_display,
                            "model": model_name,
                            "model_display": MODEL_DISPLAY_NAMES.get(
                                model_name, model_name
                            ),
                            "fold": fold,
                            "train_size": len(X_train),
                            "test_size": len(X_test),
                            "n_features": features.shape[1],
                            "mae": np.nan,
                            "r2": np.nan,
                            "fit_seconds": np.nan,
                            "predict_seconds": np.nan,
                            "random_seed": random_seed,
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                        }
                        model_metrics.append(error_row)
                        metrics.append(error_row)
                        _save_combined_outputs(paths, metrics, predictions)
                        if not continue_on_error:
                            raise

                if model_metrics:
                    pd.DataFrame(model_metrics).to_csv(
                        paths.metrics
                        / f"{task_name}_{feature_set}_{model_name}_fold_metrics.csv",
                        index=False,
                    )
                if model_predictions:
                    pd.concat(model_predictions, ignore_index=True).to_csv(
                        paths.predictions
                        / f"{task_name}_{feature_set}_{model_name}_predictions.csv",
                        index=False,
                    )
                _save_combined_outputs(paths, metrics, predictions)

            write_json(
                paths.logs / f"{task_name}_{feature_set}_run_timing.json",
                {
                    "task": task_name,
                    "feature_set": feature_set,
                    "elapsed_seconds": time.time() - feature_start,
                },
            )

        write_json(
            paths.logs / f"{task_name}_run_timing.json",
            {"task": task_name, "elapsed_seconds": time.time() - task_start},
        )

    return _save_combined_outputs(paths, metrics, predictions)
