"""Automatic result summaries and error-analysis tables."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .paths import RunPaths
from .settings import MODEL_DISPLAY_NAMES


ENSEMBLE_MODEL = "tabpfn_extra_trees_ensemble"
INNER_TUNED_ENSEMBLE_MODEL = "tabpfn_extra_trees_inner_tuned"
WEIGHT_SCAN_NOTE = "diagnostic_test_fold_scan_not_for_final_model_selection"
REPORT_EXCLUDED_MODELS = frozenset({ENSEMBLE_MODEL, INNER_TUNED_ENSEMBLE_MODEL})


def best_models_by_task(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()
    return (
        summary_df.sort_values(["task", "mean_mae"])
        .groupby("task", as_index=False)
        .first()
        .sort_values("task")
        .reset_index(drop=True)
    )


def best_standalone_models_by_task(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty or "model" not in summary_df.columns:
        return pd.DataFrame()
    standalone = summary_df.loc[
        ~summary_df["model"].isin(REPORT_EXCLUDED_MODELS)
    ].copy()
    return best_models_by_task(standalone)


def structure_feature_branch_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty or "feature_set" not in summary_df.columns:
        return pd.DataFrame()

    data = summary_df.query("matbench_input_type == 'structure'").copy()
    if data.empty or data["feature_set"].nunique() < 2:
        return pd.DataFrame()

    best_by_feature = (
        data.sort_values("mean_mae")
        .groupby(["task", "feature_set", "feature_set_display"], as_index=False)
        .first()
    )
    rows = []
    for task_name, task_df in best_by_feature.groupby("task", sort=False):
        proxy = task_df.query("feature_set == 'magpie'")
        branches = task_df.query("feature_set != 'magpie'").copy()
        if proxy.empty or branches.empty:
            continue
        proxy_row = proxy.iloc[0]
        branches = branches.sort_values("mean_mae").reset_index(drop=True)
        for rank, (_, branch_row) in enumerate(branches.iterrows(), start=1):
            delta = branch_row["mean_mae"] - proxy_row["mean_mae"]
            pct = 100.0 * delta / proxy_row["mean_mae"]
            rows.append(
                {
                    "task": task_name,
                    "proxy_best_model": proxy_row["model"],
                    "proxy_best_model_display": proxy_row["model_display"],
                    "proxy_mean_mae": proxy_row["mean_mae"],
                    "structure_feature_set": branch_row["feature_set"],
                    "structure_feature_set_display": branch_row[
                        "feature_set_display"
                    ],
                    "structure_branch_rank_by_mae": rank,
                    "structure_best_model": branch_row["model"],
                    "structure_best_model_display": branch_row["model_display"],
                    "structure_mean_mae": branch_row["mean_mae"],
                    "mae_delta_structure_minus_proxy": delta,
                    "mae_pct_change_structure_minus_proxy": pct,
                }
            )
    return pd.DataFrame(rows)


def top_absolute_errors(
    predictions_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    *,
    top_n: int = 10,
) -> pd.DataFrame:
    if predictions_df.empty or summary_df.empty:
        return pd.DataFrame()

    best = best_models_by_task(summary_df)[["task", "feature_set", "model"]]
    rows = []
    for _, best_row in best.iterrows():
        task_name = best_row["task"]
        feature_set = best_row["feature_set"]
        model_name = best_row["model"]
        subset = predictions_df.query(
            "task == @task_name and feature_set == @feature_set and model == @model_name"
        ).copy()
        if subset.empty:
            continue
        subset = subset.sort_values("absolute_error", ascending=False).head(top_n)
        subset["analysis_group"] = "best_model_by_task"
        rows.append(subset)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def tabpfn_vs_best_baseline_sample_errors(
    predictions_df: pd.DataFrame,
    baseline_comparison_df: pd.DataFrame,
    *,
    top_n: int = 20,
) -> pd.DataFrame:
    if predictions_df.empty or baseline_comparison_df.empty:
        return pd.DataFrame()

    rows = []
    tabpfn_rows = baseline_comparison_df.query("model == 'tabpfn'").copy()
    for _, row in tabpfn_rows.iterrows():
        tabpfn_pred = predictions_df.query(
            "task == @row.task and feature_set == @row.feature_set and model == 'tabpfn'"
        ).copy()
        baseline_pred = predictions_df.query(
            "task == @row.task and "
            "feature_set == @row.best_baseline_feature_set and "
            "model == @row.best_baseline_model"
        ).copy()
        if tabpfn_pred.empty or baseline_pred.empty:
            continue

        merged = tabpfn_pred.merge(
            baseline_pred[
                [
                    "task",
                    "fold",
                    "mbid",
                    "feature_set",
                    "model",
                    "y_pred",
                    "absolute_error",
                ]
            ],
            on=["task", "fold", "mbid"],
            how="inner",
            suffixes=("_tabpfn", "_baseline"),
        )
        if merged.empty:
            continue

        merged = merged.rename(
            columns={
                "feature_set_tabpfn": "tabpfn_feature_set",
                "feature_set_baseline": "baseline_feature_set",
                "model_baseline": "baseline_model",
                "y_pred_tabpfn": "tabpfn_y_pred",
                "y_pred_baseline": "baseline_y_pred",
                "absolute_error_tabpfn": "tabpfn_absolute_error",
                "absolute_error_baseline": "baseline_absolute_error",
            }
        )
        merged["tabpfn_error_minus_baseline_error"] = (
            merged["tabpfn_absolute_error"] - merged["baseline_absolute_error"]
        )
        merged["tabpfn_error_ratio_vs_baseline"] = (
            merged["tabpfn_absolute_error"] / merged["baseline_absolute_error"].replace(0, pd.NA)
        )
        rows.append(
            merged.sort_values(
                "tabpfn_error_minus_baseline_error", ascending=False
            ).head(top_n)
        )

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def tabpfn_extra_trees_ensemble_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty or "model" not in summary_df.columns:
        return pd.DataFrame()

    rows = []
    ensemble_rows = summary_df.query("model == @ENSEMBLE_MODEL").copy()
    for _, ensemble in ensemble_rows.iterrows():
        task_name = ensemble["task"]
        feature_set = ensemble["feature_set"]
        same_branch = summary_df.query(
            "task == @task_name and feature_set == @feature_set"
        ).copy()
        tabpfn = same_branch.query("model == 'tabpfn'")
        extra_trees = same_branch.query("model == 'extra_trees'")
        if tabpfn.empty or extra_trees.empty:
            continue
        tabpfn_row = tabpfn.iloc[0]
        extra_trees_row = extra_trees.iloc[0]
        ensemble_mae = ensemble["mean_mae"]
        tabpfn_mae = tabpfn_row["mean_mae"]
        extra_trees_mae = extra_trees_row["mean_mae"]
        rows.append(
            {
                "task": ensemble["task"],
                "feature_set": ensemble["feature_set"],
                "feature_set_display": ensemble["feature_set_display"],
                "ensemble_model": ENSEMBLE_MODEL,
                "ensemble_model_display": MODEL_DISPLAY_NAMES[ENSEMBLE_MODEL],
                "ensemble_mean_mae": ensemble_mae,
                "tabpfn_mean_mae": tabpfn_mae,
                "extra_trees_mean_mae": extra_trees_mae,
                "mae_delta_ensemble_minus_tabpfn": ensemble_mae - tabpfn_mae,
                "mae_pct_change_ensemble_minus_tabpfn": 100.0
                * (ensemble_mae - tabpfn_mae)
                / tabpfn_mae,
                "mae_delta_ensemble_minus_extra_trees": ensemble_mae
                - extra_trees_mae,
                "mae_pct_change_ensemble_minus_extra_trees": 100.0
                * (ensemble_mae - extra_trees_mae)
                / extra_trees_mae,
            }
        )
    return pd.DataFrame(rows)


def tabpfn_extra_trees_inner_tuned_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty or "model" not in summary_df.columns:
        return pd.DataFrame()

    rows = []
    tuned_rows = summary_df.query("model == @INNER_TUNED_ENSEMBLE_MODEL").copy()
    for _, tuned in tuned_rows.iterrows():
        task_name = tuned["task"]
        feature_set = tuned["feature_set"]
        same_branch = summary_df.query(
            "task == @task_name and feature_set == @feature_set"
        ).copy()
        tabpfn = same_branch.query("model == 'tabpfn'")
        fixed = same_branch.query("model == @ENSEMBLE_MODEL")
        extra_trees = same_branch.query("model == 'extra_trees'")
        if tabpfn.empty or extra_trees.empty:
            continue
        tabpfn_mae = tabpfn.iloc[0]["mean_mae"]
        extra_trees_mae = extra_trees.iloc[0]["mean_mae"]
        fixed_mae = fixed.iloc[0]["mean_mae"] if not fixed.empty else np.nan
        tuned_mae = tuned["mean_mae"]
        rows.append(
            {
                "task": tuned["task"],
                "feature_set": tuned["feature_set"],
                "feature_set_display": tuned["feature_set_display"],
                "inner_tuned_model": INNER_TUNED_ENSEMBLE_MODEL,
                "inner_tuned_model_display": MODEL_DISPLAY_NAMES[
                    INNER_TUNED_ENSEMBLE_MODEL
                ],
                "inner_tuned_mean_mae": tuned_mae,
                "tabpfn_mean_mae": tabpfn_mae,
                "extra_trees_mean_mae": extra_trees_mae,
                "fixed_50_50_mean_mae": fixed_mae,
                "mae_delta_inner_tuned_minus_tabpfn": tuned_mae - tabpfn_mae,
                "mae_pct_change_inner_tuned_minus_tabpfn": 100.0
                * (tuned_mae - tabpfn_mae)
                / tabpfn_mae,
                "mae_delta_inner_tuned_minus_fixed_50_50": tuned_mae - fixed_mae,
                "mae_pct_change_inner_tuned_minus_fixed_50_50": 100.0
                * (tuned_mae - fixed_mae)
                / fixed_mae,
            }
        )
    return pd.DataFrame(rows)


def _r2_score(y_true: pd.Series, y_pred: pd.Series) -> float:
    y_true_array = pd.to_numeric(y_true, errors="coerce").to_numpy(dtype=float)
    y_pred_array = pd.to_numeric(y_pred, errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(y_true_array) & np.isfinite(y_pred_array)
    y_true_array = y_true_array[valid]
    y_pred_array = y_pred_array[valid]
    if len(y_true_array) == 0:
        return np.nan
    denominator = np.sum((y_true_array - y_true_array.mean()) ** 2)
    if denominator == 0:
        return np.nan
    numerator = np.sum((y_true_array - y_pred_array) ** 2)
    return float(1.0 - numerator / denominator)


def tabpfn_extra_trees_weight_scan(
    predictions_df: pd.DataFrame,
    *,
    weights: np.ndarray | None = None,
) -> pd.DataFrame:
    """Diagnostic test-fold scan of TabPFN/ExtraTrees ensemble weights."""

    if predictions_df.empty or "model" not in predictions_df.columns:
        return pd.DataFrame()
    weights = np.round(
        np.asarray(weights if weights is not None else np.linspace(0.0, 1.0, 21)),
        6,
    )
    if len(weights) == 0:
        return pd.DataFrame()

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

    tabpfn = predictions_df.query("model == 'tabpfn'").copy()
    extra_trees = predictions_df.query("model == 'extra_trees'").copy()
    if tabpfn.empty or extra_trees.empty:
        return pd.DataFrame()

    join_keys = ["task", "feature_set", "fold", "mbid"]
    metadata_columns = [
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
    paired = tabpfn[metadata_columns].merge(
        extra_trees[join_keys + ["y_true", "y_pred"]],
        on=join_keys,
        how="inner",
        suffixes=("_tabpfn", "_extra_trees"),
        validate="one_to_one",
    )
    if paired.empty:
        return pd.DataFrame()

    y_true_delta = paired["y_true_tabpfn"] - paired["y_true_extra_trees"]
    if not np.allclose(y_true_delta.to_numpy(dtype=float), 0.0, rtol=0.0, atol=1e-10):
        raise ValueError("TabPFN and ExtraTrees predictions have mismatched y_true values.")
    paired["y_true"] = paired["y_true_tabpfn"]

    rows = []
    group_cols = [
        "task",
        "target",
        "unit",
        "feature_source",
        "feature_set",
        "feature_set_display",
    ]
    for group_values, group in paired.groupby(group_cols, sort=False):
        group_meta = dict(zip(group_cols, group_values))
        for weight in weights:
            fold_rows = []
            y_pred = (
                weight * group["y_pred_tabpfn"]
                + (1.0 - weight) * group["y_pred_extra_trees"]
            )
            scored = group.assign(y_pred_ensemble=y_pred)
            for fold, fold_df in scored.groupby("fold", sort=False):
                errors = (fold_df["y_true"] - fold_df["y_pred_ensemble"]).abs()
                fold_rows.append(
                    {
                        "fold": fold,
                        "mae": float(errors.mean()),
                        "r2": _r2_score(fold_df["y_true"], fold_df["y_pred_ensemble"]),
                        "test_size": len(fold_df),
                    }
                )
            fold_metrics = pd.DataFrame(fold_rows)
            rows.append(
                {
                    **group_meta,
                    "tabpfn_weight": float(weight),
                    "extra_trees_weight": float(1.0 - weight),
                    "folds_completed": int(fold_metrics["fold"].nunique()),
                    "mean_mae": float(fold_metrics["mae"].mean()),
                    "std_mae": float(fold_metrics["mae"].std(ddof=1)),
                    "sem_mae": float(
                        fold_metrics["mae"].std(ddof=1) / np.sqrt(len(fold_metrics))
                    ),
                    "mean_r2": float(fold_metrics["r2"].mean()),
                    "test_size_sum": int(fold_metrics["test_size"].sum()),
                    "diagnostic_note": WEIGHT_SCAN_NOTE,
                }
            )

    scan_df = pd.DataFrame(rows)
    if scan_df.empty:
        return scan_df
    scan_df["rank_by_mae_within_branch"] = scan_df.groupby(
        ["task", "feature_set"]
    )["mean_mae"].rank(method="dense", ascending=True).astype(int)
    return scan_df.sort_values(
        ["task", "feature_set", "rank_by_mae_within_branch", "tabpfn_weight"]
    ).reset_index(drop=True)


def best_tabpfn_extra_trees_weight_scan(scan_df: pd.DataFrame) -> pd.DataFrame:
    if scan_df.empty:
        return pd.DataFrame()

    rows = []
    for (task_name, feature_set), branch in scan_df.groupby(
        ["task", "feature_set"], sort=False
    ):
        branch = branch.copy()
        best = branch.sort_values(["mean_mae", "tabpfn_weight"]).iloc[0]
        tabpfn = branch.query("tabpfn_weight == 1.0")
        extra_trees = branch.query("tabpfn_weight == 0.0")
        fixed = branch.query("tabpfn_weight == 0.5")
        tabpfn_mae = tabpfn.iloc[0]["mean_mae"] if not tabpfn.empty else np.nan
        extra_trees_mae = (
            extra_trees.iloc[0]["mean_mae"] if not extra_trees.empty else np.nan
        )
        fixed_mae = fixed.iloc[0]["mean_mae"] if not fixed.empty else np.nan
        rows.append(
            {
                "task": task_name,
                "feature_set": feature_set,
                "feature_set_display": best["feature_set_display"],
                "unit": best["unit"],
                "best_tabpfn_weight": best["tabpfn_weight"],
                "best_extra_trees_weight": best["extra_trees_weight"],
                "best_mean_mae": best["mean_mae"],
                "tabpfn_mean_mae": tabpfn_mae,
                "extra_trees_mean_mae": extra_trees_mae,
                "fixed_50_50_mean_mae": fixed_mae,
                "mae_delta_best_minus_tabpfn": best["mean_mae"] - tabpfn_mae,
                "mae_pct_change_best_minus_tabpfn": 100.0
                * (best["mean_mae"] - tabpfn_mae)
                / tabpfn_mae,
                "mae_delta_best_minus_fixed_50_50": best["mean_mae"] - fixed_mae,
                "mae_pct_change_best_minus_fixed_50_50": 100.0
                * (best["mean_mae"] - fixed_mae)
                / fixed_mae,
                "diagnostic_note": WEIGHT_SCAN_NOTE,
            }
        )
    return pd.DataFrame(rows).sort_values(["task", "best_mean_mae"]).reset_index(
        drop=True
    )


def write_auto_summary_markdown(
    summary_df: pd.DataFrame,
    baseline_comparison_df: pd.DataFrame,
    structure_summary_df: pd.DataFrame,
    path: Path,
    ensemble_summary_df: pd.DataFrame | None = None,
    weight_scan_best_df: pd.DataFrame | None = None,
    inner_tuned_summary_df: pd.DataFrame | None = None,
) -> str:
    lines = ["# Automatic Results Summary", ""]

    best = best_standalone_models_by_task(summary_df)
    if not best.empty:
        lines.extend(["## Best Standalone Model by Task", ""])
        for _, row in best.iterrows():
            lines.append(
                "- `{task}`: {model} with {feature} features, "
                "MAE={mae:.4g} {unit}, R2={r2:.3g}.".format(
                    task=row["task"],
                    model=row["model_display"],
                    feature=row["feature_set_display"],
                    mae=row["mean_mae"],
                    unit=row["unit"],
                    r2=row["mean_r2"],
                )
            )
        lines.append("")

        lines.extend(
            [
                "_Post-hoc ensembles are reported below as diagnostics and are "
                "not used as the headline model-selection result._",
                "",
            ]
        )

    if baseline_comparison_df.empty or "model" not in baseline_comparison_df.columns:
        tabpfn = pd.DataFrame()
    else:
        tabpfn = baseline_comparison_df.query("model == 'tabpfn'").copy()
    if not tabpfn.empty:
        lines.extend(["## TabPFN Claim Check", ""])
        for _, row in tabpfn.sort_values(["task", "feature_set"]).iterrows():
            direction = "improves over" if row["mae_delta_vs_best_baseline"] < 0 else "does not beat"
            lines.append(
                "- `{task}` ({feature}): TabPFN {direction} the best non-TabPFN "
                "baseline ({baseline}, {baseline_feature}) by {pct:+.2f}% MAE.".format(
                    task=row["task"],
                    feature=row["feature_set_display"],
                    direction=direction,
                    baseline=row["best_baseline_display"],
                    baseline_feature=row["best_baseline_feature_set_display"],
                    pct=row["mae_pct_change_vs_best_baseline"],
                )
            )
        lines.append("")

    if not structure_summary_df.empty:
        lines.extend(["## Structure-Aware Feature Check", ""])
        best_structure = (
            structure_summary_df.sort_values("structure_mean_mae")
            .groupby("task", as_index=False)
            .first()
            .sort_values("task")
        )
        for _, row in best_structure.iterrows():
            direction = (
                "improves over"
                if row["mae_delta_structure_minus_proxy"] < 0
                else "does not improve over"
            )
            lines.append(
                "- `{task}`: best structure-aware branch ({feature}) {direction} "
                "the composition-proxy branch by {pct:+.2f}% MAE.".format(
                    task=row["task"],
                    feature=row["structure_feature_set_display"],
                    direction=direction,
                    pct=row["mae_pct_change_structure_minus_proxy"],
                )
            )
        lines.append("")

    if ensemble_summary_df is not None and not ensemble_summary_df.empty:
        lines.extend(["## Fixed 50/50 Ensemble Check", ""])
        best_ensemble = (
            ensemble_summary_df.sort_values("ensemble_mean_mae")
            .groupby("task", as_index=False)
            .first()
            .sort_values("task")
        )
        for _, row in best_ensemble.iterrows():
            direction = (
                "improves over"
                if row["mae_delta_ensemble_minus_tabpfn"] < 0
                else "does not improve over"
            )
            lines.append(
                "- `{task}` ({feature}): the fixed, non-tuned 50/50 TabPFN + ExtraTrees ensemble "
                "{direction} standalone TabPFN by {pct:+.2f}% MAE.".format(
                    task=row["task"],
                    feature=row["feature_set_display"],
                    direction=direction,
                    pct=row["mae_pct_change_ensemble_minus_tabpfn"],
                )
            )
        lines.append("")

    if weight_scan_best_df is not None and not weight_scan_best_df.empty:
        lines.extend(["## Weight Scan Diagnostic", ""])
        task_best = (
            weight_scan_best_df.sort_values("best_mean_mae")
            .groupby("task", as_index=False)
            .first()
            .sort_values("task")
        )
        for _, row in task_best.iterrows():
            direction = (
                "would improve over"
                if row["mae_delta_best_minus_tabpfn"] < 0
                else "would not improve over"
            )
            lines.append(
                "- `{task}` ({feature}): diagnostic test-fold scan selects "
                "TabPFN weight {weight:.2f} and {direction} standalone TabPFN by "
                "{pct:+.2f}% MAE. Do not report this as a final tuned model without "
                "inner-validation weight selection.".format(
                    task=row["task"],
                    feature=row["feature_set_display"],
                    weight=row["best_tabpfn_weight"],
                    direction=direction,
                    pct=row["mae_pct_change_best_minus_tabpfn"],
                )
            )
        lines.append("")

    if inner_tuned_summary_df is not None and not inner_tuned_summary_df.empty:
        lines.extend(["## Inner-Validation Tuned Ensemble Check", ""])
        for _, row in inner_tuned_summary_df.sort_values("task").iterrows():
            direction = (
                "improves over"
                if row["mae_delta_inner_tuned_minus_tabpfn"] < 0
                else "does not improve over"
            )
            lines.append(
                "- `{task}` ({feature}): inner-validation tuned TabPFN + ExtraTrees "
                "{direction} standalone TabPFN by {pct:+.2f}% MAE.".format(
                    task=row["task"],
                    feature=row["feature_set_display"],
                    direction=direction,
                    pct=row["mae_pct_change_inner_tuned_minus_tabpfn"],
                )
            )
        lines.append("")

    text = "\n".join(lines).strip() + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


def create_report_artifacts(
    *,
    metrics_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    baseline_comparison_df: pd.DataFrame,
    paired_comparison_df: pd.DataFrame,
    paths: RunPaths,
    top_n_errors: int = 10,
) -> dict[str, Path | str]:
    """Save tables and markdown that convert raw metrics into report evidence."""

    paths.tables.mkdir(parents=True, exist_ok=True)

    best_df = best_models_by_task(summary_df)
    best_standalone_df = best_standalone_models_by_task(summary_df)
    structure_df = structure_feature_branch_summary(summary_df)
    ensemble_df = tabpfn_extra_trees_ensemble_summary(summary_df)
    inner_tuned_df = tabpfn_extra_trees_inner_tuned_summary(summary_df)
    weight_scan_df = tabpfn_extra_trees_weight_scan(predictions_df)
    weight_scan_best_df = best_tabpfn_extra_trees_weight_scan(weight_scan_df)
    top_errors_df = top_absolute_errors(
        predictions_df, summary_df, top_n=top_n_errors
    )
    tabpfn_errors_df = tabpfn_vs_best_baseline_sample_errors(
        predictions_df, baseline_comparison_df, top_n=top_n_errors
    )

    outputs: dict[str, Path | str] = {}
    tables = {
        "best_models_by_task": best_df,
        "best_standalone_models_by_task": best_standalone_df,
        "structure_feature_branch_summary": structure_df,
        "tabpfn_extra_trees_ensemble_summary": ensemble_df,
        "tabpfn_extra_trees_inner_tuned_summary": inner_tuned_df,
        "tabpfn_extra_trees_weight_scan": weight_scan_df,
        "tabpfn_extra_trees_weight_scan_best": weight_scan_best_df,
        "top_absolute_errors": top_errors_df,
        "tabpfn_vs_best_baseline_sample_errors": tabpfn_errors_df,
        "paired_fold_comparisons": paired_comparison_df,
    }

    for name, table in tables.items():
        path = paths.tables / f"{name}.csv"
        table.to_csv(path, index=False)
        outputs[name] = path

    summary_text = write_auto_summary_markdown(
        summary_df,
        baseline_comparison_df,
        structure_df,
        paths.tables / "auto_summary.md",
        ensemble_summary_df=ensemble_df,
        weight_scan_best_df=weight_scan_best_df,
        inner_tuned_summary_df=inner_tuned_df,
    )
    outputs["auto_summary"] = paths.tables / "auto_summary.md"
    outputs["auto_summary_text"] = summary_text
    return outputs
