"""Unified plotting style and report figures."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .paths import RunPaths
from .settings import MODEL_COLORS, MODEL_DISPLAY_NAMES, DEFAULT_MODELS


PLOT_STYLE_VERSION = "report_structure_ablation_v1"


def set_plot_style() -> None:
    sns.set_theme(
        context="paper",
        style="whitegrid",
        font="DejaVu Sans",
        rc={
            "figure.dpi": 120,
            "savefig.dpi": 220,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.title_fontsize": 9,
            "grid.alpha": 0.25,
        },
    )


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path


def _ordered_models(models: pd.Series) -> list[str]:
    present = list(dict.fromkeys(models.dropna().tolist()))
    ordered = [model for model in DEFAULT_MODELS if model in present]
    ordered.extend([model for model in present if model not in ordered])
    return ordered


def _add_plot_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "feature_set_display" not in df.columns:
        df["feature_set_display"] = df.get("feature_set", "")
    df["plot_label"] = df["model_display"].astype(str)
    for task_name, task_df in df.groupby("task"):
        if task_df["feature_set"].nunique() > 1:
            mask = df["task"] == task_name
            feature_label = df.loc[mask, "feature_set"].map(
                {
                    "magpie": "comp",
                    "magpie_density": "density",
                    "magpie_symmetry": "sym",
                    "magpie_packing": "packing",
                    "magpie_density_packing": "dens+pack",
                    "magpie_density_packing_symmetry": "dens+pack+sym",
                    "magpie_complexity": "complex",
                    "magpie_density_packing_complexity": "dens+pack+complex",
                    "magpie_structure_all": "all",
                    "magpie_structure": "all",
                }
            )
            feature_label = feature_label.fillna(df.loc[mask, "feature_set"].astype(str))
            df.loc[mask, "plot_label"] = (
                df.loc[mask, "model_display"].astype(str) + " | " + feature_label
            )
    return df


def _short_feature_label(feature_set: str, feature_set_display: str | None = None) -> str:
    labels = {
        "magpie": "comp",
        "magpie_density": "density",
        "magpie_symmetry": "sym",
        "magpie_packing": "packing",
        "magpie_density_packing": "dens+pack",
        "magpie_density_packing_symmetry": "dens+pack+sym",
        "magpie_complexity": "complex",
        "magpie_density_packing_complexity": "dens+pack+complex",
        "magpie_structure_all": "all",
        "magpie_structure": "all",
    }
    return labels.get(feature_set, feature_set_display or feature_set)


def _report_feature_label(feature_set: str, feature_set_display: str | None = None) -> str:
    labels = {
        "magpie": "composition proxy",
        "magpie_density": "density",
        "magpie_symmetry": "symmetry",
        "magpie_packing": "packing",
        "magpie_density_packing": "density + packing",
        "magpie_density_packing_symmetry": "density + packing + symmetry",
        "magpie_complexity": "complexity",
        "magpie_density_packing_complexity": "density + packing + complexity",
        "magpie_structure_all": "all structure",
        "magpie_structure": "all structure",
    }
    return labels.get(feature_set, feature_set_display or feature_set)


def _structure_feature_color(feature_set: str) -> str:
    if feature_set == "magpie":
        return "#9aa3ad"
    if feature_set == "magpie_structure_all":
        return "#2f5f8f"
    if "density_packing" in feature_set:
        return "#3f8f7f"
    if feature_set in {"magpie_density", "magpie_packing"}:
        return "#6fa3c7"
    return "#c27d52"


def _robust_fold_axis_upper(values: pd.Series) -> tuple[float, bool]:
    """Return a plot-only y-axis cap that keeps single catastrophic folds visible."""
    clean = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if clean.empty:
        return 1.0, False

    maximum = float(clean.max())
    if maximum <= 0:
        return 1.0, False
    if len(clean) < 8:
        return maximum * 1.12, False

    q1, q3 = np.percentile(clean, [25, 75])
    sorted_values = np.sort(clean.to_numpy())
    p90 = sorted_values[int(math.floor(0.90 * (len(sorted_values) - 1)))]
    iqr = q3 - q1
    whisker_top = q3 + 1.5 * iqr if iqr > 0 else p90
    typical_top = max(float(p90), float(whisker_top))
    if typical_top <= 0:
        return maximum * 1.12, False

    should_clip = maximum > typical_top * 3.0 and (maximum - typical_top) > 0.2 * maximum
    if should_clip:
        return typical_top * 1.18, True
    return maximum * 1.12, False


def _grid_for_tasks(
    tasks: list[str],
    row_counts: dict[str, int],
    *,
    width_per_col: float,
    min_panel_height: float,
    height_per_item: float,
) -> tuple[plt.Figure, np.ndarray]:
    ncols = min(2, len(tasks))
    nrows = math.ceil(len(tasks) / ncols)
    max_items = max(row_counts.values()) if row_counts else 1
    panel_height = max(min_panel_height, 1.1 + height_per_item * max_items)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(width_per_col * ncols, panel_height * nrows),
        squeeze=False,
    )
    return fig, axes.reshape(-1)


def plot_model_mae_comparison(summary_df: pd.DataFrame, figures_dir: Path) -> Path | None:
    if summary_df.empty:
        return None

    set_plot_style()
    tasks = list(summary_df["task"].drop_duplicates())
    row_counts = {
        task_name: len(summary_df.query("task == @task_name")) for task_name in tasks
    }
    fig, axes = _grid_for_tasks(
        tasks,
        row_counts,
        width_per_col=8.5,
        min_panel_height=3.6,
        height_per_item=0.43,
    )

    for ax, task_name in zip(axes, tasks):
        task_summary = _add_plot_labels(
            summary_df.query("task == @task_name").sort_values("mean_mae")
        )
        y = np.arange(len(task_summary))
        colors = [MODEL_COLORS.get(model, "#777777") for model in task_summary["model"]]
        ax.barh(
            y,
            task_summary["mean_mae"],
            xerr=task_summary["sem_mae"].fillna(0),
            color=colors,
            alpha=0.88,
        )
        ax.set_yticks(y)
        ax.set_yticklabels(task_summary["plot_label"])
        ax.invert_yaxis()
        unit = task_summary["unit"].iloc[0]
        ax.set_title(task_name)
        ax.set_xlabel(f"Mean MAE ({unit})")
        xmax = (
            task_summary["mean_mae"] + task_summary["sem_mae"].fillna(0)
        ).max()
        ax.set_xlim(left=0, right=xmax * 1.18 if xmax > 0 else 1)
        for idx, value in enumerate(task_summary["mean_mae"]):
            ax.text(value, idx, f" {value:.3g}", va="center", fontsize=8)
        ax.tick_params(axis="y", labelsize=8)

    for ax in axes[len(tasks) :]:
        ax.axis("off")

    fig.suptitle("Official-fold MAE by task and model", y=0.995, fontsize=13)
    return _save(fig, figures_dir / "01_model_mae_comparison.png")


def plot_fold_mae_distribution(metrics_df: pd.DataFrame, figures_dir: Path) -> Path | None:
    ok = metrics_df.query("status == 'ok'").copy()
    if ok.empty:
        return None

    set_plot_style()
    tasks = list(ok["task"].drop_duplicates())
    row_counts = {
        task_name: ok.query("task == @task_name")["model"].nunique()
        * ok.query("task == @task_name")["feature_set"].nunique()
        for task_name in tasks
    }
    fig, axes = _grid_for_tasks(
        tasks,
        row_counts,
        width_per_col=8.8,
        min_panel_height=4.0,
        height_per_item=0.24,
    )

    for ax, task_name in zip(axes, tasks):
        task_metrics = _add_plot_labels(ok.query("task == @task_name").copy())
        if task_metrics["feature_set"].nunique() > 1:
            display_order = (
                task_metrics.groupby("plot_label")["mae"]
                .mean()
                .sort_values()
                .index.tolist()
            )
        else:
            order = _ordered_models(task_metrics["model"])
            display_order = [MODEL_DISPLAY_NAMES.get(model, model) for model in order]
        sns.boxplot(
            data=task_metrics,
            x="mae",
            y="plot_label",
            order=display_order,
            ax=ax,
            width=0.55,
            color="#d9dee7",
            fliersize=0,
        )
        sns.stripplot(
            data=task_metrics,
            x="mae",
            y="plot_label",
            order=display_order,
            hue="model",
            palette=MODEL_COLORS,
            ax=ax,
            size=4,
            jitter=0.18,
            legend=False,
        )
        unit = task_metrics["unit"].iloc[0]
        ax.set_title(task_name)
        ax.set_xlabel(f"Fold MAE ({unit})")
        ax.set_ylabel("")
        axis_upper, has_clipped = _robust_fold_axis_upper(task_metrics["mae"])
        ax.set_xlim(left=0, right=axis_upper)
        if has_clipped:
            clipped = task_metrics.query("mae > @axis_upper").copy()
            if not clipped.empty:
                label_to_y = {label: idx for idx, label in enumerate(display_order)}
                within_label = clipped.groupby("plot_label").cumcount().astype(float)
                label_count = (
                    clipped.groupby("plot_label")["plot_label"]
                    .transform("size")
                    .astype(float)
                )
                x_positions = np.full(len(clipped), axis_upper * 0.985)
                y_positions = (
                    clipped["plot_label"].map(label_to_y).astype(float)
                    + (within_label - (label_count - 1) / 2) * 0.08
                )
                colors = [MODEL_COLORS.get(model, "#777777") for model in clipped["model"]]
                ax.scatter(
                    x_positions,
                    y_positions,
                    marker=">",
                    s=50,
                    color=colors,
                    edgecolors="white",
                    linewidth=0.7,
                    zorder=5,
                )
                noun = "fold" if len(clipped) == 1 else "folds"
                ax.text(
                    0.98,
                    0.95,
                    f"{len(clipped)} clipped {noun}; max={clipped['mae'].max():.3g} {unit}",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=8,
                    bbox={
                        "boxstyle": "round,pad=0.25",
                        "facecolor": "white",
                        "edgecolor": "#c8cdd4",
                        "alpha": 0.86,
                    },
                )
        ax.tick_params(axis="y", labelsize=7)
        ax.tick_params(axis="x", labelsize=8)

    for ax in axes[len(tasks) :]:
        ax.axis("off")

    fig.suptitle("Fold-level MAE dispersion", y=0.995, fontsize=13)
    return _save(fig, figures_dir / "02_fold_mae_distribution.png")


def plot_tabpfn_vs_baseline(comparison_df: pd.DataFrame, figures_dir: Path) -> Path | None:
    if comparison_df.empty or "tabpfn" not in set(comparison_df["model"]):
        return None

    tabpfn = comparison_df.query("model == 'tabpfn'").copy()
    if tabpfn.empty:
        return None
    tabpfn = _add_plot_labels(tabpfn)
    tabpfn["x_label"] = tabpfn["task"]
    if tabpfn.groupby("task")["feature_set"].transform("nunique").max() > 1:
        tabpfn["x_label"] = tabpfn["task"] + "\n" + tabpfn["feature_set_display"]

    set_plot_style()
    fig, ax = plt.subplots(figsize=(max(8.2, 1.4 * len(tabpfn)), 3.8))
    colors = [
        "#3f8f5f" if value < 0 else "#b44c4c"
        for value in tabpfn["mae_pct_change_vs_best_baseline"]
    ]
    ax.bar(
        tabpfn["x_label"],
        tabpfn["mae_pct_change_vs_best_baseline"],
        color=colors,
        alpha=0.9,
    )
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_ylabel("MAE change vs best non-TabPFN baseline (%)")
    ax.set_xlabel("")
    ax.set_title("TabPFN relative to the best classical baseline")
    ax.tick_params(axis="x", rotation=20)
    for idx, row in tabpfn.reset_index(drop=True).iterrows():
        value = row["mae_pct_change_vs_best_baseline"]
        va = "bottom" if value >= 0 else "top"
        ax.text(idx, value, f"{value:+.1f}%", ha="center", va=va, fontsize=9)
    return _save(fig, figures_dir / "03_tabpfn_vs_best_baseline.png")


def plot_tabpfn_parity(predictions_df: pd.DataFrame, figures_dir: Path) -> Path | None:
    if predictions_df.empty or "tabpfn" not in set(predictions_df["model"]):
        return None

    data = predictions_df.query("model == 'tabpfn'").copy()
    if data.empty:
        return None

    set_plot_style()
    groups = list(data.groupby(["task", "feature_set"], sort=False))
    ncols = min(2, len(groups))
    nrows = math.ceil(len(groups) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.6 * nrows))
    axes = np.atleast_1d(axes).reshape(-1)

    for ax, ((task_name, feature_set), task_pred) in zip(axes, groups):
        ax.scatter(
            task_pred["y_true"],
            task_pred["y_pred"],
            s=18,
            alpha=0.55,
            color=MODEL_COLORS["tabpfn"],
            edgecolors="none",
        )
        low = min(task_pred["y_true"].min(), task_pred["y_pred"].min())
        high = max(task_pred["y_true"].max(), task_pred["y_pred"].max())
        pad = 0.04 * (high - low) if high > low else 1.0
        low -= pad
        high += pad
        ax.plot([low, high], [low, high], color="#333333", linewidth=1, linestyle="--")
        ax.set_xlim(low, high)
        ax.set_ylim(low, high)
        unit = task_pred["unit"].iloc[0]
        mae = task_pred["absolute_error"].mean()
        feature_label = task_pred["feature_set_display"].iloc[0]
        short_feature = _short_feature_label(feature_set, feature_label)
        ax.set_title(task_name, fontsize=11, pad=8)
        ax.text(
            0.03,
            0.97,
            f"{short_feature} | MAE={mae:.3g} {unit}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.25",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.78,
            },
        )
        ax.set_xlabel(f"True ({unit})")
        ax.set_ylabel(f"Predicted ({unit})")

    for ax in axes[len(groups) :]:
        ax.axis("off")

    fig.suptitle("TabPFN parity plots across official folds", y=1.02, fontsize=13)
    return _save(fig, figures_dir / "04_tabpfn_parity.png")


def plot_structure_feature_set_comparison(
    summary_df: pd.DataFrame, figures_dir: Path
) -> Path | None:
    if summary_df.empty or "feature_set" not in summary_df.columns:
        return None

    data = summary_df.query("matbench_input_type == 'structure'").copy()
    if data.empty or data["feature_set"].nunique() < 2:
        return None

    best_by_feature = (
        data.sort_values("mean_mae")
        .groupby(["task", "feature_set", "feature_set_display"], as_index=False)
        .first()
    )

    set_plot_style()
    tasks = list(best_by_feature["task"].drop_duplicates())
    ncols = min(2, len(tasks))
    nrows = math.ceil(len(tasks) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.4 * ncols, 3.4 * nrows))
    axes = np.atleast_1d(axes).reshape(-1)

    for ax, task_name in zip(axes, tasks):
        task_df = best_by_feature.query("task == @task_name").sort_values("mean_mae")
        labels = task_df["feature_set_display"]
        ax.barh(labels, task_df["mean_mae"], color="#607d8b", alpha=0.88)
        ax.invert_yaxis()
        unit = task_df["unit"].iloc[0]
        ax.set_title(task_name)
        ax.set_xlabel(f"Best model mean MAE ({unit})")
        for idx, (_, row) in enumerate(task_df.iterrows()):
            ax.text(
                row["mean_mae"],
                idx,
                f" {row['mean_mae']:.3g} ({row['model_display']})",
                va="center",
                fontsize=8,
            )

    for ax in axes[len(tasks) :]:
        ax.axis("off")

    fig.suptitle("Structure feature branch ablation", y=1.02, fontsize=13)
    return _save(fig, figures_dir / "05_structure_feature_branch_comparison.png")


def plot_tabpfn_structure_ablation_report(
    summary_df: pd.DataFrame, figures_dir: Path
) -> Path | None:
    if summary_df.empty or "feature_set" not in summary_df.columns:
        return None

    data = summary_df.query(
        "matbench_input_type == 'structure' and model == 'tabpfn'"
    ).copy()
    if data.empty or data["feature_set"].nunique() < 2:
        return None

    data["report_feature_label"] = [
        _report_feature_label(feature_set, display)
        for feature_set, display in zip(
            data["feature_set"], data["feature_set_display"]
        )
    ]

    set_plot_style()
    tasks = list(data["task"].drop_duplicates())
    row_counts = {
        task_name: len(data.query("task == @task_name")) for task_name in tasks
    }
    fig, axes = _grid_for_tasks(
        tasks,
        row_counts,
        width_per_col=7.4,
        min_panel_height=4.6,
        height_per_item=0.42,
    )

    for ax, task_name in zip(axes, tasks):
        task_df = data.query("task == @task_name").sort_values("mean_mae").copy()
        task_df = task_df.reset_index(drop=True)
        y = np.arange(len(task_df))
        colors = [_structure_feature_color(value) for value in task_df["feature_set"]]
        ax.barh(y, task_df["mean_mae"], color=colors, alpha=0.9)
        ax.set_yticks(y)
        ax.set_yticklabels(task_df["report_feature_label"])
        ax.invert_yaxis()

        unit = task_df["unit"].iloc[0]
        ax.set_title(task_name)
        ax.set_xlabel(f"TabPFN mean MAE ({unit})")
        ax.set_ylabel("")
        xmax = task_df["mean_mae"].max()
        ax.set_xlim(left=0, right=xmax * 1.34 if xmax > 0 else 1)

        proxy = task_df.query("feature_set == 'magpie'")
        proxy_mae = proxy["mean_mae"].iloc[0] if not proxy.empty else np.nan
        for idx, row in task_df.iterrows():
            value = row["mean_mae"]
            if row["feature_set"] == "magpie" or not np.isfinite(proxy_mae):
                label = f" {value:.3g}"
            else:
                pct = 100.0 * (value - proxy_mae) / proxy_mae
                label = f" {value:.3g} ({pct:+.1f}%)"
            ax.text(value, idx, label, va="center", fontsize=8)
        ax.tick_params(axis="y", labelsize=8)

    for ax in axes[len(tasks) :]:
        ax.axis("off")

    fig.suptitle(
        "TabPFN structure-feature ablation across official folds",
        y=0.995,
        fontsize=13,
    )
    return _save(fig, figures_dir / "06_tabpfn_structure_ablation_report.png")


def save_all_figures(
    metrics_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    paths: RunPaths,
) -> list[Path]:
    figures = [
        plot_model_mae_comparison(summary_df, paths.figures),
        plot_fold_mae_distribution(metrics_df, paths.figures),
        plot_tabpfn_vs_baseline(comparison_df, paths.figures),
        plot_tabpfn_parity(predictions_df, paths.figures),
        plot_structure_feature_set_comparison(summary_df, paths.figures),
        plot_tabpfn_structure_ablation_report(summary_df, paths.figures),
    ]
    return [path for path in figures if path is not None]
