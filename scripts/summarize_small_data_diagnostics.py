"""Post-process small-data diagnostic outputs.

This script reads one `results/small_data_diagnostics/<run_id>` folder generated
by `run_small_data_diagnostics.py` and adds three lightweight report extensions:

1. Area under the learning curve (AULC) as a data-efficiency score.
2. TabPFN/ExtraTrees disagreement as a simple error-risk proxy.
3. High-target-regime improvement from adding structure descriptors.

It does not rerun models and does not touch the main benchmark outputs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _latest_run_dir(base_dir: Path) -> Path:
    candidates = [path for path in base_dir.glob("*") if path.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No diagnostic run folders found under {base_dir}")
    return sorted(candidates, key=lambda path: path.stat().st_mtime)[-1]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize a small-data diagnostic run with report-ready tables and figures."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Path to a results/small_data_diagnostics/<run_id> folder. Defaults to latest.",
    )
    parser.add_argument(
        "--high-disagreement-quantile",
        type=float,
        default=0.75,
        help="Quantile threshold used to mark high-disagreement predictions.",
    )
    return parser.parse_args()


def _read_run(run_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_path = run_dir / "tables" / "small_data_learning_curve_summary.csv"
    regime_path = run_dir / "tables" / "target_regime_summary.csv"
    predictions_path = run_dir / "predictions" / "small_data_predictions.csv"
    missing = [
        path
        for path in [summary_path, regime_path, predictions_path]
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing required diagnostic output(s): "
            + ", ".join(str(path) for path in missing)
        )
    return (
        pd.read_csv(summary_path),
        pd.read_csv(regime_path),
        pd.read_csv(predictions_path),
    )


def compute_aulc(summary_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["task", "feature_set", "feature_set_display", "model", "model_display"]
    for keys, group in summary_df.groupby(group_cols, sort=False):
        group = group.sort_values("train_fraction")
        fractions = group["train_fraction"].to_numpy(dtype=float)
        maes = group["mean_mae"].to_numpy(dtype=float)
        if len(group) < 2:
            aulc = np.nan
        else:
            aulc = float(np.trapz(maes, fractions) / (fractions.max() - fractions.min()))
        rows.append(
            {
                **dict(zip(group_cols, keys)),
                "n_curve_points": len(group),
                "fraction_min": float(fractions.min()),
                "fraction_max": float(fractions.max()),
                "aulc_mean_mae": aulc,
                "full_train_mae": float(
                    group.sort_values("train_fraction").iloc[-1]["mean_mae"]
                ),
            }
        )
    aulc_df = pd.DataFrame(rows)
    if not aulc_df.empty:
        aulc_df["rank_by_aulc"] = (
            aulc_df.groupby("task")["aulc_mean_mae"]
            .rank(method="dense", ascending=True)
            .astype(int)
        )
    return aulc_df.sort_values(["task", "rank_by_aulc", "aulc_mean_mae"])


def compute_high_target_improvement(regime_df: pd.DataFrame) -> pd.DataFrame:
    full = regime_df.query("train_fraction == 1.0 and target_regime == 'high'").copy()
    rows = []
    for (task, model), group in full.groupby(["task", "model"], sort=False):
        proxy = group.query("feature_set == 'magpie'")
        structure = group.query("feature_set == 'magpie_structure_all'")
        if proxy.empty or structure.empty:
            continue
        proxy_row = proxy.iloc[0]
        structure_row = structure.iloc[0]
        proxy_mae = float(proxy_row["mean_regime_mae"])
        structure_mae = float(structure_row["mean_regime_mae"])
        rows.append(
            {
                "task": task,
                "model": model,
                "model_display": structure_row["model_display"],
                "target_regime": "high",
                "composition_high_regime_mae": proxy_mae,
                "structure_high_regime_mae": structure_mae,
                "mae_delta_structure_minus_composition": structure_mae - proxy_mae,
                "mae_pct_change_structure_minus_composition": 100.0
                * (structure_mae - proxy_mae)
                / proxy_mae,
            }
        )
    return pd.DataFrame(rows).sort_values(["task", "model"])


def compute_disagreement(predictions_df: pd.DataFrame, high_quantile: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    full = predictions_df.query("train_fraction == 1.0").copy()
    join_keys = ["task", "feature_set", "fold", "repeat", "mbid"]
    tabpfn = full.query("model == 'tabpfn'").copy()
    extra = full.query("model == 'extra_trees'").copy()
    if tabpfn.empty or extra.empty:
        return pd.DataFrame(), pd.DataFrame()

    keep = [
        "task",
        "target",
        "unit",
        "feature_set",
        "feature_set_display",
        "fold",
        "repeat",
        "mbid",
        "y_true",
        "y_pred",
        "absolute_error",
        "target_regime",
    ]
    paired = tabpfn[keep].merge(
        extra[join_keys + ["y_true", "y_pred", "absolute_error"]],
        on=join_keys,
        how="inner",
        suffixes=("_tabpfn", "_extra_trees"),
        validate="one_to_one",
    )
    paired["model_disagreement_abs"] = (
        paired["y_pred_tabpfn"] - paired["y_pred_extra_trees"]
    ).abs()
    paired["mean_absolute_error_between_models"] = (
        paired["absolute_error_tabpfn"] + paired["absolute_error_extra_trees"]
    ) / 2.0

    rows = []
    for (task, feature_set), group in paired.groupby(["task", "feature_set"], sort=False):
        threshold = group["model_disagreement_abs"].quantile(high_quantile)
        high = group.query("model_disagreement_abs >= @threshold")
        low = group.query("model_disagreement_abs < @threshold")
        rows.append(
            {
                "task": task,
                "feature_set": feature_set,
                "feature_set_display": group["feature_set_display"].iloc[0],
                "n_samples": len(group),
                "high_disagreement_quantile": high_quantile,
                "high_disagreement_threshold": threshold,
                "spearman_disagreement_vs_tabpfn_error": group[
                    ["model_disagreement_abs", "absolute_error_tabpfn"]
                ].corr(method="spearman").iloc[0, 1],
                "spearman_disagreement_vs_mean_model_error": group[
                    ["model_disagreement_abs", "mean_absolute_error_between_models"]
                ].corr(method="spearman").iloc[0, 1],
                "tabpfn_mae_high_disagreement": high["absolute_error_tabpfn"].mean(),
                "tabpfn_mae_low_disagreement": low["absolute_error_tabpfn"].mean(),
                "tabpfn_mae_delta_high_minus_low": high[
                    "absolute_error_tabpfn"
                ].mean()
                - low["absolute_error_tabpfn"].mean(),
            }
        )
    return paired, pd.DataFrame(rows).sort_values(["task", "feature_set"])


def _plot_aulc(aulc_df: pd.DataFrame, figures_dir: Path) -> Path | None:
    if aulc_df.empty:
        return None
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="paper")
    plot_df = aulc_df.copy()
    plot_df["model_feature"] = (
        plot_df["model_display"] + "\n" + plot_df["feature_set_display"]
    )
    tasks = list(plot_df["task"].drop_duplicates())
    fig, axes = plt.subplots(len(tasks), 1, figsize=(8.5, 3.5 * len(tasks)), squeeze=False)
    for ax, task in zip(axes.reshape(-1), tasks):
        task_df = plot_df.query("task == @task").sort_values("aulc_mean_mae")
        sns.barplot(data=task_df, x="model_feature", y="aulc_mean_mae", ax=ax)
        ax.set_title(f"{task}: area under learning curve")
        ax.set_xlabel("")
        ax.set_ylabel("AULC mean MAE, lower is better")
        ax.tick_params(axis="x", labelrotation=25)
    fig.tight_layout()
    path = figures_dir / "aulc_data_efficiency_ranking.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_disagreement(paired_df: pd.DataFrame, figures_dir: Path) -> Path | None:
    if paired_df.empty:
        return None
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="paper")
    plot_df = paired_df.copy()
    plot_df["feature_label"] = plot_df["feature_set_display"]
    tasks = list(plot_df["task"].drop_duplicates())
    fig, axes = plt.subplots(len(tasks), 1, figsize=(8.5, 3.8 * len(tasks)), squeeze=False)
    for ax, task in zip(axes.reshape(-1), tasks):
        task_df = plot_df.query("task == @task")
        sns.scatterplot(
            data=task_df,
            x="model_disagreement_abs",
            y="absolute_error_tabpfn",
            hue="feature_label",
            style="target_regime",
            alpha=0.72,
            ax=ax,
        )
        ax.set_title(f"{task}: TabPFN/ExtraTrees disagreement vs TabPFN error")
        ax.set_xlabel("|TabPFN prediction - ExtraTrees prediction|")
        ax.set_ylabel("TabPFN absolute error")
        ax.legend(fontsize=8, title=None)
    fig.tight_layout()
    path = figures_dir / "model_disagreement_error_proxy.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_high_target(high_df: pd.DataFrame, figures_dir: Path) -> Path | None:
    if high_df.empty:
        return None
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="whitegrid", context="paper")
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    plot_df = high_df.copy()
    plot_df["label"] = plot_df["task"] + "\n" + plot_df["model_display"]
    colors = [
        "#3f8f5f" if value < 0 else "#b44c4c"
        for value in plot_df["mae_pct_change_structure_minus_composition"]
    ]
    ax.bar(
        plot_df["label"],
        plot_df["mae_pct_change_structure_minus_composition"],
        color=colors,
        alpha=0.9,
    )
    ax.axhline(0, color="#444444", linewidth=1)
    ax.set_ylabel("High-regime MAE change from adding structure (%)")
    ax.set_title("Structure descriptors reduce high-target-regime error")
    ax.tick_params(axis="x", labelrotation=20)
    fig.tight_layout()
    path = figures_dir / "high_target_structure_improvement.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return path


def write_summary(
    *,
    aulc_df: pd.DataFrame,
    high_df: pd.DataFrame,
    disagreement_summary_df: pd.DataFrame,
    path: Path,
) -> str:
    lines = ["# Extended Small-Data Diagnostic Summary", ""]
    if not aulc_df.empty:
        lines.append("## Data-Efficiency Ranking")
        lines.append("")
        for task, group in aulc_df.groupby("task", sort=False):
            best = group.sort_values("aulc_mean_mae").iloc[0]
            lines.append(
                "- `{}`: lowest AULC is {} with {}, AULC={:.4g}.".format(
                    task,
                    best["model_display"],
                    best["feature_set_display"],
                    best["aulc_mean_mae"],
                )
            )
        lines.append("")

    if not high_df.empty:
        lines.append("## High-Target Regime")
        lines.append("")
        for _, row in high_df.iterrows():
            direction = "reduces" if row["mae_pct_change_structure_minus_composition"] < 0 else "increases"
            lines.append(
                "- `{}` / {}: all-structure {} high-regime MAE by {:+.2f}%.".format(
                    row["task"],
                    row["model_display"],
                    direction,
                    row["mae_pct_change_structure_minus_composition"],
                )
            )
        lines.append("")

    if not disagreement_summary_df.empty:
        lines.append("## Disagreement As Error Proxy")
        lines.append("")
        for _, row in disagreement_summary_df.iterrows():
            lines.append(
                "- `{}` / {}: Spearman(disagreement, TabPFN error)={:.3g}; "
                "high-disagreement TabPFN MAE is {:.4g} vs {:.4g} for lower-disagreement samples.".format(
                    row["task"],
                    row["feature_set_display"],
                    row["spearman_disagreement_vs_tabpfn_error"],
                    row["tabpfn_mae_high_disagreement"],
                    row["tabpfn_mae_low_disagreement"],
                )
            )

    text = "\n".join(lines).strip() + "\n"
    path.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    args = _parse_args()
    base_dir = PROJECT_ROOT / "results" / "small_data_diagnostics"
    run_dir = args.run_dir.resolve() if args.run_dir else _latest_run_dir(base_dir)
    figures_dir = run_dir / "figures"
    tables_dir = run_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    summary_df, regime_df, predictions_df = _read_run(run_dir)
    aulc_df = compute_aulc(summary_df)
    high_df = compute_high_target_improvement(regime_df)
    paired_df, disagreement_summary_df = compute_disagreement(
        predictions_df, args.high_disagreement_quantile
    )

    aulc_df.to_csv(tables_dir / "aulc_data_efficiency_summary.csv", index=False)
    high_df.to_csv(tables_dir / "high_target_structure_improvement.csv", index=False)
    paired_df.to_csv(tables_dir / "model_disagreement_paired_predictions.csv", index=False)
    disagreement_summary_df.to_csv(
        tables_dir / "model_disagreement_error_summary.csv", index=False
    )

    _plot_aulc(aulc_df, figures_dir)
    _plot_disagreement(paired_df, figures_dir)
    _plot_high_target(high_df, figures_dir)
    text = write_summary(
        aulc_df=aulc_df,
        high_df=high_df,
        disagreement_summary_df=disagreement_summary_df,
        path=tables_dir / "extended_small_data_diagnostic_summary.md",
    )
    print(text)
    print(f"Updated run folder: {run_dir}")


if __name__ == "__main__":
    main()
