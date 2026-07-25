"""
Build slide-ready figures for the neural-network (MLP / Tuned MLP)
diagnostics, presented together with the primary TabPFN and classical-baseline
results so the final talk can combine both into one coherent story.

Combined narrative
------------------
1. Structure-aware features help the neural network too  -> the main
   structure-feature finding generalises beyond TabPFN.
2. Even a *tuned* dense NN lands below TabPFN and the tree baselines on
   these small tabular materials datasets  -> "deep NN is not automatically
   better on small, high-dimensional tabular data" (the Lab3 theme).

All "our" numbers are read from CSVs (single source of truth):
    results/mlp_baseline_diagnostics/        -> MLP run outputs
    results/tuned_mlp_baseline_diagnostics/  -> Tuned MLP run outputs
    results/                                 -> TabPFN + classical baseline summaries

Outputs (PNG @ 200 dpi + PDF) go into the SAME deck folder so set 1 (fig1-10)
and set 2 (fig11-16) live together:
    presentation_graphs/

Run with the project conda environment:
    python scripts/make_results2_figures.py
"""

from __future__ import annotations

import glob
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

mpl.use("Agg")

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES1 = os.path.join(ROOT, "results")              # set 1: TabPFN + classical
RES2 = RES1                                       # set 2: MLP diagnostic run trees
OUT = os.path.join(ROOT, "presentation_graphs")   # one combined deck folder
os.makedirs(OUT, exist_ok=True)


def _latest(pattern):
    """Resolve the newest gpu_* run directory under a diagnostic folder."""
    hits = sorted(glob.glob(pattern))
    if not hits:
        raise FileNotFoundError(
            f"No diagnostic run matched {pattern!r}. Run the corresponding "
            "MLP diagnostic script first."
        )
    return hits[-1]


MLP_DIR = _latest(os.path.join(RES2, "mlp_baseline_diagnostics", "gpu_*"))
TUNED_DIR = _latest(os.path.join(RES2, "tuned_mlp_baseline_diagnostics", "gpu_*"))

# --------------------------------------------------------------------------
# Global style (identical to the set-1 deck for visual consistency)
# --------------------------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
    "font.size": 14,
    "axes.titlesize": 18,
    "axes.titleweight": "bold",
    "axes.labelsize": 15,
    "axes.labelweight": "semibold",
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "legend.fontsize": 13,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#dddddd",
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
})

# Consistent, colourblind-safe palette (Okabe-Ito based) — same as set 1,
# plus two dedicated neural-network colours.
C_TABPFN = "#0072B2"     # blue   = TabPFN
C_STRUCT = "#1b7837"     # green  = structure-feature story
C_STRUCT_DK = "#14532d"  # darker green for callouts
C_PROXY = "#9aa3ab"      # gray   = classical baselines (trees / HistGB)
C_WORSE = "#D55E00"      # vermillion = composition-only ("worse")
C_ENSEMBLE = "#7B4FA3"   # purple = TabPFN + ExtraTrees ensemble
C_MLP = "#E69F00"        # amber  = plain MLP (neural net)
C_TUNED = "#CC79A7"      # reddish-purple = Tuned MLP (neural net)
C_NEUTRAL = "#222222"    # neutral dark for value text

TASKS = ["matbench_jdft2d", "matbench_phonons"]
UNIT = {"matbench_jdft2d": "meV/atom", "matbench_phonons": "cm$^{-1}$"}
SHORT = {"matbench_jdft2d": "JDFT2D", "matbench_phonons": "Phonons"}
TITLE = {"matbench_jdft2d": "JDFT2D exfoliation energy",
         "matbench_phonons": "Phonon peak frequency"}
BEST_FS = "magpie_structure_all"   # both NN tasks are structure tasks


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".png"))
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    plt.close(fig)
    print("wrote", name)


# --------------------------------------------------------------------------
# Load data
# --------------------------------------------------------------------------
mlp = pd.read_csv(os.path.join(MLP_DIR, "tables", "mlp_summary.csv"))
tuned = pd.read_csv(os.path.join(TUNED_DIR, "tables", "tuned_mlp_summary.csv"))
mlp_fold = pd.read_csv(os.path.join(MLP_DIR, "metrics", "mlp_fold_metrics.csv"))
tuned_fold = pd.read_csv(os.path.join(TUNED_DIR, "metrics", "tuned_mlp_fold_metrics.csv"))
tuned_pred = pd.read_csv(os.path.join(TUNED_DIR, "predictions", "tuned_mlp_predictions.csv"))
summary1 = pd.read_csv(os.path.join(RES1, "metrics", "model_summary.csv"))


def nn_val(df, task, fs, col="mean_mae"):
    d = df[(df.task == task) & (df.feature_set == fs)]
    return float(d[col].iloc[0])


def s1_val(task, fs, model, col="mean_mae"):
    d = summary1[(summary1.task == task) & (summary1.feature_set == fs)
                 & (summary1.model == model)]
    return float(d[col].iloc[0]) if len(d) else np.nan


def s1_fold_mae(task, fs, model):
    """Per-fold MAE for a set-1 model from its individual fold-metrics file."""
    path = os.path.join(RES1, "metrics",
                        f"{task}_{fs}_{model}_fold_metrics.csv")
    d = pd.read_csv(path)
    return d.sort_values("fold").mae.values


# ==========================================================================
# FIGURE 11 — Structure features help the neural network too
# ==========================================================================
def fig_nn_structure_gain():
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8))
    models = [("MLP", mlp), ("Tuned MLP", tuned)]
    for ax, task in zip(axes, TASKS):
        comp = [nn_val(df, task, "magpie") for _, df in models]
        strc = [nn_val(df, task, BEST_FS) for _, df in models]
        x = np.arange(len(models))
        w = 0.36
        b1 = ax.bar(x - w / 2, comp, w, color=C_WORSE, zorder=3,
                    edgecolor="white", linewidth=1.3, label="composition only")
        b2 = ax.bar(x + w / 2, strc, w, color=C_STRUCT, zorder=3,
                    edgecolor="white", linewidth=1.3, label="+ structure")
        ymax = max(comp + strc)
        for bars, vals in ((b1, comp), (b2, strc)):
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + ymax * 0.015,
                        f"{v:.1f}", ha="center", va="bottom", fontsize=12.5,
                        fontweight="bold")
        # % change annotation above each model pair
        for xi, (c, s) in enumerate(zip(comp, strc)):
            pct = (s - c) / c * 100
            ax.annotate(f"{pct:+.1f}%", xy=(xi, max(c, s) + ymax * 0.10),
                        ha="center", va="bottom", fontsize=14,
                        fontweight="bold", color=C_STRUCT_DK)
        ax.set_xticks(x)
        ax.set_xticklabels([m for m, _ in models])
        ax.set_ylabel(f"5-fold mean MAE ({UNIT[task]})\n(lower is better)")
        ax.set_title(TITLE[task])
        ax.set_ylim(0, ymax * 1.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.80),
        ncol=2,
        frameon=True,
        framealpha=0.95,
    )
    fig.suptitle("Structure-aware features help the neural network too\n"
                 "$\\it{the\\ structure\\text{-}feature\\ gain\\ is\\ not\\ unique\\ to\\ TabPFN}$",
                 fontsize=18, fontweight="bold", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.70))
    save(fig, "fig11_nn_structure_gain")


# ==========================================================================
# FIGURE 12 — Combined leaderboard: where does the neural net land?
#   all model classes on each structure task's best feature set
# ==========================================================================
def fig_combined_leaderboard():
    drop = {"dummy_mean", "ridge_cv", "tabpfn_extra_trees_inner_tuned"}
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.4))
    for ax, task in zip(axes, TASKS):
        d = summary1[(summary1.task == task) & (summary1.feature_set == BEST_FS)
                     & (~summary1.model.isin(drop))][
            ["model", "model_display", "mean_mae", "sem_mae"]].copy()
        # append the two neural nets from result set 2
        rows = [d]
        for disp, df in (("MLP", mlp), ("Tuned MLP", tuned)):
            rows.append(pd.DataFrame([{
                "model": disp.lower().replace(" ", "_"),
                "model_display": disp,
                "mean_mae": nn_val(df, task, BEST_FS),
                "sem_mae": nn_val(df, task, BEST_FS, "sem_mae"),
            }]))
        d = pd.concat(rows, ignore_index=True).sort_values(
            "mean_mae", ascending=False)
        names = d.model_display.values
        vals = d.mean_mae.values
        sem = d.sem_mae.values

        def colour(n):
            if n == "TabPFN":
                return C_TABPFN
            if "TabPFN" in n:
                return C_ENSEMBLE
            if n == "MLP":
                return C_MLP
            if n == "Tuned MLP":
                return C_TUNED
            return C_PROXY
        colors = [colour(n) for n in names]
        ax.barh(range(len(names)), vals, color=colors, zorder=3,
                edgecolor="white", linewidth=1.0, xerr=sem,
                error_kw=dict(ecolor="#555", lw=1.2, capsize=3))
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=12.5)
        ax.set_xlabel(f"5-fold mean MAE ({UNIT[task]})")
        ax.set_title(f"{TITLE[task]}\n(Magpie + all structure)", fontsize=15)
        ax.set_xlim(0, vals.max() * 1.18)
        for i, v in enumerate(vals):
            ax.text(v, i, f"  {v:.3g}", va="center", ha="left",
                    fontsize=11.5, color=C_NEUTRAL, fontweight="bold")
    handles = [Patch(color=C_TABPFN, label="TabPFN"),
               Patch(color=C_ENSEMBLE, label="TabPFN + ExtraTrees"),
               Patch(color=C_PROXY, label="Classical (trees / HistGB)"),
               Patch(color=C_TUNED, label="Tuned MLP (neural net)"),
               Patch(color=C_MLP, label="MLP (neural net)")]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, 1.0), fontsize=12.5)
    fig.suptitle("Where does the neural network land? Below TabPFN and the tree baselines",
                 fontsize=18, fontweight="bold", y=1.10)
    fig.tight_layout()
    save(fig, "fig12_combined_leaderboard")


# ==========================================================================
# FIGURE 13 — Does tuning the MLP help? (honest, mixed result)
# ==========================================================================
def fig_nn_tuning_effect():
    combos = [("matbench_jdft2d", "magpie", "JDFT2D\ncomposition"),
              ("matbench_jdft2d", BEST_FS, "JDFT2D\n+ structure"),
              ("matbench_phonons", "magpie", "Phonons\ncomposition"),
              ("matbench_phonons", BEST_FS, "Phonons\n+ structure")]
    base = [nn_val(mlp, t, fs) for t, fs, _ in combos]
    tun = [nn_val(tuned, t, fs) for t, fs, _ in combos]
    pct = [(t - b) / b * 100 for b, t in zip(base, tun)]
    x = np.arange(len(combos))
    w = 0.38
    fig, ax = plt.subplots(figsize=(12, 6.2))
    b1 = ax.bar(x - w / 2, base, w, color=C_MLP, zorder=3,
                edgecolor="white", linewidth=1.3, label="MLP (default)")
    b2 = ax.bar(x + w / 2, tun, w, color=C_TUNED, zorder=3,
                edgecolor="white", linewidth=1.3, label="Tuned MLP")
    ymax = max(base + tun)
    for bars, vals in ((b1, base), (b2, tun)):
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + ymax * 0.012,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=11.5,
                    fontweight="bold")
    for xi, p in enumerate(pct):
        col = C_STRUCT_DK if p < 0 else C_WORSE
        word = "better" if p < 0 else "worse"
        ax.annotate(f"{p:+.1f}%\n({word})", xy=(xi, max(base[xi], tun[xi]) + ymax * 0.07),
                    ha="center", va="bottom", fontsize=11.5, fontweight="bold",
                    color=col)
    ax.set_xticks(x)
    ax.set_xticklabels([c for _, _, c in combos])
    ax.set_ylabel("5-fold mean MAE (per task unit)\n(lower is better)")
    ax.set_ylim(0, ymax * 1.30)
    ax.set_title("Does tuning the MLP help? Marginal and mixed")
    ax.legend(loc="upper left", frameon=True, framealpha=0.95)
    ax.text(0.5, -0.22, "inner-validation tuning of hidden layers / alpha / learning rate — "
            "units differ per task (meV/atom vs cm$^{-1}$)",
            transform=ax.transAxes, ha="center", va="top", fontsize=11,
            style="italic", color="#555")
    fig.tight_layout()
    save(fig, "fig13_nn_tuning_effect")


# ==========================================================================
# FIGURE 14 — Tuned-MLP parity plots (predicted vs actual), best branch
# ==========================================================================
def fig_nn_parity():
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.2))
    for ax, task in zip(axes, TASKS):
        p = tuned_pred[(tuned_pred.task == task)
                       & (tuned_pred.feature_set == BEST_FS)]
        yt, yp = p.y_true.values, p.y_pred.values
        lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
        pad = (hi - lo) * 0.05
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#888",
                ls="--", lw=1.5, zorder=2)
        ax.scatter(yt, yp, s=24, color=C_TUNED, alpha=0.45,
                   edgecolor="none", zorder=3)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal", adjustable="box")
        r2 = nn_val(tuned, task, BEST_FS, "mean_r2")
        mae = nn_val(tuned, task, BEST_FS, "mean_mae")
        ax.set_title(f"{TITLE[task]}\n(Magpie + all structure)", fontsize=15)
        ax.set_xlabel(f"actual ({UNIT[task]})")
        ax.set_ylabel(f"Tuned MLP predicted ({UNIT[task]})")
        ax.text(0.04, 0.96, f"$R^2$ = {r2:.2f}\nMAE = {mae:.3g}",
                transform=ax.transAxes, va="top", ha="left", fontsize=13,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          edgecolor="#ccc", alpha=0.9))
    fig.suptitle("Tuned-MLP predicted vs actual (held-out official folds, best feature set)",
                 fontsize=17, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "fig14_nn_parity")


# ==========================================================================
# FIGURE 15 — Fold-level MAE spread: NN vs TabPFN vs best tree
# ==========================================================================
def fig_fold_spread():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.0))
    for ax, task in zip(axes, TASKS):
        series = [
            ("TabPFN", s1_fold_mae(task, BEST_FS, "tabpfn"), C_TABPFN),
            ("Extra trees", s1_fold_mae(task, BEST_FS, "extra_trees"), C_PROXY),
            ("Tuned MLP", tuned_fold[(tuned_fold.task == task)
             & (tuned_fold.feature_set == BEST_FS)].sort_values("fold").mae.values,
             C_TUNED),
            ("MLP", mlp_fold[(mlp_fold.task == task)
             & (mlp_fold.feature_set == BEST_FS)].sort_values("fold").mae.values,
             C_MLP),
        ]
        # order best (lowest mean) at top
        series = sorted(series, key=lambda s: np.mean(s[1]), reverse=True)
        labels = [s[0] for s in series]
        data = [s[1] for s in series]
        colors = [s[2] for s in series]
        bp = ax.boxplot(data, vert=False, patch_artist=True, widths=0.55,
                        medianprops=dict(color="#111", lw=1.8), showmeans=True,
                        meanprops=dict(marker="D", markerfacecolor="white",
                                       markeredgecolor="#111", markersize=9))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.85)
        for i, x in enumerate(data):
            ax.scatter(x, np.full_like(x, i + 1), color="#111", s=28,
                       alpha=0.6, zorder=5)
        ax.set_yticks(range(1, len(labels) + 1))
        ax.set_yticklabels(labels)
        ax.set_xlabel(f"per-fold MAE ({UNIT[task]}) — lower is better")
        ax.set_title(f"{TITLE[task]}\n(Magpie + all structure)", fontsize=15)
    fig.suptitle("Fold-level MAE spread across the 5 official folds   "
                 "($\\it{\\diamond = mean,\\ line = median}$)",
                 fontsize=17, fontweight="bold", y=1.04)
    fig.tight_layout()
    save(fig, "fig15_fold_spread_nn")


# ==========================================================================
# FIGURE 16 — R^2 by model class (fit quality), best feature set
# ==========================================================================
def fig_r2_by_model_class():
    classes = [("TabPFN", C_TABPFN, lambda t: s1_val(t, BEST_FS, "tabpfn", "mean_r2")),
               ("Extra trees", C_PROXY, lambda t: s1_val(t, BEST_FS, "extra_trees", "mean_r2")),
               ("Tuned MLP", C_TUNED, lambda t: nn_val(tuned, t, BEST_FS, "mean_r2")),
               ("MLP", C_MLP, lambda t: nn_val(mlp, t, BEST_FS, "mean_r2"))]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    for ax, task in zip(axes, TASKS):
        names = [c[0] for c in classes]
        vals = [c[2](task) for c in classes]
        colors = [c[1] for c in classes]
        x = np.arange(len(names))
        ax.bar(x, vals, width=0.62, color=colors, zorder=3,
               edgecolor="white", linewidth=1.3)
        ax.axhline(1.0, color="#aaa", ls=":", lw=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=12)
        ax.set_ylim(min(0, min(vals) - 0.05), 1.08)
        ax.set_ylabel("$R^2$ (variance explained, higher is better)")
        ax.set_title(f"{TITLE[task]}\n(Magpie + all structure)", fontsize=15)
        for xi, v in zip(x, vals):
            ax.text(xi, v + (0.02 if v >= 0 else -0.02), f"{v:.2f}",
                    ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=12.5, fontweight="bold")
    fig.suptitle("Fit quality by model class — the neural nets trail, most on the harder task",
                 fontsize=17, fontweight="bold", y=1.03)
    fig.tight_layout()
    save(fig, "fig16_r2_by_model_class")


if __name__ == "__main__":
    fig_nn_structure_gain()
    fig_combined_leaderboard()
    fig_nn_tuning_effect()
    fig_nn_parity()
    fig_fold_spread()
    fig_r2_by_model_class()
    print("\nMLP diagnostic figures written to:", OUT)
