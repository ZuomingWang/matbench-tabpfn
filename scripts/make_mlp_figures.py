"""
Build figures for the neural-network (MLP / Tuned MLP) diagnostics alongside
the primary TabPFN and classical-model results.

Combined interpretation
-----------------------
1. Structure-aware features help the neural network too  -> the main
   structure-feature finding generalises beyond TabPFN.
2. Even a *tuned* dense NN lands below TabPFN and the tree baselines on
   these small tabular materials datasets: dense neural networks are not
   automatically better on small, high-dimensional tabular data.

All aggregate and fold-level numbers are read from Git-tracked CSVs:
    results/metrics/fold_metrics.csv
    results/diagnostics/mlp/

The script can still use the newest local timestamped MLP runs when the
compact diagnostics are unavailable.  Raw predictions are optional: a fresh
clone rebuilds every aggregate/fold-supported figure, reports the parity-only
figure as skipped, and exits successfully.

Outputs (PNG @ 200 dpi + PDF) go to:
    figures/

Run with the repository environment:
    python scripts/make_mlp_figures.py
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
COMPACT_MLP = os.path.join(RES1, "diagnostics", "mlp")
OUT = os.path.join(ROOT, "figures")
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


def _resolve_mlp_inputs():
    """Use committed compact evidence first, then optional local run trees."""
    compact = {
        "mlp_summary": os.path.join(COMPACT_MLP, "tables", "mlp_summary.csv"),
        "tuned_summary":
            os.path.join(COMPACT_MLP, "tables", "tuned_mlp_summary.csv"),
        "mlp_folds":
            os.path.join(COMPACT_MLP, "metrics", "mlp_fold_metrics.csv"),
        "tuned_folds":
            os.path.join(COMPACT_MLP, "metrics", "tuned_mlp_fold_metrics.csv"),
        "tuned_predictions":
            os.path.join(
                COMPACT_MLP, "predictions", "tuned_mlp_predictions.csv"
            ),
    }
    required = {key: path for key, path in compact.items()
                if key != "tuned_predictions"}
    if all(os.path.isfile(path) for path in required.values()):
        print("using committed compact MLP diagnostics:", COMPACT_MLP)
        if not os.path.isfile(compact["tuned_predictions"]):
            local_predictions = []
            for candidate in (
                os.path.join(ROOT, "results"),
                os.path.join(ROOT, "results 2"),
            ):
                local_predictions.extend(glob.glob(os.path.join(
                    candidate, "tuned_mlp_baseline_diagnostics", "gpu_*",
                    "predictions", "tuned_mlp_predictions.csv",
                )))
            if local_predictions:
                compact["tuned_predictions"] = sorted(local_predictions)[-1]
                print(
                    "using local tuned-MLP raw predictions:",
                    compact["tuned_predictions"],
                )
        return compact

    candidates = [
        os.path.join(ROOT, "results"),
        os.path.join(ROOT, "results 2"),
    ]
    for candidate in candidates:
        mlp_hits = glob.glob(
            os.path.join(candidate, "mlp_baseline_diagnostics", "gpu_*")
        )
        tuned_hits = glob.glob(
            os.path.join(candidate, "tuned_mlp_baseline_diagnostics", "gpu_*")
        )
        if mlp_hits and tuned_hits:
            mlp_dir = _latest(
                os.path.join(candidate, "mlp_baseline_diagnostics", "gpu_*")
            )
            tuned_dir = _latest(
                os.path.join(
                    candidate, "tuned_mlp_baseline_diagnostics", "gpu_*"
                )
            )
            print("using local timestamped MLP diagnostics:", candidate)
            return {
                "mlp_summary":
                    os.path.join(mlp_dir, "tables", "mlp_summary.csv"),
                "tuned_summary":
                    os.path.join(
                        tuned_dir, "tables", "tuned_mlp_summary.csv"
                    ),
                "mlp_folds":
                    os.path.join(
                        mlp_dir, "metrics", "mlp_fold_metrics.csv"
                    ),
                "tuned_folds":
                    os.path.join(
                        tuned_dir, "metrics", "tuned_mlp_fold_metrics.csv"
                    ),
                "tuned_predictions":
                    os.path.join(
                        tuned_dir, "predictions",
                        "tuned_mlp_predictions.csv"
                    ),
            }
    missing = [
        os.path.relpath(path, ROOT)
        for path in required.values() if not os.path.isfile(path)
    ]
    raise FileNotFoundError(
        "The committed compact MLP evidence is incomplete and no paired "
        "local run folders were found. Missing: " + ", ".join(missing)
    )


MLP_INPUTS = _resolve_mlp_inputs()

# --------------------------------------------------------------------------
# Global style shared across benchmark figures
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

# Consistent, colourblind-safe palette (Okabe-Ito based), with two dedicated
# neural-network colours.
C_TABPFN = "#0072B2"     # blue   = TabPFN
C_STRUCT = "#1b7837"     # green  = structure-aware features
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
mlp = pd.read_csv(MLP_INPUTS["mlp_summary"])
tuned = pd.read_csv(MLP_INPUTS["tuned_summary"])
mlp_fold = pd.read_csv(MLP_INPUTS["mlp_folds"])
tuned_fold = pd.read_csv(MLP_INPUTS["tuned_folds"])
tuned_pred = (
    pd.read_csv(MLP_INPUTS["tuned_predictions"])
    if os.path.isfile(MLP_INPUTS["tuned_predictions"])
    else None
)
folds1 = pd.read_csv(os.path.join(RES1, "metrics", "fold_metrics.csv"))
if "status" in folds1:
    folds1 = folds1[folds1.status == "ok"]
folds1 = folds1.drop_duplicates(
    subset=["task", "feature_set", "model", "fold"], keep="last"
)
summary1 = folds1.groupby(
    ["task", "feature_set", "feature_set_display", "model", "model_display"],
    as_index=False,
).agg(
    mean_mae=("mae", "mean"),
    sem_mae=("mae", "sem"),
    mean_r2=("r2", "mean"),
)


def nn_val(df, task, fs, col="mean_mae"):
    d = df[(df.task == task) & (df.feature_set == fs)]
    return float(d[col].iloc[0])


def s1_val(task, fs, model, col="mean_mae"):
    d = summary1[(summary1.task == task) & (summary1.feature_set == fs)
                 & (summary1.model == model)]
    return float(d[col].iloc[0]) if len(d) else np.nan


def s1_fold_mae(task, fs, model):
    """Per-fold MAE for a primary model from the committed fold table."""
    d = folds1[
        (folds1.task == task)
        & (folds1.feature_set == fs)
        & (folds1.model == model)
    ]
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
    axes[0].legend(loc="upper right", frameon=True, framealpha=0.95)
    fig.suptitle("Effect of structure-aware features on neural-network performance\n"
                  "$\\it{the\\ improvement\\ is\\ not\\ unique\\ to\\ TabPFN}$",
                 fontsize=18, fontweight="bold", y=1.07)
    fig.tight_layout()
    save(fig, "fig11_nn_structure_gain")


# ==========================================================================
# FIGURE 12 — Model comparison on each structure task's best feature set
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
        ax.set_xlabel(f"5-fold mean MAE ({UNIT[task]}) — lower is better")
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
    fig.suptitle("Model performance with structure-aware features",
                 fontsize=18, fontweight="bold", y=1.10)
    fig.tight_layout()
    save(fig, "fig12_combined_leaderboard")


# ==========================================================================
# FIGURE 13 — Effect of MLP hyperparameter tuning
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
    ax.set_title("Effect of MLP hyperparameter tuning")
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
    if tuned_pred is None:
        print(
            "SKIP fig14_nn_parity: tuned-MLP raw predictions are not "
            "committed in a fresh clone (aggregate/fold figures remain "
            "reproducible)."
        )
        print(
            "  missing:",
            os.path.relpath(MLP_INPUTS["tuned_predictions"], ROOT),
        )
        return False
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
    return True


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
    fig.suptitle("Fit quality by model class",
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
    print("\nAvailable MLP diagnostic figures written to:", OUT)
