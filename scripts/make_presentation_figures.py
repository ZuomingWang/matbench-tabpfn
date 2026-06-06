"""
Build a cohesive, slide-ready set of presentation figures for the
Matbench TabPFN structure-feature study.

All "our" numbers are read from results/ CSVs (single source of truth).
Published reference numbers are taken from the presentation outline
(paper Table 1/2/4 and Supplementary Table S13) and are clearly labeled.

Colour system (colourblind-safe, consistent across every figure):
    GREEN  = the structure-feature story (the project's hero)
    ORANGE = composition-only / the weaker number being beaten ("worse")
    BLUE   = TabPFN (our model) / "better than paper"
    GRAY   = other classical baselines / neutral

Outputs (PNG @ 200 dpi + PDF) go to:  presentation_graphs/

Run with the project conda env:
    /Users/zuoming/miniforge3/envs/matbench-tabpfn/bin/python scripts/make_presentation_figures.py
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
METRICS = os.path.join(ROOT, "results", "metrics")
TABLES = os.path.join(ROOT, "results", "tables")
PREDS = os.path.join(ROOT, "results", "predictions")
OUT = os.path.join(ROOT, "presentation_graphs")
os.makedirs(OUT, exist_ok=True)

# --------------------------------------------------------------------------
# Global style (large, clean, slide-friendly)
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

# Consistent, colourblind-safe palette (Okabe-Ito based)
C_TABPFN = "#0072B2"     # blue  = TabPFN / "better than paper"
C_STRUCT = "#1b7837"     # green = structure-feature story
C_STRUCT_DK = "#14532d"  # darker green for callout text/arrows (contrast)
C_PROXY = "#9aa3ab"      # gray  = neutral classical baselines
C_WORSE = "#D55E00"      # orange/vermillion = composition-only / "worse"
C_ENSEMBLE = "#7B4FA3"   # purple = TabPFN+ExtraTrees ensemble (distinct)
C_NEUTRAL = "#222222"    # neutral dark for value text

# task -> feature set used as that task's "best" branch (TabPFN-optimal)
TASK_FS = {
    "matbench_steels": "magpie",
    "matbench_expt_gap": "magpie",
    "matbench_jdft2d": "magpie_structure_all",
    "matbench_phonons": "magpie_structure_all",
}
UNIT = {"matbench_steels": "MPa", "matbench_expt_gap": "eV",
        "matbench_jdft2d": "meV/atom", "matbench_phonons": "cm$^{-1}$"}
SHORT = {"matbench_steels": "Steels", "matbench_expt_gap": "Exp. gap",
         "matbench_jdft2d": "JDFT2D", "matbench_phonons": "Phonons"}


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".png"))
    fig.savefig(os.path.join(OUT, name + ".pdf"))
    plt.close(fig)
    print("wrote", name)


def hbar_labels(ax, bars, fmt="{:.1f}", pad=4, fontsize=13, color=C_NEUTRAL):
    for b in bars:
        w = b.get_width()
        ax.annotate(fmt.format(w), xy=(w, b.get_y() + b.get_height() / 2),
                    xytext=(pad, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=fontsize, color=color,
                    fontweight="bold")


# --------------------------------------------------------------------------
# Load data
# --------------------------------------------------------------------------
summary = pd.read_csv(os.path.join(METRICS, "model_summary.csv"))
branch = pd.read_csv(os.path.join(TABLES, "structure_feature_branch_summary.csv"))
best = pd.read_csv(os.path.join(TABLES, "best_models_by_task.csv"))
weight = pd.read_csv(os.path.join(TABLES, "tabpfn_extra_trees_weight_scan.csv"))
paired = pd.read_csv(os.path.join(TABLES, "paired_fold_comparisons.csv"))

fold_frames = []
for f in glob.glob(os.path.join(METRICS, "*_fold_metrics.csv")):
    try:
        df = pd.read_csv(f)
        if {"task", "model", "feature_set", "fold", "mae"}.issubset(df.columns):
            fold_frames.append(df[["task", "model", "model_display",
                                   "feature_set", "feature_set_display",
                                   "fold", "mae", "r2"]])
    except Exception as exc:  # pragma: no cover
        print("skip", f, exc)
folds = pd.concat(fold_frames, ignore_index=True).drop_duplicates()


def our_mae(task, fs):
    d = summary[(summary.task == task) & (summary.feature_set == fs)]
    return d.mean_mae.min()


# ==========================================================================
# FIGURE 1 — Headline: structure-aware features cut error
# ==========================================================================
def fig_structure_gain():
    tasks = [("matbench_jdft2d", "JDFT2D exfoliation energy", "MAE (meV/atom)"),
             ("matbench_phonons", "Phonon peak frequency", "MAE (cm$^{-1}$)")]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.6))
    for ax, (task, title, ylab) in zip(axes, tasks):
        row = branch[branch.task == task].iloc[0]
        proxy, struct = row.proxy_mean_mae, row.structure_mean_mae
        pct = row.mae_pct_change_structure_minus_proxy
        bars = ax.bar([0, 1], [proxy, struct], color=[C_WORSE, C_STRUCT],
                      width=0.62, zorder=3, edgecolor="white", linewidth=1.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Composition\nonly (Magpie)",
                            "Magpie + structure\ndescriptors"])
        ax.set_ylabel(ylab + "  (lower is better)")
        ax.set_title(title)
        ax.set_ylim(0, proxy * 1.22)
        for b, v in zip(bars, [proxy, struct]):
            ax.text(b.get_x() + b.get_width() / 2, v + proxy * 0.015, f"{v:.1f}",
                    ha="center", va="bottom", fontsize=14, fontweight="bold")
        ax.annotate("", xy=(1, struct + proxy * 0.08), xytext=(1, proxy),
                    arrowprops=dict(arrowstyle="-|>", color=C_STRUCT_DK, lw=2.4))
        ax.text(1.33, (proxy + struct) / 2, f"{pct:+.1f}%", color=C_STRUCT_DK,
                fontsize=16, fontweight="bold", va="center", ha="left")
        ax.set_xlim(-0.6, 1.95)
    fig.suptitle("Structure-aware features substantially improve TabPFN\n"
                 "$\\it{TabPFN \\cdot 5\\text{-}fold\\ mean\\ MAE}$",
                 fontsize=18, fontweight="bold", y=1.07)
    fig.tight_layout()
    save(fig, "fig1_structure_gain")


# ==========================================================================
# FIGURE 2 — Our best vs published paper numbers (colourblind-safe)
# ==========================================================================
def fig_published_comparison():
    rows = [
        ("Steels\n(composition)", our_mae("matbench_steels", "magpie"), 83.605,
         "paper ICL-FM", "MPa"),
        ("Exp. gap\n(composition)", our_mae("matbench_expt_gap", "magpie"), 0.3142,
         "paper ICL-FM", "eV"),
        ("JDFT2D\n(composition)", our_mae("matbench_jdft2d", "magpie"), 44.736,
         "paper FM-Magpie", "meV/atom"),
        ("Phonons\n(composition)", our_mae("matbench_phonons", "magpie"), 41.948,
         "paper FM-MagpieEX", "cm$^{-1}$"),
        ("JDFT2D\n(+ structure)", our_mae("matbench_jdft2d", "magpie_structure_all"),
         40.880, "paper FMEF", "meV/atom"),
        ("Phonons\n(+ structure)", our_mae("matbench_phonons", "magpie_structure_all"),
         29.753, "paper FMEF", "cm$^{-1}$"),
    ]
    labels = [f"{r[0]}\n{r[1]:.3g} vs {r[2]:.3g} {r[4]}" for r in rows]
    pct = [(r[1] - r[2]) / r[2] * 100 for r in rows]
    colors = [C_WORSE if p > 0 else C_TABPFN for p in pct]

    fig, ax = plt.subplots(figsize=(12, 6.6))
    y = np.arange(len(rows))[::-1]
    ax.barh(y, pct, color=colors, height=0.6, zorder=3,
            edgecolor="white", linewidth=1.2)
    ax.axvline(0, color="#333", lw=1.4, zorder=4)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=12.5)
    ax.set_xlabel("Our best MAE relative to published paper number  (%)")
    ax.set_title("How our results compare to the reference paper", pad=18)
    lim = max(abs(min(pct)), abs(max(pct))) * 1.28
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    for yi, p in zip(y, pct):
        off, ha = (0.6, "left") if p >= 0 else (-0.6, "right")
        ax.text(p + off, yi, f"{p:+.1f}%", va="center", ha=ha, fontsize=14,
                fontweight="bold", color=colors[list(y).index(yi)])
    # redundant, non-colour cue: direction labels on the axis
    ax.text(-lim * 0.99, -0.62, "◀ better than paper", color=C_TABPFN,
            fontsize=12.5, fontweight="bold", ha="left", va="center")
    ax.text(lim * 0.99, -0.62, "worse than paper ▶", color=C_WORSE,
            fontsize=12.5, fontweight="bold", ha="right", va="center")
    leg = ax.legend(handles=[Patch(color=C_TABPFN, label="better than paper"),
                             Patch(color=C_WORSE, label="worse than paper")],
                    loc="upper left", frameon=True, framealpha=0.95)
    leg.get_frame().set_edgecolor("#cccccc")
    fig.tight_layout()
    save(fig, "fig2_published_comparison")


# ==========================================================================
# FIGURE 3 — Per-task model leaderboard (fair: same feature set)
#   drops Dummy mean + RidgeCV (off-scale, not informative) for legibility
# ==========================================================================
def fig_model_leaderboard():
    titles = {"matbench_steels": "Steels  (composition)",
              "matbench_expt_gap": "Exp. band gap  (composition)",
              "matbench_jdft2d": "JDFT2D  (+ structure)",
              "matbench_phonons": "Phonons  (+ structure)"}
    drop = {"dummy_mean", "ridge_cv"}
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))
    for ax, task in zip(axes.ravel(), TASK_FS):
        fs = TASK_FS[task]
        d = summary[(summary.task == task) & (summary.feature_set == fs)
                    & (~summary.model.isin(drop))].copy()
        d = d.sort_values("mean_mae", ascending=False)
        names, vals, sem = d.model_display.values, d.mean_mae.values, d.sem_mae.values
        colors = [C_TABPFN if (n == "TabPFN")
                  else (C_ENSEMBLE if "TabPFN" in n else C_PROXY) for n in names]
        ax.barh(range(len(names)), vals, color=colors, zorder=3,
                edgecolor="white", linewidth=1.0,
                xerr=sem, error_kw=dict(ecolor="#555", lw=1.2, capsize=3))
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=12.5)
        ax.set_xlabel(f"5-fold mean MAE ({UNIT[task]}) — lower is better")
        ax.set_title(titles[task])
        ax.set_xlim(0, vals.max() * 1.18)
        for i, v in enumerate(vals):
            ax.text(v, i, f"  {v:.3g}", va="center", ha="left",
                    fontsize=11.5, color=C_NEUTRAL, fontweight="bold")
    handles = [Patch(color=C_TABPFN, label="TabPFN"),
               Patch(color=C_ENSEMBLE, label="TabPFN + ExtraTrees"),
               Patch(color=C_PROXY, label="Classical baselines")]
    fig.legend(handles=handles, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=14)
    fig.suptitle("Model comparison on each task's best feature set\n"
                 "$\\it{Dummy\\ and\\ RidgeCV\\ baselines\\ omitted\\ (off\\text{-}scale)}$",
                 fontsize=18, fontweight="bold", y=1.07)
    fig.tight_layout()
    save(fig, "fig3_model_leaderboard")


# ==========================================================================
# FIGURE 4 — Structure descriptor ablation (short labels)
# ==========================================================================
def fig_structure_ablation():
    tasks = [("matbench_jdft2d", "JDFT2D exfoliation", "MAE (meV/atom)"),
             ("matbench_phonons", "Phonon peak", "MAE (cm$^{-1}$)")]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))
    for ax, (task, title, xlab) in zip(axes, tasks):
        d = branch[branch.task == task].copy().sort_values("structure_mean_mae")
        # drop the shared "Magpie + " prefix for legibility
        names = [n.replace("Magpie + ", "") for n in d.structure_feature_set_display]
        vals = list(d.structure_mean_mae.values)
        proxy = d.proxy_mean_mae.iloc[0]
        names.append("composition only (proxy)")
        vals.append(proxy)
        colors = []
        for n in names:
            if "proxy" in n:
                colors.append(C_WORSE)
            elif "all structure" in n:
                colors.append(C_STRUCT)
            elif ("density" in n) or ("packing" in n):
                colors.append(C_TABPFN)
            else:
                colors.append(C_PROXY)
        bars = ax.barh(range(len(names)), vals, color=colors, zorder=3,
                       edgecolor="white", linewidth=1.0)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=13)
        ax.invert_yaxis()
        hbar_labels(ax, bars, fmt="{:.1f}", fontsize=12)
        ax.set_xlabel(xlab + " — lower is better")
        ax.set_title(title, pad=10)
        ax.set_xlim(0, max(vals) * 1.16)
    handles = [Patch(color=C_STRUCT, label="all structure descriptors"),
               Patch(color=C_TABPFN, label="density / packing branches"),
               Patch(color=C_PROXY, label="symmetry / complexity only"),
               Patch(color=C_WORSE, label="composition-only proxy")]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 1.0), fontsize=13)
    fig.suptitle("Which structure descriptors help?  (best model per branch; "
                 "all branches use the Magpie base)",
                 fontsize=17, fontweight="bold", y=1.07)
    fig.tight_layout()
    save(fig, "fig4_structure_ablation")


# ==========================================================================
# FIGURE 5 — TabPFN vs best classical baseline (compact)
# ==========================================================================
def fig_tabpfn_vs_baseline():
    order = list(TASK_FS)
    fig, axes = plt.subplots(1, 4, figsize=(14, 5.0))
    for ax, task in zip(axes, order):
        d = summary[summary.task == task]
        tp = d[d.model == "tabpfn"].sort_values("mean_mae").iloc[0]
        base = d[~d.model.isin(["tabpfn", "tabpfn_extra_trees_ensemble",
                                "tabpfn_extra_trees_inner_tuned", "dummy_mean",
                                "ridge_cv"])].sort_values("mean_mae").iloc[0]
        vals = [tp.mean_mae, base.mean_mae]
        bars = ax.bar([0, 1], vals, color=[C_TABPFN, C_PROXY], width=0.6,
                      zorder=3, edgecolor="white", linewidth=1.3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["TabPFN", base.model_display], fontsize=12)
        ax.set_title(f"{SHORT[task]}\n({UNIT[task]})", fontsize=14)
        ax.set_ylim(0, max(vals) * 1.26)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.02,
                    f"{v:.3g}", ha="center", va="bottom", fontsize=12,
                    fontweight="bold")
        pct = (tp.mean_mae - base.mean_mae) / base.mean_mae * 100
        ax.text(0.5, max(vals) * 1.15, f"▼ {pct:+.1f}%", ha="center",
                color=C_NEUTRAL, fontsize=13, fontweight="bold")
    axes[0].set_ylabel("5-fold mean MAE\n(lower is better)")
    fig.suptitle("TabPFN vs the strongest classical baseline  "
                 "($\\it{vs\\ classical\\ ML,\\ not\\ vs\\ the\\ published\\ paper}$)",
                 fontsize=16, fontweight="bold", y=1.05)
    fig.tight_layout()
    save(fig, "fig5_tabpfn_vs_baseline")


# ==========================================================================
# FIGURE 6 — Fold-level MAE spread (stability)
# ==========================================================================
def fig_fold_distribution():
    keep = ["tabpfn", "extra_trees", "random_forest", "hist_gradient_boosting"]
    label = {"tabpfn": "TabPFN", "extra_trees": "Extra trees",
             "random_forest": "Random forest", "hist_gradient_boosting": "HistGB"}
    titles = {"matbench_steels": "Steels (composition)",
              "matbench_expt_gap": "Exp. band gap (composition)",
              "matbench_jdft2d": "JDFT2D (+ structure)",
              "matbench_phonons": "Phonons (+ structure)"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, task in zip(axes.ravel(), TASK_FS):
        fs = TASK_FS[task]
        d = folds[(folds.task == task) & (folds.feature_set == fs)
                  & (folds.model.isin(keep))]
        models = [m for m in keep if m in d.model.unique()]
        data = [d[d.model == m].mae.values for m in models]
        order = np.argsort([np.mean(x) for x in data])
        models = [models[i] for i in order]
        data = [data[i] for i in order]
        colors = [C_TABPFN if m == "tabpfn" else C_PROXY for m in models]
        bp = ax.boxplot(data, vert=False, patch_artist=True, widths=0.55,
                        medianprops=dict(color="#111", lw=1.8), showmeans=True,
                        meanprops=dict(marker="D", markerfacecolor="white",
                                       markeredgecolor="#111", markersize=9))
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.85)
        for i, x in enumerate(data):
            ax.scatter(x, np.full_like(x, i + 1), color="#111", s=26,
                       alpha=0.6, zorder=5)
        ax.set_yticks(range(1, len(models) + 1))
        ax.set_yticklabels([label[m] for m in models])
        ax.set_xlabel(f"per-fold MAE ({UNIT[task]}) — lower is better")
        ax.set_title(titles[task])
    fig.suptitle("Fold-level MAE spread across the 5 official Matbench folds   "
                 "($\\it{\\diamond = mean,\\ line = median}$)",
                 fontsize=17, fontweight="bold", y=1.01)
    fig.tight_layout()
    save(fig, "fig6_fold_distribution")


# ==========================================================================
# FIGURE 7 — R^2 per task (units-free fit quality)
# ==========================================================================
def fig_r2_by_task():
    order = ["matbench_phonons", "matbench_steels",
             "matbench_expt_gap", "matbench_jdft2d"]
    d = best.set_index("task").loc[order]
    r2 = d.mean_r2.values
    err = d.std_r2.values
    mae = d.mean_mae.values
    names = [f"{SHORT[t]}\n({UNIT[t]})" for t in order]
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(order))
    bars = ax.bar(x, r2, width=0.6, color=C_TABPFN, zorder=3,
                  edgecolor="white", linewidth=1.3,
                  yerr=err, error_kw=dict(ecolor="#444", lw=1.4, capsize=5))
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("$R^2$  (variance explained, higher is better)")
    ax.set_ylim(0, 1.08)
    ax.axhline(1.0, color="#aaa", ls=":", lw=1.2)
    ax.set_title("How well does best-model TabPFN actually fit each task?", pad=30)
    ax.text(0.5, 1.04, "structure features help most where the task is hardest (JDFT2D)",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=11.5, style="italic", color="#555")
    for xi, (v, m) in zip(x, zip(r2, mae)):
        ax.text(xi, v + 0.03, f"$R^2$={v:.2f}", ha="center", va="bottom",
                fontsize=13, fontweight="bold")
        ax.text(xi, 0.03, f"MAE\n{m:.3g}", ha="center", va="bottom",
                fontsize=11, color="white", fontweight="bold")
    fig.tight_layout()
    save(fig, "fig7_r2_by_task")


# ==========================================================================
# FIGURE 8 — Ensemble weight sweep (honest negative result)
# ==========================================================================
def fig_weight_sweep():
    line_c = {"matbench_steels": "#CC79A7", "matbench_expt_gap": "#E69F00",
              "matbench_jdft2d": "#0072B2", "matbench_phonons": "#009E73"}
    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    for task, fs in TASK_FS.items():
        w = weight[(weight.task == task) & (weight.feature_set == fs)].sort_values(
            "tabpfn_weight")
        if w.empty:
            continue
        pure = w[w.tabpfn_weight == 1.0].mean_mae.iloc[0]
        rel = (w.mean_mae.values / pure - 1.0) * 100.0
        ax.plot(w.tabpfn_weight.values, rel, "-o", color=line_c[task], lw=2.2,
                ms=5, label=f"{SHORT[task]}", zorder=3)
        # mark the optimum
        j = int(np.argmin(w.mean_mae.values))
        ax.scatter([w.tabpfn_weight.values[j]], [rel[j]], s=160,
                   facecolor="white", edgecolor=line_c[task], linewidth=2.4,
                   zorder=5)
    ax.axhline(0, color="#333", lw=1.3)
    ax.axvline(1.0, color="#bbb", ls="--", lw=1.4)
    ax.text(1.0, ax.get_ylim()[1] * 0.92, " pure TabPFN", ha="right",
            va="top", color="#666", fontsize=12, fontweight="bold")
    ax.set_xlabel("TabPFN weight in TabPFN + ExtraTrees ensemble  "
                  "(0 = pure ExtraTrees, 1 = pure TabPFN)")
    ax.set_ylabel("MAE change vs pure TabPFN  (%)\n(below 0 = ensemble helps)")
    ax.set_title("Does ensembling help? Mostly no — the optimum sits at pure TabPFN")
    ax.legend(title="task (best feature set)", frameon=True, loc="upper center",
              ncol=4, bbox_to_anchor=(0.5, -0.16))
    ax.set_xlim(-0.03, 1.03)
    fig.tight_layout()
    save(fig, "fig8_ensemble_weight_sweep")


# ==========================================================================
# FIGURE 9 — Paired per-fold wins (robustness)
# ==========================================================================
def fig_paired_wins():
    order = ["matbench_phonons", "matbench_jdft2d",
             "matbench_expt_gap", "matbench_steels"]
    wins, n = {}, {}
    for task in order:
        fs = TASK_FS[task]
        r = paired[(paired.task == task) & (paired.reference_feature_set == fs)]
        r = r.iloc[0]
        wins[task] = int(r.tabpfn_win_folds)
        n[task] = int(r.n_paired_folds)
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    y = np.arange(len(order))
    for yi, task in zip(y, order):
        w, tot = wins[task], n[task]
        ax.barh(yi, w, color=C_TABPFN, zorder=3, edgecolor="white", height=0.62)
        ax.barh(yi, tot - w, left=w, color=C_PROXY, zorder=3,
                edgecolor="white", height=0.62)
        ax.text(tot + 0.08, yi, f"TabPFN wins {w} / {tot} folds", va="center",
                ha="left", fontsize=13, fontweight="bold", color=C_NEUTRAL)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{SHORT[t]}\n({TASK_FS[t].replace('magpie_structure_all','+structure').replace('magpie','composition')})"
                        for t in order])
    ax.set_xlabel("number of official folds  (out of 5)")
    ax.set_xlim(0, max(n.values()) * 1.5)
    ax.set_xticks(range(0, 6))
    ax.set_title("Head-to-head: TabPFN vs best classical baseline, per fold")
    leg = ax.legend(handles=[Patch(color=C_TABPFN, label="TabPFN lower error"),
                             Patch(color=C_PROXY, label="baseline lower error")],
                    loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2,
                    frameon=True)
    leg.get_frame().set_edgecolor("#cccccc")
    fig.tight_layout()
    save(fig, "fig9_paired_fold_wins")


# ==========================================================================
# FIGURE 10 — Parity plots (predicted vs actual), TabPFN best branch
# ==========================================================================
def fig_parity():
    titles = {"matbench_steels": "Steels  (composition)",
              "matbench_expt_gap": "Exp. band gap  (composition)",
              "matbench_jdft2d": "JDFT2D  (+ structure)",
              "matbench_phonons": "Phonons  (+ structure)"}
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 10.5))
    for ax, task in zip(axes.ravel(), TASK_FS):
        fs = TASK_FS[task]
        path = os.path.join(PREDS, f"{task}_{fs}_tabpfn_predictions.csv")
        p = pd.read_csv(path)
        yt, yp = p.y_true.values, p.y_pred.values
        lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
        pad = (hi - lo) * 0.05
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#888",
                ls="--", lw=1.5, zorder=2, label="perfect")
        ax.scatter(yt, yp, s=22, color=C_TABPFN, alpha=0.45,
                   edgecolor="none", zorder=3)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal", adjustable="box")
        # metrics from summary (single source of truth)
        row = summary[(summary.task == task) & (summary.feature_set == fs)
                      & (summary.model == "tabpfn")].iloc[0]
        ax.set_title(titles[task])
        ax.set_xlabel(f"actual ({UNIT[task]})")
        ax.set_ylabel(f"TabPFN predicted ({UNIT[task]})")
        ax.text(0.04, 0.96, f"$R^2$ = {row.mean_r2:.2f}\nMAE = {row.mean_mae:.3g}",
                transform=ax.transAxes, va="top", ha="left", fontsize=13,
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          edgecolor="#ccc", alpha=0.9))
    fig.suptitle("TabPFN predicted vs actual (held-out official folds, best feature set)",
                 fontsize=17, fontweight="bold", y=1.0)
    fig.tight_layout()
    save(fig, "fig10_tabpfn_parity")


if __name__ == "__main__":
    fig_structure_gain()
    fig_published_comparison()
    fig_model_leaderboard()
    fig_structure_ablation()
    fig_tabpfn_vs_baseline()
    fig_fold_distribution()
    fig_r2_by_task()
    fig_weight_sweep()
    fig_paired_wins()
    fig_parity()
    print("\nAll figures written to:", OUT)
