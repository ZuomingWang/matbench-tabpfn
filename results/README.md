# Results Folder

This folder contains the latest synced result summaries and figures. The
top-level `metrics/`, `tables/`, and `figures/` directories describe the
primary TabPFN/classical-model benchmark.

Most useful files:

- `tables/best_models_by_task.csv`: headline best model for each task.
- `metrics/model_summary.csv`: complete aggregated model summary.
- `metrics/fold_metrics.csv`: fold-level MAE/R2 metrics.
- `tables/structure_feature_branch_summary.csv`: structure-feature ablation summary.
- `tables/auto_summary.md`: generated text summary of the run.
- `figures/03_tabpfn_vs_best_baseline.png`: TabPFN relative to the best classical baseline.
- `figures/05_structure_feature_branch_comparison.png`: structure-feature branch ablation.
- `diagnostics/mlp/`: curated MLP and tuned-MLP summaries, fold metrics,
  configurations, and compact figures.

The committed primary and MLP summaries reproduce the numerical claims used
in presentation slides 6–12 and 17–19. The small-data and active-learning
presentation result tables are not currently committed. Their runners and
Colab notebooks are preserved, but exact verification of slides 14–16 and
20–24 requires either rerunning them or importing Kyle Xu's original
timestamped result folders.

Timestamped diagnostic runs are created locally in:

```text
results/small_data_diagnostics/
results/active_learning_screening/
results/mlp_baseline_diagnostics/
results/tuned_mlp_baseline_diagnostics/
```

These run trees can contain raw predictions, feature caches, logs, and other
generated artifacts, so they are ignored by Git. Only compact, presentation-
relevant summaries should be copied into `results/diagnostics/`.

Use `python scripts/verify_presentation_results.py` to check the committed
headline values and, when the missing run trees are available, the extension
results against the rounded values shown in the presentation.
