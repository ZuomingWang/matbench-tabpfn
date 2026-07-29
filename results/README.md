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
- `diagnostics/small_data/`: sanitized numeric claims extracted from the
  executed learning-curve notebook.
- `diagnostics/active_learning/`: 40 standard and 16 stress-test aggregate
  rows extracted from the executed active-learning notebook HTML tables.

The committed primary and MLP summaries reproduce the numerical claims used
in presentation slides 6–12 and 17–19. Committed notebook-display evidence
supports the headline values in slides 14–16 and 20–24 at the precision shown
in the executed notebooks. The original timestamped small-data and
active-learning result trees are still unavailable, so raw predictions,
complete traces, and undisplayed full-precision table fields cannot be
reconstructed.

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

Use `python scripts/extract_colab_notebook_evidence.py --check` to prove that
the compact evidence still matches the unchanged notebook outputs. Use
`python scripts/verify_presentation_results.py` to compare all committed
headline values with the presentation. When the relevant run directories are
available, `--require-extensions` performs a stricter audit of the two
small-data source summaries and active-learning aggregate summaries. It does
not validate every raw prediction or acquisition-trace file in a run tree.
