# Results Folder

This folder contains the latest synced result summaries and figures.

Most useful files:

- `tables/best_models_by_task.csv`: headline best model for each task.
- `metrics/model_summary.csv`: complete aggregated model summary.
- `metrics/fold_metrics.csv`: fold-level MAE/R2 metrics.
- `tables/structure_feature_branch_summary.csv`: structure-feature ablation summary.
- `tables/auto_summary.md`: generated text summary of the run.
- `figures/03_tabpfn_vs_best_baseline.png`: TabPFN relative to the best classical baseline.
- `figures/05_structure_feature_branch_comparison.png`: structure-feature branch ablation.

Large generated files such as raw predictions, feature caches, logs, and rerun directories are ignored by Git by default.
