# Results

## Primary benchmark

- `metrics/fold_metrics.csv`: fold-level MAE and R²
- `metrics/model_summary.csv`: aggregate model results
- `tables/best_models_by_task.csv`: best model per task
- `tables/structure_feature_branch_summary.csv`: structure-feature ablation
- `figures/`: benchmark plots

## Diagnostics

- `diagnostics/mlp/`: plain and tuned MLP metrics, configurations, and figures
- `diagnostics/small_data/`: learning-curve values extracted from notebook
  output
- `diagnostics/active_learning/`: acquisition values extracted from notebook
  output

Raw feature caches, predictions, logs, and timestamped run directories are
ignored. Only compact result files required to support reported values are
tracked.

Verify the committed evidence without retraining:

```bash
python scripts/extract_colab_notebook_evidence.py --check
python scripts/verify_results.py
```

The learning-curve and active-learning tables preserve values displayed in
the executed notebooks. Their missing raw run directories cannot be
reconstructed from the compact tables.
