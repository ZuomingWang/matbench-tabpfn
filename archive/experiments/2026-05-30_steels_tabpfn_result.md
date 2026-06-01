# Matbench Steels TabPFN Result

Date: 2026-05-30

## Goal

Run `TabPFNRegressor` on the same `matbench_steels` folds and Magpie features used by the classical baseline models.

## Setup

- Task: `matbench_steels`
- Target: yield strength
- Unit: MPa
- Samples: 312
- Folds: official Matbench 5-fold split
- Features: 132 Magpie composition features
- Model: `TabPFNRegressor`
- Estimators: 8
- Device: auto

## Results

| Fold | Train Size | Test Size | MAE (MPa) | R2 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 249 | 63 | 108.745 | 0.586 |
| 1 | 249 | 63 | 70.066 | 0.892 |
| 2 | 250 | 62 | 72.542 | 0.866 |
| 3 | 250 | 62 | 85.370 | 0.866 |
| 4 | 250 | 62 | 101.664 | 0.511 |

Mean MAE: 87.677 MPa

Std MAE: 17.205 MPa

Mean R2: 0.744

## Interpretation

TabPFN gives the lowest MAE so far on `matbench_steels`.

- Random Forest MAE: 104.364 MPa
- Extra Trees MAE: 90.834 MPa
- TabPFN MAE: 87.677 MPa

TabPFN improves MAE by 15.99% relative to Random Forest and 3.47% relative to Extra Trees. Extra Trees still has higher mean R2, so the final report should discuss the metric tradeoff rather than claiming TabPFN dominates every metric.

## Outputs

- `results/metrics/matbench_steels_tabpfn_magpie_metrics.csv`
- `results/metrics/matbench_steels_tabpfn_magpie_summary.json`
- `results/predictions/matbench_steels_tabpfn_magpie_predictions.csv`
- `results/figures/notebook_steels_final_model_comparison.png`
