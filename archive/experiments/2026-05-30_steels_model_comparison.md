# Matbench Steels Model Comparison

Date: 2026-05-30

## Goal

Extend the first notebook beyond a single Random Forest baseline by comparing several conventional tabular models and an ensemble on the same official Matbench folds.

## Notebook

`notebooks/01_matbench_steels_magpie_rf_tabpfn.ipynb`

## Setup

- Task: `matbench_steels`
- Target: yield strength
- Unit: MPa
- Samples: 312
- Folds: official Matbench 5-fold split
- Features: 132 Magpie composition features

## Results

| Model | Mean MAE (MPa) | Std MAE | Mean R2 | Features |
| --- | ---: | ---: | ---: | ---: |
| TabPFN | 87.677 | 17.205 | 0.744 | 132 |
| Extra Trees | 90.834 | 10.036 | 0.773 | 132 |
| Voting ensemble | 98.300 | 11.734 | 0.754 | 132 |
| Random Forest top-20 features | 103.685 | 13.006 | 0.743 | 20 |
| Random Forest | 104.364 | 12.127 | 0.737 | 132 |
| HistGradientBoosting | 107.712 | 14.622 | 0.697 | 132 |
| RidgeCV | 153.577 | 6.920 | 0.476 | 132 |

## Interpretation

TabPFN is the best MAE model on this task after license/token setup, improving the Random Forest baseline from 104.364 MPa to 87.677 MPa mean MAE. Extra Trees remains the strongest classical baseline and has the best mean R2, so both metrics should be reported. The top-20 feature-selection extension reduces dimensionality substantially but does not beat TabPFN or Extra Trees on MAE. The simple voting ensemble improves over Random Forest but does not beat Extra Trees.

## Outputs

- `results/metrics/notebook_steels_model_summary.csv`
- `results/metrics/notebook_steels_model_comparison.csv`
- `results/figures/notebook_steels_model_comparison.png`
- `results/figures/notebook_steels_final_model_comparison.png`
