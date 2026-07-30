# Matbench Experimental Band Gap Model Comparison

Date: 2026-05-30

## Goal

Check whether the workflow and modeling conclusions from `matbench_steels` transfer to a second Matbench composition task.

## Notebook

`notebooks/02_matbench_expt_gap_model_comparison.ipynb`

## Setup

- Task: `matbench_expt_gap`
- Target: experimental band gap
- Unit: eV
- Samples: 4604
- Folds: official Matbench 5-fold split
- Features: 132 Magpie composition features

## Results

| Model | Mean MAE (eV) | Std MAE | Mean R2 | Features |
| --- | ---: | ---: | ---: | ---: |
| Extra Trees | 0.409 | 0.027 | 0.693 | 132 |
| Voting ensemble | 0.420 | 0.027 | 0.695 | 132 |
| HistGradientBoosting | 0.433 | 0.031 | 0.682 | 132 |
| Random Forest | 0.446 | 0.021 | 0.670 | 132 |
| Dummy mean baseline | 1.144 | 0.035 | -0.002 | 132 |
| RidgeCV | 9.972 | 20.562 | -188236.342 | 132 |

## Interpretation

Extra Trees is again the strongest non-TabPFN model, improving over Random Forest on a second task. Together with the steels result, this shows that randomized tree ensembles are a robust classical baseline for Magpie composition features on these small-to-medium Matbench regression tasks.

RidgeCV performs poorly on this task, likely because the linear model is not well matched to the nonlinear composition-property relationship and the descriptor space is highly correlated. The dummy mean baseline is included as a stable lower reference.

## Outputs

- `results/metrics/notebook_expt_gap_model_summary.csv`
- `results/metrics/notebook_expt_gap_model_comparison.csv`
- `results/predictions/notebook_expt_gap_model_predictions.csv`
- `results/figures/notebook_expt_gap_model_comparison.png`
- `results/figures/notebook_expt_gap_best_model_parity.png`
