# Matbench Experimental Band Gap TabPFN OOM Attempt

Date: 2026-05-30

## Goal

Attempt to run `TabPFNRegressor` on `matbench_expt_gap` using the same Magpie composition features and official Matbench folds.

## Result

The run failed with MPS out-of-memory errors during prediction.

First attempt:

- Test fold size: 921 samples.
- Error: MPS out of memory during prediction.

Second attempt:

- Added `--predict-batch-size 100`.
- Still failed with MPS out of memory on a 100-sample prediction batch.

## Interpretation

This suggests the larger training set size for `matbench_expt_gap` makes local MPS inference too memory-heavy in the current environment. TabPFN was feasible on smaller tasks but did not run locally on the 4604-sample band-gap task without further memory-management changes or different hardware.

## Next Options

- Try CPU inference, likely slower.
- Try smaller `n_estimators`.
- Subsample the training set to study TabPFN small-data scaling.
- Use TabPFN only for `matbench_steels` and `matbench_jdft2d`, while keeping classical models for `matbench_expt_gap`.
