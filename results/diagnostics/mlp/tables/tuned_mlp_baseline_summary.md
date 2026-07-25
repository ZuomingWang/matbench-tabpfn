# Tuned MLP Baseline Summary

The MLP hyperparameters were selected by inner validation within each official Matbench training fold. The official test folds were used only for final evaluation.

- `matbench_jdft2d`: best tuned MLP branch is Magpie + all structure, MAE=55.19, R2=0.2006.
- `matbench_phonons`: best tuned MLP branch is Magpie + all structure, MAE=73.99, R2=0.9182.

Use this as a Lab3-linked model-complexity check: even a tuned dense NN must be judged against TabPFN and tree baselines on the same official folds.
