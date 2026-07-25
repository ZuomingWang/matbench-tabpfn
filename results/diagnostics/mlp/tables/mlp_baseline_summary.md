# MLP Baseline Diagnostic Summary

This optional extension tests whether a plain dense neural network can compete with the TabPFN/ExtraTrees workflow on the same descriptor matrices. It is mainly a Lab3 connection and a model-complexity check.

- `matbench_jdft2d`: best MLP branch is Magpie + all structure, MAE=55.36, R2=0.1797.
- `matbench_phonons`: best MLP branch is Magpie + all structure, MAE=77.35, R2=0.916.

Interpretation guide: if the MLP underperforms TabPFN or tree baselines, that supports the course theme that dense neural networks are not automatically better on small, high-dimensional tabular materials data.
