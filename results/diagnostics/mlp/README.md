# MLP Diagnostic Results

These are the compact, Git-tracked outputs from the dense-neural-network
extension. The full timestamped run trees remain local because they contain
feature caches, raw predictions, logs, and other generated artifacts.

## Runs

- Plain MLP: `gpu_20260606_220748_utc`
- Inner-validation-tuned MLP: `gpu_20260606_221140_utc`
- Tasks: `matbench_jdft2d`, `matbench_phonons`
- Feature branches: Magpie composition and Magpie plus all structure descriptors
- Evaluation: official Matbench five-fold splits
- Random seed: 42

## Main result

Structure-aware features improve both neural-network variants, but neither MLP
matches the TabPFN or strongest tree baselines on these small tabular datasets.
The tuned MLP reaches MAE 55.19 on `matbench_jdft2d` and 73.99 on
`matbench_phonons` with all structure descriptors.

The `configs/`, `metrics/`, `tables/`, and `figures/` subdirectories contain
the minimal data needed to audit the reported aggregate result. Presentation
comparisons are in `presentation_graphs/fig11` through `fig16`.
