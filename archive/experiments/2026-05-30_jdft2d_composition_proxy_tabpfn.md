# Matbench JDFT2D Composition-Proxy TabPFN Result

Date: 2026-05-30

## Goal

Test whether TabPFN also improves over classical models on a smaller Matbench task beyond `matbench_steels`.

`matbench_jdft2d` is a structure task. For this quick extension, each structure was converted to its composition and featurized with Magpie descriptors. This is a composition-only proxy, not a full structure model.

## Setup

- Task: `matbench_jdft2d`
- Matbench input type: structure
- Feature source used here: composition extracted from structure
- Target: exfoliation energy
- Unit: meV/atom
- Samples: 636
- Folds: official Matbench 5-fold split
- Features: 132 Magpie composition features

## Results

| Model | Mean MAE (meV/atom) | Std MAE | Mean R2 |
| --- | ---: | ---: | ---: |
| Random Forest | 50.086 | 8.939 | 0.120 |
| Extra Trees | 49.435 | 8.708 | 0.014 |
| TabPFN | 46.150 | 8.399 | 0.094 |

## TabPFN Fold Metrics

| Fold | Train Size | Test Size | MAE (meV/atom) | R2 |
| --- | ---: | ---: | ---: | ---: |
| 0 | 508 | 128 | 40.431 | -0.237 |
| 1 | 509 | 127 | 47.690 | 0.155 |
| 2 | 509 | 127 | 57.699 | 0.346 |
| 3 | 509 | 127 | 35.886 | 0.146 |
| 4 | 509 | 127 | 49.044 | 0.058 |

## Interpretation

TabPFN gives the lowest MAE on this composition-only proxy task:

- 7.86% lower MAE than Random Forest.
- 6.65% lower MAE than Extra Trees.

The low and inconsistent R2 values show that composition alone is not enough to model this structure-dependent property well. This is still useful because it isolates the effect of TabPFN on the same limited feature representation.

## Outputs

- `results/metrics/matbench_jdft2d_random_forest_magpie_summary.json`
- `results/metrics/matbench_jdft2d_extra_trees_magpie_summary.json`
- `results/metrics/matbench_jdft2d_tabpfn_magpie_summary.json`
- `results/predictions/matbench_jdft2d_tabpfn_magpie_predictions.csv`
