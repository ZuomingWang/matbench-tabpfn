# Next Session Context

Last updated: 2026-05-31

## Scope

Work only inside this project folder:

```text
/Users/zuoming/Documents/New project/matbench-tabpfn
```

Do not search or edit the parent folder, Google Drive, or other local folders unless the user explicitly asks.

## Project In One Sentence

This project evaluates whether TabPFN improves small-data materials property prediction on selected Matbench regression tasks, then extends a Matbench-style baseline through structure-aware featurization and error analysis.

## Active Entry Points

- Main notebook: `notebooks/matbench_tabpfn_official_folds.ipynb`
- Active config: `configs/gpu_rerun.yml`
- Reusable code: `src/matbench_tabpfn/`
- References: `docs/references.md`
- Stage report and slide outline: `docs/presentation_outline_and_stage_report.md`
- Presentation assets: `docs/presentation_assets/`

The active notebook contains the GPU rerun workflow. The latest synced result summaries and figures are also available directly under `results/`.

## Current Experiment Design

Tasks:

- `matbench_steels`
- `matbench_jdft2d`
- `matbench_phonons`
- `matbench_expt_gap`

Models:

- Dummy mean
- RidgeCV
- Random forest
- Extra trees
- HistGradientBoosting
- TabPFN
- Post-hoc TabPFN + ExtraTrees 50/50 ensemble, generated from saved predictions without retraining.
- Targeted TabPFN + ExtraTrees inner-validation tuned ensemble for selected task-feature branches.

Feature branches:

- `magpie`: Magpie composition features. For structure-input tasks, this is the composition-proxy baseline.
- `magpie_density`: Magpie plus density descriptors for structure-input tasks.
- `magpie_symmetry`: Magpie plus global symmetry descriptors for structure-input tasks.
- `magpie_packing`: Magpie plus maximum-packing-efficiency descriptors for structure-input tasks.
- `magpie_density_packing`: Magpie plus density and packing descriptors for structure-input tasks.
- `magpie_density_packing_symmetry`: Magpie plus density, packing, and symmetry descriptors for structure-input tasks.
- `magpie_complexity`: Magpie plus structural-complexity descriptors for structure-input tasks.
- `magpie_density_packing_complexity`: Magpie plus density, packing, and complexity descriptors for structure-input tasks.
- `magpie_structure_all`: Magpie plus all four structure descriptor groups.

Evaluation controls:

- Matbench official folds only.
- MAE is the primary metric.
- R2 is secondary context.
- Fold-level metrics and predictions are saved before aggregation.
- TabPFN is compared against the best non-TabPFN baseline.

## Current Result Interpretation

The latest user-run results indicated:

- `matbench_steels`: TabPFN best or near-best, about `87.7 MPa`.
- `matbench_expt_gap`: TabPFN best or near-best, about `0.369 eV`.
- `matbench_jdft2d`: all-structure TabPFN best or near-best, about `34.4 meV/atom`.
- `matbench_phonons`: all-structure TabPFN best or near-best, about `30.6 cm^-1`.

Current conclusion:

TabPFN is competitive and often best on the selected small-data Matbench tasks. The stronger scientific claim is not simply that TabPFN wins, but that TabPFN plus meaningful feature representation is especially effective. For structure-input tasks, structure-aware descriptors are necessary because composition-only features are a weaker proxy.

## Known Issue

`matbench_expt_gap` has a RidgeCV fold with an unusually high MAE above `40 eV`. This is interpreted as a fold-specific extrapolation failure of a linear Magpie-feature baseline, not as a GPU training issue.

The fold-dispersion plot now handles this with a robust y-axis and marks clipped catastrophic folds at the top of the panel.

Expected plotting version printed by the figure cell after the report-structure-ablation update:

```text
Plot style version: report_structure_ablation_v1
```

If the notebook prints an older plot style version, rerun the first code cell to synchronize the embedded source files, then rerun the figure-generation cell.

## Output Locations

Latest synced result folders:

```text
results/figures/
results/metrics/
results/tables/
```

Large raw generated outputs may exist locally under `results/features/`,
`results/logs/`, `results/predictions/`, or `results/runs/`, but these are
ignored by Git by default.

GPU or Colab reruns may also save to:

```text
/content/drive/MyDrive/MAT459 - project/results
```

Each full run writes:

```text
results/runs/<run_id>/
  features/
  figures/
  logs/
  metrics/
  predictions/
  tables/
```

Important synced files:

- `metrics/fold_metrics.csv`
- `metrics/model_summary.csv`
- `metrics/best_baseline_comparison.csv`
- `metrics/paired_fold_comparisons.csv`
- `tables/auto_summary.md`
- `tables/best_models_by_task.csv` (headline best model per task)
- `tables/top_absolute_errors.csv`
- `tables/tabpfn_vs_best_baseline_sample_errors.csv`
- `tables/structure_feature_branch_summary.csv` (one row per structure branch compared with the Magpie composition-proxy branch)
- `tables/tabpfn_extra_trees_ensemble_summary.csv` (fixed 50/50 ensemble compared with its TabPFN and ExtraTrees components)
- `tables/tabpfn_extra_trees_weight_scan.csv` (diagnostic test-fold scan over TabPFN/ExtraTrees weights)
- `tables/tabpfn_extra_trees_weight_scan_best.csv` (best diagnostic weight per task-feature branch; not leakage-free model selection)
- `tables/tabpfn_extra_trees_inner_tuned_summary.csv` (leakage-free fold-wise tuned ensemble compared with TabPFN and fixed 50/50 ensemble)
- `tables/tabpfn_extra_trees_inner_tuned_inner_validation_weights.csv` (selected inner-validation weights by official fold)
- `figures/03_tabpfn_vs_best_baseline.png` (presentation-useful internal baseline comparison)
- `figures/05_structure_feature_branch_comparison.png` (presentation-useful structure branch ablation)

## Next Recommended Extensions

Priority 1: run the structure feature ablation.

The code and config now support these feature branches for structure-input tasks only:

- `magpie`
- `magpie_density`
- `magpie_symmetry`
- `magpie_packing`
- `magpie_density_packing`
- `magpie_density_packing_symmetry`
- `magpie_complexity`
- `magpie_density_packing_complexity`
- `magpie_structure_all`

Run these on `matbench_jdft2d` and `matbench_phonons`. This directly satisfies the project requirement to extend featurization and can identify whether the gain comes from a single descriptor group, lower-cost density/packing combinations, the new density/packing/symmetry branch, or the full structure stack.

Priority 2: evaluate the TabPFN plus ExtraTrees ensemble.

The code now builds a fixed-weight ensemble using saved predictions:

```text
ensemble = 0.5 * TabPFN + 0.5 * ExtraTrees
```

This is cheap and avoids retraining. Rerun the figure/report cell after the full experiment to append the ensemble rows, rewrite derived CSVs, create `tabpfn_extra_trees_ensemble_summary.csv`, and run a diagnostic weight scan. The weight scan uses official test-fold predictions, so it is for diagnosis only. If it suggests useful nontrivial weights, upgrade to an inner-validation tuned weight per official train fold to avoid test leakage.

The notebook now includes an optional `6b. Inner-validation tuned ensemble` cell. It currently targets:

- `matbench_steels` with `magpie`
- `matbench_phonons` with `magpie_structure_all`

For each official fold, it splits the official train fold into inner train/validation data, trains TabPFN and ExtraTrees on the inner train split, selects a weight from `0.00, 0.05, ..., 1.00` using inner-validation MAE, then applies that selected weight to the already saved full-train official test predictions.

Suggested two-person split:

- Person A: reference-paper framing, classical baselines, structure feature ablation, method/results tables.
- Person B: TabPFN runs, ensemble model, error analysis, figures, conclusions.

## Resume Checklist

1. Open `notebooks/matbench_tabpfn_official_folds.ipynb`.
2. Run the first code cell to locate or bootstrap the project.
3. Run the CUDA check and dependency-install cells. If the dependency cell installs packages or repairs a binary ABI mismatch, restart the kernel/runtime immediately, rerun cells 1 through 2b, and only then continue. This prevents SciPy/scikit-learn/TabPFN compiled-extension mismatches in Colab or remote kernels. Matbench is still installed as `matbench==0.6` with `--no-deps` to avoid its old dependency pins.
4. Set the TabPFN token through the notebook prompt or environment variable. Do not write the token into files.
5. Confirm tasks and models from `configs/gpu_rerun.yml`.
6. To regenerate figures without retraining, rerun only the figure-generation and display cells after loading existing result CSVs.
7. Confirm the figure cell prints `report_structure_ablation_v1`.

## Cleaning Notes

The working folder was simplified by removing macOS `.DS_Store`, clearing notebook outputs, deleting duplicate Markdown notes, removing unused placeholder folders (`data/`, `reports/`, and `tests/`), deleting stale archive docs/configs/scripts, and renaming the active notebook to `notebooks/matbench_tabpfn_official_folds.ipynb`. A backup of the earlier notebook is stored at `archive/notebooks/04_vscode_gpu_matbench_rerun_backup_2026-05-31.ipynb`.

`docs/.references.md.swp` was intentionally kept because it may correspond to an editor swap file. Remove it only after confirming no editor session needs it.
