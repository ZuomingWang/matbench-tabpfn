# Matbench TabPFN Structure-Feature Study

This project evaluates a TabPFN/ICL-FM-style workflow for small-data materials property prediction on selected Matbench regression tasks, then extends the baseline with lightweight structure-aware features.

The main result is that physically meaningful structure descriptors substantially improve TabPFN performance on structure-dependent materials tasks.

## Start Here

- Main notebook: `notebooks/matbench_tabpfn_official_folds.ipynb`
- Experiment config: `configs/gpu_rerun.yml`
- Stage report and slide outline: `docs/presentation_outline_and_stage_report.md`
- Presentation figures: `docs/presentation_assets/`
- Latest synced results: `results/`

## Current Results

The current run evaluates:

- `matbench_steels`
- `matbench_expt_gap`
- `matbench_jdft2d`
- `matbench_phonons`

The current best-model summary is in:

```text
results/tables/best_models_by_task.csv
```

The complete model summary is in:

```text
results/metrics/model_summary.csv
```

The most useful presentation figures are:

```text
docs/presentation_assets/03_tabpfn_vs_best_baseline.png
docs/presentation_assets/05_structure_feature_branch_comparison.png
```

## Headline Interpretation

- TabPFN is competitive on the selected small-data Matbench tasks, but it does not uniformly beat the published paper or leaderboard numbers.
- Structure-aware descriptors are the clearest improvement:
  - `matbench_jdft2d`: best structure branch improves over the composition proxy by about 25% MAE.
  - `matbench_phonons`: best structure branch improves over the composition proxy by about 17% MAE.
- The project should be presented as a staged reproduction of the TabPFN/ICL-FM idea plus a structure-aware featurization extension.
- This stage does not reproduce the paper's ALIGNN/CGCNN embedding pipeline.

## Setup

```bash
bash scripts/setup_env.sh
conda activate matbench-tabpfn
python scripts/check_setup.py
```

Do not save a TabPFN token in project files. Use the notebook prompt or an environment variable.

## Repository Layout

- `configs/`: experiment configuration.
- `docs/`: project context, references, presentation outline, and presentation assets.
- `notebooks/`: active runnable notebook.
- `src/matbench_tabpfn/`: reusable feature, model, evaluation, plotting, and analysis code.
- `scripts/`: environment setup and dependency checks.
- `results/`: synced result summaries, tables, and figures.
- `ref_lab_notebook/`: assignment reference notebooks and reference paper PDFs.
- `archive/`: older exploratory notebooks and notes kept for provenance.

## GitHub Notes

The repository is configured to keep key result summaries and figures while ignoring heavier generated files such as raw predictions, feature caches, logs, temporary PDF renders, and rerun directories.
