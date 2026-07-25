# Matbench TabPFN Small-Data Workflow

This project evaluates a TabPFN/ICL-FM-style workflow for small-data materials
property prediction on selected Matbench regression tasks. It reproduces the
core official-fold benchmark, extends the representation with lightweight
structure-aware features, and adds data-efficiency, neural-network, and
active-learning diagnostics.

The main result is that physically meaningful structure descriptors
substantially improve performance on structure-dependent materials tasks.
TabPFN is strong in the small-data regime, but the diagnostic extensions show
that representation, model class, and label-acquisition strategy all matter.

## Start Here

- Main notebook: `notebooks/matbench_tabpfn_official_folds.ipynb`
- Experiment config: `configs/gpu_rerun.yml`
- Small-data/data-efficiency workflow: `scripts/run_small_data_diagnostics.py`
- Dense-neural-network workflows:
  `scripts/run_mlp_baseline_diagnostics.py` and
  `scripts/run_tuned_mlp_baseline_diagnostics.py`
- Active-learning workflow: `scripts/run_active_learning_screening.py`
- Stage report and slide outline: `docs/presentation_outline_and_stage_report.md`
- Presentation script and reproducibility notes: `docs/presentation_script.md`
- Presentation figures: `presentation_graphs/`
- Latest synced results: `results/`

## Workflow

### 1. Set up the environment

```bash
bash scripts/setup_env.sh
conda activate matbench-tabpfn
python scripts/check_setup.py
```

Do not save a TabPFN token in project files. Use the notebook prompt or the
`TABPFN_TOKEN` environment variable.

### 2. Run the official-fold benchmark

Use `notebooks/matbench_tabpfn_official_folds.ipynb` for the primary
TabPFN/classical-model study. `configs/gpu_rerun.yml` records the reproducible
GPU rerun configuration.

### 3. Run optional diagnostic extensions

```bash
python scripts/run_small_data_diagnostics.py
python scripts/run_mlp_baseline_diagnostics.py
python scripts/run_tuned_mlp_baseline_diagnostics.py
python scripts/run_active_learning_screening.py
```

Each diagnostic creates a timestamped run below `results/`. Raw feature
caches, predictions, logs, and run snapshots stay local and are ignored by
Git. Curated summary tables and figures are committed under
`results/diagnostics/`.

### 4. Regenerate presentation figures

```bash
python scripts/make_presentation_figures.py
python scripts/make_results2_figures.py
```

The second command uses the newest local MLP and tuned-MLP diagnostic runs.
Both commands write slide-ready PNG and PDF files to `presentation_graphs/`.

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
presentation_graphs/fig1_structure_gain.png
presentation_graphs/fig5_tabpfn_vs_baseline.png
presentation_graphs/fig11_nn_structure_gain.png
presentation_graphs/fig12_combined_leaderboard.png
```

## Headline Interpretation

- TabPFN is competitive on the selected small-data Matbench tasks, but it does not uniformly beat the published paper or leaderboard numbers.
- Structure-aware descriptors are the clearest improvement:
  - `matbench_jdft2d`: best structure branch improves over the composition proxy by about 25% MAE.
  - `matbench_phonons`: best structure branch improves over the composition proxy by about 17% MAE.
- The project should be presented as a staged reproduction of the TabPFN/ICL-FM idea plus a structure-aware featurization extension.
- A plain or inner-validation-tuned dense MLP improves with structure features
  but remains behind TabPFN and the strongest tree baselines on the two
  structure tasks.
- The small-data and active-learning extensions test label efficiency and
  model-guided candidate screening beyond the static full-data benchmark.
- This stage does not reproduce the paper's ALIGNN/CGCNN embedding pipeline.

## Repository Layout

- `configs/`: experiment configuration.
- `colab notebooks/`: Colab entrypoints for the extension workflows.
- `docs/`: project context, references, presentation outline, and speaker script.
- `notebooks/`: active runnable notebook.
- `presentation_graphs/`: slide-ready figures from the primary and MLP studies.
- `src/matbench_tabpfn/`: reusable feature, model, evaluation, plotting, and analysis code.
- `scripts/`: environment setup, experiment runners, summaries, and figure generation.
- `results/`: synced primary results and curated diagnostic summaries.
- `ref/` and `ref_lab_notebook/`: reference papers and course notebooks.
- `archive/`: older exploratory notebooks and notes kept for provenance.

## GitHub Notes

The repository is configured to keep key result summaries and figures while ignoring heavier generated files such as raw predictions, feature caches, logs, temporary PDF renders, and rerun directories.
