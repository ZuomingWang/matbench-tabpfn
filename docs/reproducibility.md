# Reproducibility

## Environment

Colab with a CUDA runtime is the primary target.

```bash
conda env create -f environment.yml
conda activate matbench-tabpfn
python scripts/check_setup.py
```

`environment.yml` is a rerun environment, not a lockfile. The recorded MLP
runtime is stored in `results/diagnostics/mlp/configs/`.

Supply the TabPFN credential through an interactive prompt or
`TABPFN_TOKEN`. Do not write credentials to notebooks, logs, or configuration
files.

## Workflows

### Official benchmark

[`notebooks/matbench_tabpfn_official_folds.ipynb`](../notebooks/matbench_tabpfn_official_folds.ipynb)

Runs four selected Matbench regression tasks on official folds with classical
baselines, TabPFN, structure-feature ablations, and ensemble diagnostics.

### Learning curves

[`colab notebooks/Learning curve_data efficiency.ipynb`](../colab%20notebooks/Learning%20curve_data%20efficiency.ipynb)

```bash
python scripts/run_small_data_diagnostics.py \
  --models extra_trees tabpfn \
  --fractions 0.1 0.2 0.4 0.6 0.8 1.0 \
  --repeats 1 \
  --device cuda
python scripts/summarize_small_data_diagnostics.py
```

### Dense MLP

[`colab notebooks/Simple neural-network baseline.ipynb`](../colab%20notebooks/Simple%20neural-network%20baseline.ipynb)

```bash
python scripts/run_mlp_baseline_diagnostics.py
python scripts/run_tuned_mlp_baseline_diagnostics.py
```

### Active learning

[`colab notebooks/Active learning.ipynb`](../colab%20notebooks/Active%20learning.ipynb)

The recorded grid is in
[`configs/extension_runs.yml`](../configs/extension_runs.yml). Active learning
is evaluated retrospectively on official-fold training pools.

## Evidence

| Section | Evidence in a fresh clone |
|---|---|
| Official benchmark | Fold and aggregate tables |
| Dense MLP | Fold metrics, selected configurations, and summaries |
| Learning curves | Values extracted from executed notebook output |
| Active learning | Aggregate values extracted from executed notebook output |

The compact notebook-derived tables preserve only displayed values. They do
not replace missing raw predictions, full learning curves, or acquisition
traces.

## Verification

```bash
python -m unittest discover -s tests -v
python scripts/extract_colab_notebook_evidence.py --check
python scripts/verify_results.py
```

The standard verifier checks committed evidence. A stricter source-summary
check becomes available when the original extension run directories are
restored:

```bash
python scripts/verify_results.py \
  --small-data-run results/small_data_diagnostics/<run_id> \
  --active-learning-run results/active_learning_screening/<run_id_1> \
  --active-learning-run results/active_learning_screening/<run_id_2> \
  --active-learning-run results/active_learning_screening/<run_id_3> \
  --active-learning-run results/active_learning_screening/<run_id_4> \
  --require-extensions
```

Without those directories, strict mode exits with code 2.

## Figures

```bash
python scripts/make_benchmark_figures.py
python scripts/make_mlp_figures.py
```

Aggregate and fold-level figures rebuild from committed tables. The two parity
plots require raw prediction files and are skipped when those files are
absent.

## Stable reruns

The extension notebooks currently follow the repository's `main` branch. For
an archival run, check out a commit SHA first and record the runtime, GPU,
CUDA version, configuration, seed, and output run ID.
