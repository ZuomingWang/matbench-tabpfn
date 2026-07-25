# Presentation Reproducibility and Contribution Boundary

This document maps the June 8, 2026 presentation and its matching transcript
to the repository. Its two goals are:

1. preserve Kyle Xu's contribution without rewriting his experiment runners
   or Colab notebooks; and
2. make every numerical presentation claim traceable to a command and result
   table.

## Contributor boundary

Kyle Xu introduced the following workflow files in commits `cfd0314`,
`2f28d78`, `a2cc6df`, `7cf3fc1`, and `6f86616`:

- `scripts/run_small_data_diagnostics.py`
- `scripts/summarize_small_data_diagnostics.py`
- `scripts/run_mlp_baseline_diagnostics.py`
- `scripts/run_tuned_mlp_baseline_diagnostics.py`
- `scripts/run_active_learning_screening.py`
- `colab notebooks/Learning curve_data efficiency.ipynb`
- `colab notebooks/Simple neural-network baseline.ipynb`
- `colab notebooks/Active learning.ipynb`

The repository update intentionally leaves those files unchanged. New
documentation, compact result copies, and verification helpers are additive.
The presentation itself attributes learning curves, target-regime analysis,
untuned/tuned MLP baselines, and active-learning screening to Kyle.

## Reproduction status

| Presentation section | Slides | Entry point | Evidence in a fresh clone | Status |
|---|---:|---|---|---|
| Official-fold benchmark, structure ablation, robustness, ensemble | 1–12 | `notebooks/matbench_tabpfn_official_folds.ipynb` | `results/metrics/`, `results/tables/`, `presentation_graphs/fig1`–`fig10` | Auditable |
| Learning curves, AULC, high-target regime | 14–16 | `scripts/run_small_data_diagnostics.py`; `scripts/summarize_small_data_diagnostics.py` | Runner and Colab notebook only | Rerunnable, original result tree missing |
| Untuned and tuned MLP | 17–19 | `scripts/run_mlp_baseline_diagnostics.py`; `scripts/run_tuned_mlp_baseline_diagnostics.py` | `results/diagnostics/mlp/`, `presentation_graphs/fig11`–`fig16` | Auditable |
| Active-learning screening | 20–24 | `scripts/run_active_learning_screening.py` | Runner and Colab notebook only | Rerunnable, original result trees missing |

“Rerunnable” means the code and run parameters are present. “Auditable” also
means the compact source tables needed to compare a rerun with the presented
numbers are committed. Until the small-data and active-learning result tables
are imported or regenerated, a fresh clone cannot independently verify every
rounded value in slides 14–16 and 20–24.

## Environment

The recorded MLP runs used:

- Python 3.12.13 on Linux
- `matbench==0.6`
- `matminer==0.10.1`
- `numpy==2.0.2`
- `pandas==2.2.2`
- `scikit-learn==1.6.1`
- `tabpfn==8.0.3`
- `torch==2.11.0+cu128`
- NVIDIA RTX PRO 6000 Blackwell Server Edition
- random seed 42

The compact environment snapshots are in
`results/diagnostics/mlp/configs/`. TabPFN runs require a CUDA-capable GPU and
a TabPFN token supplied through `TABPFN_TOKEN` or the notebook prompt. Do not
commit the token.

For a fresh environment:

```bash
bash scripts/setup_env.sh
conda activate matbench-tabpfn
python scripts/check_setup.py
```

The Colab notebooks are the most direct route when a local CUDA GPU is not
available.

## Exact extension commands

### Slides 14–16: data efficiency and target regimes

```bash
python scripts/run_small_data_diagnostics.py \
  --models extra_trees tabpfn \
  --fractions 0.1 0.2 0.4 0.6 0.8 1.0 \
  --repeats 1 \
  --device cuda

python scripts/summarize_small_data_diagnostics.py
```

The generated run should contain:

```text
tables/small_data_learning_curve_summary.csv
tables/target_regime_summary.csv
tables/aulc_data_efficiency_summary.csv
tables/high_target_structure_improvement.csv
figures/small_data_learning_curve.png
figures/aulc_data_efficiency_ranking.png
figures/high_target_structure_improvement.png
logs/config.json
logs/environment.json
```

### Slides 17–19: MLP baselines

```bash
python scripts/run_mlp_baseline_diagnostics.py
python scripts/run_tuned_mlp_baseline_diagnostics.py
python scripts/make_results2_figures.py
```

The committed compact tables reproduce the slide values:

| Task | MLP composition | Tuned MLP composition | MLP + structure | Tuned MLP + structure | TabPFN + structure |
|---|---:|---:|---:|---:|---:|
| JDFT2D | 63.3 | 64.0 | 55.4 | 55.2 | 34.4 |
| Phonons | 96.5 | 101.2 | 77.4 | 74.0 | 30.6 |

### Slides 20–24: active-learning screening

The presentation runs each task/representation combination separately and
combines the compact outputs:

```bash
for task in matbench_jdft2d matbench_phonons; do
  for feature_set in magpie magpie_structure_all; do
    python scripts/run_active_learning_screening.py \
      --tasks "$task" \
      --feature-sets "$feature_set" \
      --strategies random extra_trees_greedy extra_trees_ucb tabpfn_greedy tabpfn_disagreement_ucb \
      --initial-fractions 0.1 0.2 \
      --max-acquisitions 100 \
      --acquisition-batch-size 10 \
      --repeats 3 \
      --device cuda
  done
done
```

For each run, preserve:

```text
tables/active_learning_aggregate_summary.csv
tables/active_learning_run_summary.csv
predictions/active_learning_trace.csv
logs/config.json
logs/environment.json
```

The clean 10%-initial-label summary in the presentation reports:

| Task | Random | ExtraTrees greedy | ExtraTrees UCB | TabPFN greedy | TabPFN disagreement UCB |
|---|---:|---:|---:|---:|---:|
| JDFT2D top-candidate hit rate | 0.29 | 0.84 | 0.86 | 0.87 | 0.86 |
| Phonons top-candidate hit rate | 0.21 | 0.98 | 0.98 | 0.97 | 0.98 |
| JDFT2D final regret | 0.244 | 0.212 | 0.263 | 0.283 | 0.263 |
| Phonons final regret | 0.106 | 0.000 | 0.000 | 0.000 | 0.000 |

These values are transcription targets from the presentation, not substitutes
for the missing source tables. The source tables should be committed under a
future `results/diagnostics/active_learning/` directory after rerun or import.

## Verification

Check the committed primary and MLP values:

```bash
python scripts/verify_presentation_results.py
```

After the missing runs are available:

```bash
python scripts/verify_presentation_results.py \
  --small-data-run results/small_data_diagnostics/<run_id> \
  --active-learning-run results/active_learning_screening/<run_id> \
  --active-learning-run results/active_learning_screening/<run_id> \
  --active-learning-run results/active_learning_screening/<run_id> \
  --active-learning-run results/active_learning_screening/<run_id> \
  --require-extensions
```

The active-learning verifier checks the 10%-initial-label,
`magpie_structure_all` condition used for the clean summary bars. The
representation-sensitivity plot on slide 20 uses both `magpie` and
`magpie_structure_all`.

## Recovery

The pre-update repository state is retained on
`backup/pre-workflow-update-2026-07-25` at commit `38d4e92`. The update is
isolated on a draft pull request and should not be merged until the missing
extension result trees are either restored or rerun and verified.
