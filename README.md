# matbench-tabpfn

Benchmarks TabPFN on official Matbench folds for four materials-regression
tasks. Covers structure features, data efficiency, MLP baselines, and
model-guided candidate screening.

## Results

| Result | JDFT2D | Phonons |
|---|---:|---:|
| TabPFN + structure MAE | 34.40 meV/atom | 30.63 cm⁻¹ |
| MAE reduction vs composition only | 25.4% | 16.9% |
| Tuned MLP + structure MAE | 55.2 meV/atom | 74.0 cm⁻¹ |
| Model-guided candidate recovery | 84–87% | 97–98% |
| Random recovery at the same label budget | 29% | 21% |

MAE values are five-fold means. Screening starts from 10% labeled data;
recovery is measured on the top 5% of candidates.

## Run

Recommended: Google Colab with CUDA. Run the notebooks in order:

1. [Official-fold benchmark](notebooks/matbench_tabpfn_official_folds.ipynb)
2. [Learning curves](colab%20notebooks/Learning%20curve_data%20efficiency.ipynb)
3. [Dense-MLP baseline](colab%20notebooks/Simple%20neural-network%20baseline.ipynb)
4. [Active-learning screening](colab%20notebooks/Active%20learning.ipynb)

Set the TabPFN credential through the notebook prompt or `TABPFN_TOKEN`.

Local setup:

```bash
git clone https://github.com/ZuomingWang/matbench-tabpfn.git
cd matbench-tabpfn
bash scripts/setup_env.sh
conda activate matbench-tabpfn
python scripts/check_setup.py
```

## Check

```bash
python -m unittest discover -s tests -v
python scripts/extract_colab_notebook_evidence.py --check
python scripts/verify_results.py
```

[Results](results/README.md) · [Reproducibility](docs/reproducibility.md) · [References](docs/references.md) · [Contributors](CONTRIBUTORS.md)
