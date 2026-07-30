# Learning-curve evidence

`tables/notebook_summary.csv` contains 26 values extracted from the executed
learning-curve notebook. `provenance.json` records source hashes, run lineage,
units, precision, and extraction rules.

```bash
python scripts/extract_colab_notebook_evidence.py --check
```

These files preserve displayed values only. The original full-precision
tables and raw predictions are not included.
