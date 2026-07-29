# Small-Data Notebook Evidence

This directory contains sanitized, machine-readable values extracted from the
already executed `colab notebooks/Learning curve_data efficiency.ipynb`.
Kyle Xu's runner and notebook remain unchanged.

The canonical presentation run is `gpu_20260605_223535_utc`. The extractor
checks the exact 240-condition Cartesian log in the successful run cell, the
subsequent `latest` selector, the base summary reproduced by that run, and the
postprocessor's run-ID-bearing output and verbatim extended summary. The
original timestamped CSV tree is absent, so `tables/notebook_summary.csv`
contains only the 26 numeric claims explicitly displayed in the notebook.

Regenerate or verify the evidence with:

```bash
python scripts/extract_colab_notebook_evidence.py
python scripts/extract_colab_notebook_evidence.py --check
```

The extractor reads allowlisted summary fields only. It never copies raw
stdout, interactive prompts, or credentials. `provenance.json` records source
hashes, execution order, run lineage, executed runner/postprocessor blobs,
task units, aggregation semantics, precision, and limitations.

This evidence supports a notebook-display audit. It is not a replacement for
the missing full learning-curve, target-regime, prediction, or environment
tables.
