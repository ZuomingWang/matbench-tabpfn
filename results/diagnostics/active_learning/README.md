# Active-Learning Notebook Evidence

This directory contains sanitized aggregate values extracted from the already
executed `colab notebooks/Active learning.ipynb`. Kyle Xu's runner and
notebook remain unchanged.

`tables/notebook_summary.csv` contains:

- 40 standard presentation conditions at the precision displayed by the
  notebook HTML; and
- all 16 conditions from the 5%-initial/top-1% stress-test table.

The standard notebook table displays final regret, final top-candidate hit
rate, and final best-objective percentile for every condition. Other standard
aggregate fields are also recoverable for 15 conditions from two additional
displayed tables. Fields not displayed for the remaining 25 conditions are
left blank rather than inferred. The stress table displays its full aggregate
columns. Every run-cell condition log and both aggregate-table grids are
validated as exact Cartesian sets, not only counted.

Regenerate or verify the evidence with:

```bash
python scripts/extract_colab_notebook_evidence.py
python scripts/extract_colab_notebook_evidence.py --check
```

The extractor reads allowlisted HTML table cells and completion identifiers
only. It never copies raw stdout, interactive prompts, or credentials.
`provenance.json` records run mappings, exact grids, field-level origins,
runner/helper blobs, effective configuration, notebook hashes, precision, and
limitations. Because the runtime clone did not print its HEAD, the recorded
active-learning commit is explicitly labeled as a repository-history
compatibility inference rather than direct runtime proof.

This evidence supports a notebook-display audit. It does not restore the
missing trace CSV, raw predictions, feature caches, or full-precision result
trees.
