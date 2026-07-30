# Active-learning evidence

`tables/notebook_summary.csv` contains 40 standard conditions and 16
stress-test conditions extracted from the executed notebook.
`provenance.json` records run mappings, configuration, field origins, hashes,
precision, and extraction rules.

```bash
python scripts/extract_colab_notebook_evidence.py --check
```

These files preserve displayed aggregates only. Raw predictions, acquisition
traces, and full timestamped runs are not included.
