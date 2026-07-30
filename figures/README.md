# Figures

- `fig1`–`fig10`: benchmark, feature, fold, ensemble, and parity results
- `fig11`–`fig16`: dense-MLP diagnostics and combined comparisons

Rebuild figures supported by committed aggregate and fold tables:

```bash
python scripts/make_benchmark_figures.py
python scripts/make_mlp_figures.py
```

`fig10` and `fig14` require raw prediction files. If they are absent, the
scripts skip those figures and complete normally.
