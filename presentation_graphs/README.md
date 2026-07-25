# Presentation Graphs

This directory is the slide-ready visual output for the final project.

- `fig1`-`fig10`: primary TabPFN, classical-baseline, structure-feature,
  published-comparison, fold, ensemble, and parity results. Regenerate with
  `python scripts/make_presentation_figures.py`.
- `fig11`-`fig16`: MLP and tuned-MLP structure gain, combined leaderboard,
  tuning, parity, fold-spread, and model-class diagnostics. Regenerate with
  `python scripts/make_results2_figures.py` after running the two MLP
  diagnostic workflows.

Both PNG and PDF versions are kept so the same figure set can be used in
slides, documents, and print-quality exports.
