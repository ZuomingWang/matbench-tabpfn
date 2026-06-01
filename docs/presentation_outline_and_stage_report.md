# Final Presentation Outline and Stage Report

Project direction: TabPFN/ICL-FM reproduction + structure-aware feature extension

Last updated: 2026-06-01

## Deliverable Scope

Final deliverables:

- A runnable notebook: `notebooks/matbench_tabpfn_official_folds.ipynb`
- A 15-20 min presentation based on the outline below

Reference materials:

- Main paper: `ref_lab_notebook/s41524-026-02089-8_reference.pdf`
- Supplementary file: `ref_lab_notebook/41524_2026_2089_MOESM1_ESM.pdf`
- Course notebooks:
  - `ref_lab_notebook/Lab1_ml_intro_clean.ipynb`
  - `ref_lab_notebook/Lab2_Featurization.ipynb`
  - `ref_lab_notebook/Lab3A_SignLanguage_NN.ipynb`
  - `ref_lab_notebook/Lab3B_SignLanguage_CNN.ipynb`

Current result source:

- Metrics: `results/metrics/model_summary.csv`
- Summary tables: `results/tables/`
- Figures: `results/figures/`

Figure assets copied for the presentation:

- `docs/presentation_assets/03_tabpfn_vs_best_baseline.png`
- `docs/presentation_assets/05_structure_feature_branch_comparison.png`
- `docs/presentation_assets/appendix_01_model_mae_comparison.png`
- `docs/presentation_assets/appendix_02_fold_mae_distribution.png`
- `docs/presentation_assets/appendix_04_tabpfn_parity.png`

## Stage Report

### What This Project Does

This project evaluates whether a TabPFN-style in-context tabular foundation model can improve small-data materials property prediction on selected Matbench regression tasks. It follows the reference paper's broad ICL-FM idea: convert materials into tabular descriptors, provide train examples as context, and predict held-out Matbench official folds using TabPFN.

The project currently evaluates four Matbench tasks:

| Task | Target | Input type | Unit |
|---|---|---|---|
| `matbench_steels` | steel yield strength | composition | MPa |
| `matbench_expt_gap` | experimental band gap | composition | eV |
| `matbench_jdft2d` | exfoliation energy | structure | meV/atom |
| `matbench_phonons` | highest optical phonon peak | structure | cm^-1 |

Models evaluated:

- Dummy mean baseline
- RidgeCV
- Random forest
- Extra trees
- HistGradientBoosting
- TabPFN
- Post-hoc TabPFN + ExtraTrees 50/50 ensemble
- Inner-validation tuned TabPFN + ExtraTrees ensemble for selected task-feature branches

Feature branches:

- Magpie composition features
- Structure-aware extensions for structure-input tasks:
  - density
  - packing
  - symmetry
  - structural complexity
  - selected combinations
  - all structure descriptors

Evaluation controls:

- Official Matbench five-fold splits
- MAE as primary metric
- R2 as secondary metric
- Fold-level metrics and predictions saved
- TabPFN compared against the strongest non-TabPFN baseline
- No official test-fold tuning for the final model claims

### Relationship To The Reference Paper

The reference paper proposes ICL-FM for materials property prediction. It combines TabPFN with tabular materials representations such as Magpie, MagpieEX, and graph-derived ALIGNN/CGCNN embeddings.

This project is a staged reproduction and extension:

- Reproduces the TabPFN/ICL-FM-style workflow using Matbench official folds.
- Uses Magpie and matminer structure descriptors rather than the paper's ALIGNN/CGCNN embeddings.
- Extends the baseline through lightweight structure-aware descriptors.
- Adds automatic summary, error analysis, and ensemble diagnostics.

Current limitation:

- This stage does not reproduce the paper's ALIGNN/CGCNN embedding pipeline.
- Therefore, structure-aware comparisons to FMEF/ALIGNN numbers should be presented as contextual comparison, not exact same-input reproduction.

### Connection To The Four Course Notebooks

Use these links to make the final story coherent:

| Lab notebook | Reused idea in this project |
|---|---|
| Lab1 ML intro | Train/test evaluation, regression metrics, materials property prediction setup |
| Lab2 Featurization | Magpie/matminer featurization, Matbench-style benchmarking, PCA/feature thinking |
| Lab3A Dense NN | Model complexity, validation behavior, GPU workflow, overfitting/generalization |
| Lab3B CNN | Motivation for architecture changes and representation learning, while noting that this project uses tabular materials descriptors rather than image CNNs |

The strongest continuity is Lab2. The project can be framed as: Lab2 introduced materials featurization and Matbench; the reference paper uses TabPFN as an in-context model on such features; this project tests that idea and extends the featurization for structure-dependent tasks.

## Quantitative Published-Paper Comparison

Sources:

- Main paper Table 1: composition-based Matbench results for `jdft2d` and `phonons`.
- Main paper Table 2/Table 4: structure-based Matbench results for `jdft2d` and `phonons`.
- Supplementary Table S13: composition-only Matbench results for `steels` and `expt_gap`.

Lower MAE is better.

| Task | Representation | Our best MAE | Published comparison | Difference | Interpretation |
|---|---:|---:|---:|---:|---|
| `matbench_steels` | Magpie composition | 87.004 MPa | ICL-FM 83.605 MPa | +4.07% | Slightly worse than paper ICL-FM |
| `matbench_steels` | Magpie composition | 87.004 MPa | TPOT-Mat 79.947 MPa | +8.83% | Worse than composition-only leaderboard best |
| `matbench_expt_gap` | Magpie composition | 0.3688 eV | ICL-FM 0.3142 eV | +17.38% | Clear gap to paper ICL-FM |
| `matbench_expt_gap` | Magpie composition | 0.3688 eV | Darwin 0.2865 eV | +28.73% | Clear gap to leaderboard best |
| `matbench_jdft2d` | Magpie composition proxy | 46.130 meV/atom | FM-Magpie 44.736 meV/atom | +3.12% | Close to paper composition-only FM |
| `matbench_phonons` | Magpie composition proxy | 36.853 cm^-1 | FM-MagpieEX 41.948 cm^-1 | -12.15% | Numerically better than reported FM-MagpieEX, but treat as a strong result that should be rerun/verified before making it a headline claim |
| `matbench_jdft2d` | Magpie + structure descriptors | 34.396 meV/atom | FMEF 40.880 meV/atom | -15.86% | Numerically lower than paper FMEF, but not exact same representation; do not present as beating the paper's full structure method |
| `matbench_jdft2d` | Magpie + structure descriptors | 34.396 meV/atom | SOTA MODNet 33.192 meV/atom | +3.63% | Near but not better than SOTA |
| `matbench_phonons` | Magpie + structure descriptors | 30.631 cm^-1 | FMEF 29.753 cm^-1 | +2.95% | Close to paper FMEF contextual number |
| `matbench_phonons` | Magpie + structure descriptors | 30.631 cm^-1 | SOTA MegNet 28.761 cm^-1 | +6.50% | Competitive but not SOTA |

Important caveat for slides:

The `jdft2d` and `phonons` structure-aware rows are not exact same-representation reproduction, because this project uses lightweight matminer structure descriptors, while the paper also reports graph-derived ALIGNN/CGCNN embeddings. Present this as a staged extension rather than a full graph-embedding reproduction. Also avoid saying the project "beats the published paper" overall: `jdft2d` remains worse than the paper's published SOTA MODNet number, and `phonons` remains worse than the paper's published structure-aware SOTA MegNet number.

## Main Scientific Conclusions

1. TabPFN is competitive on selected small-data Matbench tasks, but it does not uniformly beat the published paper or leaderboard numbers.
2. The strongest result is the structure-aware featurization extension.
3. On structure-input tasks, composition-only features are insufficient:
   - `matbench_jdft2d`: best structure branch improves over composition proxy by 25.44% MAE.
   - `matbench_phonons`: best structure branch improves over composition proxy by 16.88% MAE.
4. Simple TabPFN + ExtraTrees ensembling is not a reliable improvement.
5. The project is now more defensible as a feature-representation study than as a claim that TabPFN alone is always better.

## Recommended Main Figures

### Figure 1 - TabPFN Relative To Classical Baselines

Use this to show where TabPFN beats or loses to the strongest non-TabPFN baseline.

![TabPFN vs best baseline](presentation_assets/03_tabpfn_vs_best_baseline.png)

Speaker point:

Green bars mean TabPFN improves MAE relative to the strongest classical baseline. Red bars mean it does not. The figure supports a nuanced claim: TabPFN is often strong, especially with useful representations, but not automatically best for every feature branch.

### Figure 2 - Structure Feature Branch Ablation

Use this as the central scientific figure.

![Structure feature branch ablation](presentation_assets/05_structure_feature_branch_comparison.png)

Speaker point:

Adding physically meaningful structure descriptors gives a large improvement on structure-dependent targets. The best branch is Magpie plus all structure descriptors for both `jdft2d` and `phonons`. Density and packing explain much of the gain; symmetry alone is weaker.

## 15-20 Min Slide Outline With Speaker Notes

Target length: 13 slides, about 16-18 minutes.

### Slide 1 - Title

Title:

TabPFN/ICL-FM for Small-Data Materials Property Prediction

Subtitle:

Reproduction-style Matbench study with structure-aware feature extension

Timing: 0.5-1 min

Speaker notes:

Introduce the project as a staged reproduction and extension of the ICL-FM paper. The goal is not only to run TabPFN, but to test whether the feature representation changes the scientific conclusion on small materials datasets.

### Slide 2 - Motivation

Main points:

- Materials datasets are often small, expensive, and heterogeneous.
- Conventional ML needs careful feature engineering and validation.
- Foundation models like TabPFN promise fast in-context prediction without task-specific training.

Timing: 1-1.5 min

Speaker notes:

Connect this to the course arc. Lab1 introduced supervised property prediction. Lab2 showed that descriptors strongly affect performance. The reference paper asks whether a pretrained tabular foundation model can make small-data prediction easier and stronger.

### Slide 3 - Reference Paper

Main points:

- Paper: In Context Learning Foundation Models for Materials Property Prediction with Small Datasets.
- Core method: ICL-FM = TabPFN + materials descriptors.
- Representations: Magpie, MagpieEX, and graph-derived ALIGNN/CGCNN embeddings.
- Benchmark: Matbench official folds.

Timing: 1.5 min

Speaker notes:

Explain that TabPFN is the prediction engine. The paper's novelty is using it as an in-context model over materials features. Emphasize that the paper reports both composition-based and structure-based results, which lets us compare our numbers quantitatively.

### Slide 4 - Course Notebook Continuity

Main points:

- Lab1: regression workflow, train/test split, metrics.
- Lab2: featurization with Magpie/matminer and Matbench thinking.
- Lab3A/B: model complexity, GPU workflow, validation and generalization.

Timing: 1 min

Speaker notes:

This project is not isolated from the labs. It extends Lab2 most directly: instead of only using classical ML on fixed features, we use TabPFN and test whether adding structure-aware descriptors improves the result.

### Slide 5 - Research Questions

Main questions:

1. Can we reproduce a similar TabPFN/ICL-FM-style workflow on Matbench official folds?
2. How close are our results to the published paper numbers?
3. Does structure-aware featurization improve over composition-only Magpie features?
4. Does a simple TabPFN + ExtraTrees ensemble improve the final model?

Timing: 1 min

Speaker notes:

Frame the project as two parts: reproduction-style benchmarking and extension. The extension is the main contribution, because we do not implement the paper's ALIGNN/CGCNN embedding pipeline in this stage.

### Slide 6 - Data And Tasks

Main points:

| Task | Samples | Target | Unit |
|---|---:|---|---|
| `matbench_steels` | 312 | yield strength | MPa |
| `matbench_jdft2d` | 636 | exfoliation energy | meV/atom |
| `matbench_phonons` | 1,265 | phonon peak frequency | cm^-1 |
| `matbench_expt_gap` | 4,604 | experimental band gap | eV |

Timing: 1-1.5 min

Speaker notes:

These tasks span composition-only and structure-input problems. The small sizes are important: they are where an in-context model like TabPFN should be useful.

### Slide 7 - Methods

Main points:

- Official Matbench five-fold splits.
- Same features used for all models within each task-feature branch.
- Baselines: Dummy, RidgeCV, Random Forest, ExtraTrees, HistGradientBoosting.
- Main model: TabPFN.
- Extension: structure descriptors from matminer.
- Diagnostics: fixed ensemble, inner-validation tuned ensemble, top-error analysis.

Timing: 1.5-2 min

Speaker notes:

Stress that the evaluation protocol avoids test-fold tuning for final claims. The ensemble weight scan is diagnostic only. The inner-validation tuned ensemble is included to check whether a leakage-free ensemble helps.

### Slide 8 - Quantitative Comparison To Published Numbers

Main table:

| Task | Our best | Published comparison | Result |
|---|---:|---:|---|
| `steels` | 87.004 MPa | ICL-FM 83.605 MPa | 4.07% worse |
| `expt_gap` | 0.3688 eV | ICL-FM 0.3142 eV | 17.38% worse |
| `jdft2d` composition | 46.130 meV/atom | FM-Magpie 44.736 | 3.12% worse |
| `phonons` composition | 36.853 cm^-1 | FM-MagpieEX 41.948 | 12.15% lower; verify before headline claim |
| `jdft2d` structure | 34.396 meV/atom | SOTA 33.192 | 3.63% worse |
| `phonons` structure | 30.631 cm^-1 | SOTA 28.761 | 6.50% worse |

Timing: 2 min

Speaker notes:

This slide should be presented carefully. The project does not beat the paper overall. It is close on `jdft2d`, numerically strong on `phonons` composition, and competitive on structure tasks. The biggest weakness is `expt_gap`, where the paper's ICL-FM and Darwin are clearly better.

### Slide 9 - TabPFN Versus Classical Baselines

Figure:

![TabPFN vs best baseline](presentation_assets/03_tabpfn_vs_best_baseline.png)

Timing: 1.5 min

Speaker notes:

Use this figure to explain internal comparison within our experiment. TabPFN beats strong classical baselines in many feature branches, especially on phonons and structure-enhanced settings. However, red bars show that TabPFN does not automatically win when the feature branch is weak.

### Slide 10 - Structure-Aware Feature Extension

Figure:

![Structure feature branch ablation](presentation_assets/05_structure_feature_branch_comparison.png)

Timing: 2 min

Speaker notes:

This is the central project contribution. The structure-aware branch moves `jdft2d` from 46.13 to 34.40 MAE and `phonons` from 36.85 to 30.63 MAE. The result supports the physical expectation that structure matters for exfoliation energy and phonon frequencies.

### Slide 11 - Ensemble And Error Analysis

Main points:

- Fixed 50/50 TabPFN + ExtraTrees helps only slightly on `steels`.
- Inner-validation tuned ensemble does not reliably beat standalone TabPFN.
- `expt_gap` has difficult outliers and a catastrophic RidgeCV fold, indicating linear extrapolation failure.
- Top-error tables show errors concentrate in extreme target regimes.

Timing: 1.5 min

Speaker notes:

This is a useful negative result. The improvement does not come from a trivial ensemble. It comes mainly from representation. Also explain that the RidgeCV outlier is not a GPU training issue; it is a fold-specific failure of a linear baseline on Magpie features.

### Slide 12 - Limitations

Main points:

- This stage does not include paper's ALIGNN/CGCNN graph embeddings.
- The published paper may use different TabPFN versions and preprocessing choices.
- Some current figures are too dense for final slides and should be simplified in the final deck.
- `expt_gap` remains below paper ICL-FM and leaderboard best.

Timing: 1 min

Speaker notes:

Be explicit about what was not done. This makes the project more credible. The correct claim is not "we fully reproduce the paper"; it is "we reproduced the TabPFN/ICL-FM-style pipeline and extended featurization with lightweight structure descriptors."

### Slide 13 - Conclusions And Next Steps

Main conclusions:

- TabPFN is a strong small-data materials predictor but not universally SOTA.
- Feature representation is the decisive factor.
- Lightweight structure descriptors substantially improve structure-dependent tasks.
- The project is a coherent staged extension of the reference paper and Lab2 featurization work.

Next steps:

- Clean final slides with compact tables and two main figures.
- Add a short final notebook cell summarizing published-paper comparison.
- Optional future extension: implement ALIGNN/CGCNN embeddings if more time is available.

Timing: 1-1.5 min

Speaker notes:

End with the balanced claim: the project shows that in-context TabPFN can be effective, but the most scientifically important result is that physically meaningful structure-aware features are necessary for structure-property prediction.

## Appendix Figures

These figures are copied for backup or appendix slides. They are useful for Q&A, but too dense for the main 15-20 min presentation.

### Appendix Figure A - Full Model MAE Comparison

![Full model MAE comparison](presentation_assets/appendix_01_model_mae_comparison.png)

### Appendix Figure B - Fold-Level MAE Dispersion

![Fold-level MAE dispersion](presentation_assets/appendix_02_fold_mae_distribution.png)

### Appendix Figure C - Parity Plots

![Parity plots](presentation_assets/appendix_04_tabpfn_parity.png)

## Final Deck Construction Notes

Recommended main-deck visuals:

- Use Slide 8 as a compact quantitative table.
- Use Slide 9 for internal TabPFN-vs-baseline comparison.
- Use Slide 10 as the central scientific result.
- Keep full model comparison, fold dispersion, and parity plots in appendix only.

Recommended final claim:

This project provides a staged reproduction of a TabPFN/ICL-FM-style Matbench workflow and extends it with lightweight structure-aware descriptors. The project does not fully reproduce the paper's ALIGNN/CGCNN embedding pipeline, but it shows that structure-aware featurization substantially improves TabPFN performance on structure-dependent materials properties and brings results close to published structure-aware benchmarks.
