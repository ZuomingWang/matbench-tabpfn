# Automatic Results Summary

## Best Model by Task

- `matbench_expt_gap`: TabPFN with Magpie composition features, MAE=0.3688 eV, R2=0.712.
- `matbench_jdft2d`: TabPFN with Magpie + all structure features, MAE=34.4 meV/atom, R2=0.41.
- `matbench_phonons`: TabPFN with Magpie + all structure features, MAE=30.63 cm^-1, R2=0.978.
- `matbench_steels`: TabPFN + ExtraTrees 50/50 with Magpie composition features, MAE=87 MPa, R2=0.774.

## TabPFN Claim Check

- `matbench_expt_gap` (Magpie composition): TabPFN improves over the best non-TabPFN baseline (Extra trees, Magpie composition) by -9.68% MAE.
- `matbench_jdft2d` (Magpie composition): TabPFN does not beat the best non-TabPFN baseline (Extra trees, Magpie + all structure) by +13.78% MAE.
- `matbench_jdft2d` (Magpie + complexity): TabPFN does not beat the best non-TabPFN baseline (Extra trees, Magpie + all structure) by +6.15% MAE.
- `matbench_jdft2d` (Magpie + density): TabPFN improves over the best non-TabPFN baseline (Extra trees, Magpie + all structure) by -11.19% MAE.
- `matbench_jdft2d` (Magpie + density + packing): TabPFN improves over the best non-TabPFN baseline (Extra trees, Magpie + all structure) by -14.11% MAE.
- `matbench_jdft2d` (Magpie + density + packing + complexity): TabPFN improves over the best non-TabPFN baseline (Extra trees, Magpie + all structure) by -13.44% MAE.
- `matbench_jdft2d` (Magpie + density + packing + symmetry): TabPFN improves over the best non-TabPFN baseline (Extra trees, Magpie + all structure) by -13.76% MAE.
- `matbench_jdft2d` (Magpie + packing): TabPFN improves over the best non-TabPFN baseline (Extra trees, Magpie + all structure) by -9.49% MAE.
- `matbench_jdft2d` (Magpie + all structure): TabPFN improves over the best non-TabPFN baseline (Extra trees, Magpie + all structure) by -15.16% MAE.
- `matbench_jdft2d` (Magpie + symmetry): TabPFN does not beat the best non-TabPFN baseline (Extra trees, Magpie + all structure) by +8.00% MAE.
- `matbench_phonons` (Magpie composition): TabPFN improves over the best non-TabPFN baseline (Extra trees, Magpie + density + packing) by -16.67% MAE.
- `matbench_phonons` (Magpie + complexity): TabPFN improves over the best non-TabPFN baseline (Extra trees, Magpie + density + packing) by -19.94% MAE.
- `matbench_phonons` (Magpie + density): TabPFN improves over the best non-TabPFN baseline (Extra trees, Magpie + density + packing) by -27.05% MAE.
- `matbench_phonons` (Magpie + density + packing): TabPFN improves over the best non-TabPFN baseline (Extra trees, Magpie + density + packing) by -29.62% MAE.
- `matbench_phonons` (Magpie + density + packing + complexity): TabPFN improves over the best non-TabPFN baseline (Extra trees, Magpie + density + packing) by -29.94% MAE.
- `matbench_phonons` (Magpie + density + packing + symmetry): TabPFN improves over the best non-TabPFN baseline (Extra trees, Magpie + density + packing) by -29.66% MAE.
- `matbench_phonons` (Magpie + packing): TabPFN improves over the best non-TabPFN baseline (Extra trees, Magpie + density + packing) by -28.46% MAE.
- `matbench_phonons` (Magpie + all structure): TabPFN improves over the best non-TabPFN baseline (Extra trees, Magpie + density + packing) by -30.74% MAE.
- `matbench_phonons` (Magpie + symmetry): TabPFN improves over the best non-TabPFN baseline (Extra trees, Magpie + density + packing) by -21.70% MAE.
- `matbench_steels` (Magpie composition): TabPFN improves over the best non-TabPFN baseline (Extra trees, Magpie composition) by -3.44% MAE.

## Structure-Aware Feature Check

- `matbench_jdft2d`: best structure-aware branch (Magpie + all structure) improves over the composition-proxy branch by -25.44% MAE.
- `matbench_phonons`: best structure-aware branch (Magpie + all structure) improves over the composition-proxy branch by -16.88% MAE.

## Fixed Ensemble Check

- `matbench_expt_gap` (Magpie composition): the 50/50 TabPFN + ExtraTrees ensemble does not improve over standalone TabPFN by +4.41% MAE.
- `matbench_jdft2d` (Magpie + all structure): the 50/50 TabPFN + ExtraTrees ensemble does not improve over standalone TabPFN by +5.71% MAE.
- `matbench_phonons` (Magpie + density + packing + complexity): the 50/50 TabPFN + ExtraTrees ensemble does not improve over standalone TabPFN by +13.88% MAE.
- `matbench_steels` (Magpie composition): the 50/50 TabPFN + ExtraTrees ensemble improves over standalone TabPFN by -0.74% MAE.

## Weight Scan Diagnostic

- `matbench_expt_gap` (Magpie composition): diagnostic test-fold scan selects TabPFN weight 1.00 and would not improve over standalone TabPFN by +0.00% MAE. Do not report this as a final tuned model without inner-validation weight selection.
- `matbench_jdft2d` (Magpie + all structure): diagnostic test-fold scan selects TabPFN weight 1.00 and would not improve over standalone TabPFN by +0.00% MAE. Do not report this as a final tuned model without inner-validation weight selection.
- `matbench_phonons` (Magpie + all structure): diagnostic test-fold scan selects TabPFN weight 0.95 and would improve over standalone TabPFN by -0.13% MAE. Do not report this as a final tuned model without inner-validation weight selection.
- `matbench_steels` (Magpie composition): diagnostic test-fold scan selects TabPFN weight 0.60 and would improve over standalone TabPFN by -0.97% MAE. Do not report this as a final tuned model without inner-validation weight selection.

## Inner-Validation Tuned Ensemble Check

- `matbench_phonons` (Magpie + all structure): inner-validation tuned TabPFN + ExtraTrees does not improve over standalone TabPFN by +0.00% MAE.
- `matbench_steels` (Magpie composition): inner-validation tuned TabPFN + ExtraTrees does not improve over standalone TabPFN by +2.65% MAE.
