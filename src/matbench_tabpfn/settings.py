"""Shared project settings."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

STARTER_TASKS = [
    "matbench_steels",
    "matbench_jdft2d",
    "matbench_phonons",
    "matbench_expt_gap",
]

DEFAULT_MODELS = [
    "dummy_mean",
    "ridge_cv",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
    "tabpfn",
    "tabpfn_extra_trees_ensemble",
    "tabpfn_extra_trees_inner_tuned",
]

MODEL_DISPLAY_NAMES = {
    "dummy_mean": "Dummy mean",
    "ridge_cv": "RidgeCV",
    "random_forest": "Random forest",
    "extra_trees": "Extra trees",
    "hist_gradient_boosting": "HistGradientBoosting",
    "tabpfn": "TabPFN",
    "tabpfn_extra_trees_ensemble": "TabPFN + ExtraTrees 50/50",
    "tabpfn_extra_trees_inner_tuned": "TabPFN + ExtraTrees tuned",
}

MODEL_COLORS = {
    "dummy_mean": "#8a8f98",
    "ridge_cv": "#4c78a8",
    "random_forest": "#59a14f",
    "extra_trees": "#f28e2b",
    "hist_gradient_boosting": "#e15759",
    "tabpfn": "#6f4eae",
    "tabpfn_extra_trees_ensemble": "#8c6d31",
    "tabpfn_extra_trees_inner_tuned": "#b07aa1",
}

TABPFN_FAMILY_MODELS = {
    "tabpfn",
    "tabpfn_extra_trees_ensemble",
    "tabpfn_extra_trees_inner_tuned",
}

RANDOM_SEED = 42
PRIMARY_METRIC = "mae"
