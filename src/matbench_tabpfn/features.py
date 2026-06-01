"""Matbench loading and feature generation."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from matbench.bench import MatbenchBenchmark
from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.structure import (
    DensityFeatures,
    GlobalSymmetryFeatures,
    MaximumPackingEfficiency,
    StructuralComplexity,
)
from pymatgen.core import Composition


FEATURE_SET_DISPLAY_NAMES = {
    "magpie": "Magpie composition",
    "magpie_density": "Magpie + density",
    "magpie_symmetry": "Magpie + symmetry",
    "magpie_packing": "Magpie + packing",
    "magpie_density_packing": "Magpie + density + packing",
    "magpie_density_packing_symmetry": (
        "Magpie + density + packing + symmetry"
    ),
    "magpie_complexity": "Magpie + complexity",
    "magpie_density_packing_complexity": (
        "Magpie + density + packing + complexity"
    ),
    "magpie_structure_all": "Magpie + all structure",
    "magpie_structure": "Magpie + all structure",
}

STRUCTURE_DESCRIPTOR_GROUPS: dict[str, Callable[[], Any]] = {
    "density": DensityFeatures,
    "symmetry": GlobalSymmetryFeatures,
    "packing": MaximumPackingEfficiency,
    "complexity": StructuralComplexity,
}

STRUCTURE_FEATURE_SET_GROUPS = {
    "magpie_density": ("density",),
    "magpie_symmetry": ("symmetry",),
    "magpie_packing": ("packing",),
    "magpie_density_packing": ("density", "packing"),
    "magpie_density_packing_symmetry": ("density", "packing", "symmetry"),
    "magpie_complexity": ("complexity",),
    "magpie_density_packing_complexity": ("density", "packing", "complexity"),
    "magpie_structure_all": tuple(STRUCTURE_DESCRIPTOR_GROUPS),
    # Backward-compatible alias for earlier runs and notebook outputs.
    "magpie_structure": tuple(STRUCTURE_DESCRIPTOR_GROUPS),
}

STRUCTURE_FEATURE_SETS = frozenset(STRUCTURE_FEATURE_SET_GROUPS)


def feature_set_requires_structure(feature_set: str) -> bool:
    """Return whether a feature set needs raw structure inputs."""

    return feature_set in STRUCTURE_FEATURE_SETS


def load_matbench_task(task_name: str):
    benchmark = MatbenchBenchmark(autoload=False, subset=[task_name])
    task = list(benchmark.tasks)[0]
    task.load()

    if task.metadata["task_type"] != "regression":
        raise ValueError(f"Expected regression task, got {task.metadata['task_type']!r}.")
    if task.metadata["input_type"] not in {"composition", "structure"}:
        raise ValueError(f"Unsupported input type: {task.metadata['input_type']!r}.")

    return task


def to_composition(value) -> Composition:
    if isinstance(value, Composition):
        return value
    if hasattr(value, "composition"):
        return value.composition
    return Composition(value)


def get_composition_inputs(task) -> tuple[pd.Series, str]:
    """Return compositions for composition tasks and composition proxies for structures."""

    input_type = task.metadata["input_type"]
    if input_type == "composition":
        return task.df[input_type].map(to_composition), "composition"
    if input_type == "structure":
        return task.df[input_type].map(lambda structure: structure.composition), (
            "composition_from_structure"
        )
    raise ValueError(f"Unsupported input type: {input_type!r}.")


def get_structure_inputs(task) -> tuple[pd.Series, str]:
    input_type = task.metadata["input_type"]
    if input_type != "structure":
        raise ValueError(
            f"Feature set requires structure input, got {input_type!r} for task."
        )
    return task.df[input_type], "structure"


def _clean_feature_frame(features: pd.DataFrame) -> pd.DataFrame:
    features = features.copy()
    drop_columns = [
        col
        for col in features.columns
        if col in {"composition", "structure"} or str(col).endswith(" Exceptions")
    ]
    if drop_columns:
        features = features.drop(columns=drop_columns)

    for col in features.select_dtypes(include=["bool"]).columns:
        features[col] = features[col].astype(int)

    categorical = features.select_dtypes(include=["object", "category"]).columns
    if len(categorical):
        features = pd.get_dummies(features, columns=list(categorical), dummy_na=True)

    features = features.apply(pd.to_numeric, errors="coerce")
    features = features.replace([np.inf, -np.inf], np.nan)
    return features.dropna(axis=1, how="all")


def featurize_magpie(
    task_name: str,
    compositions: pd.Series,
    feature_dir: Path | str,
    *,
    use_cache: bool = True,
    show_progress: bool = True,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Build Magpie composition features with a run-local cache."""

    feature_dir = Path(feature_dir)
    feature_dir.mkdir(parents=True, exist_ok=True)
    cache_path = feature_dir / f"{task_name}_magpie_features.csv"

    if use_cache and cache_path.exists():
        features = pd.read_csv(cache_path, index_col=0)
        if len(features) != len(compositions):
            raise ValueError(
                f"Cached feature length mismatch for {task_name}: "
                f"{len(features)} != {len(compositions)}."
            )
        features.index = compositions.index
        return features

    feature_input = pd.DataFrame({"composition": compositions.map(to_composition)})
    feature_input.index = compositions.index

    featurizer = ElementProperty.from_preset("magpie")
    featurizer.set_n_jobs(n_jobs)
    features = featurizer.featurize_dataframe(
        feature_input,
        col_id="composition",
        ignore_errors=False,
        inplace=False,
        pbar=show_progress,
    )
    features = _clean_feature_frame(features)
    features.to_csv(cache_path)
    return features


def _normalize_structure_groups(
    descriptor_groups: Sequence[str] | None = None,
) -> tuple[str, ...]:
    groups = (
        tuple(STRUCTURE_DESCRIPTOR_GROUPS)
        if descriptor_groups is None
        else tuple(descriptor_groups)
    )
    unknown = [group for group in groups if group not in STRUCTURE_DESCRIPTOR_GROUPS]
    if unknown:
        raise ValueError(f"Unsupported structure descriptor group(s): {unknown!r}.")
    return groups


def featurize_structure_descriptor_group(
    task_name: str,
    structures: pd.Series,
    descriptor_group: str,
    feature_dir: Path | str,
    *,
    use_cache: bool = True,
    show_progress: bool = True,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Build one cached group of lightweight structure descriptors."""

    feature_dir = Path(feature_dir)
    feature_dir.mkdir(parents=True, exist_ok=True)
    if descriptor_group not in STRUCTURE_DESCRIPTOR_GROUPS:
        raise ValueError(
            f"Unsupported structure descriptor group: {descriptor_group!r}."
        )
    cache_path = feature_dir / f"{task_name}_structure_{descriptor_group}_features.csv"

    if use_cache and cache_path.exists():
        features = pd.read_csv(cache_path, index_col=0)
        if len(features) != len(structures):
            raise ValueError(
                f"Cached {descriptor_group} feature length mismatch for {task_name}: "
                f"{len(features)} != {len(structures)}."
            )
        features.index = structures.index
        return features

    featurizer = STRUCTURE_DESCRIPTOR_GROUPS[descriptor_group]()
    featurizer.set_n_jobs(n_jobs)
    input_df = pd.DataFrame({"structure": structures}, index=structures.index)
    features = featurizer.featurize_dataframe(
        input_df,
        col_id="structure",
        ignore_errors=True,
        return_errors=True,
        inplace=False,
        pbar=show_progress,
    )
    features = _clean_feature_frame(features)
    features = features.rename(columns=lambda col: f"{descriptor_group}|{col}")
    features.to_csv(cache_path)
    return features


def featurize_structure_descriptors(
    task_name: str,
    structures: pd.Series,
    feature_dir: Path | str,
    *,
    descriptor_groups: Sequence[str] | None = None,
    use_cache: bool = True,
    show_progress: bool = True,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """Build selected structure descriptor groups for Matbench structure tasks."""

    groups = _normalize_structure_groups(descriptor_groups)
    frames = [
        featurize_structure_descriptor_group(
            task_name,
            structures,
            group,
            feature_dir,
            use_cache=use_cache,
            show_progress=show_progress,
            n_jobs=n_jobs,
        )
        for group in groups
    ]
    if not frames:
        return pd.DataFrame(index=structures.index)

    features = pd.concat(frames, axis=1)
    features = _clean_feature_frame(features)
    return features


def build_task_features(
    task_name: str,
    task,
    feature_set: str,
    feature_dir: Path | str,
    *,
    use_cache: bool = True,
    show_progress: bool = True,
    n_jobs: int = 1,
) -> tuple[pd.DataFrame, str, str]:
    """Return features, source label, and display label for one task/feature set."""

    if feature_set == "magpie":
        compositions, feature_source = get_composition_inputs(task)
        features = featurize_magpie(
            task_name,
            compositions,
            feature_dir,
            use_cache=use_cache,
            show_progress=show_progress,
            n_jobs=n_jobs,
        )
        return features, feature_source, FEATURE_SET_DISPLAY_NAMES[feature_set]

    if feature_set in STRUCTURE_FEATURE_SET_GROUPS:
        structures, feature_source = get_structure_inputs(task)
        compositions = structures.map(lambda structure: structure.composition)
        composition_features = featurize_magpie(
            task_name,
            compositions,
            feature_dir,
            use_cache=use_cache,
            show_progress=show_progress,
            n_jobs=n_jobs,
        ).add_prefix("Magpie|")
        structure_features = featurize_structure_descriptors(
            task_name,
            structures,
            feature_dir,
            descriptor_groups=STRUCTURE_FEATURE_SET_GROUPS[feature_set],
            use_cache=use_cache,
            show_progress=show_progress,
            n_jobs=n_jobs,
        )
        features = pd.concat([composition_features, structure_features], axis=1)
        features = _clean_feature_frame(features)
        return features, feature_source, FEATURE_SET_DISPLAY_NAMES[feature_set]

    raise ValueError(f"Unsupported feature set: {feature_set!r}.")
