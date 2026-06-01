"""Model builders used by the notebooks and scripts."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tabpfn import TabPFNRegressor

from .settings import RANDOM_SEED


def build_model(
    model_name: str,
    *,
    random_seed: int = RANDOM_SEED,
    n_estimators: int = 500,
    tabpfn_n_estimators: int = 8,
    tabpfn_device: str = "cuda",
    tabpfn_kwargs: dict[str, Any] | None = None,
) -> Pipeline:
    """Build one model with consistent preprocessing."""

    if model_name == "dummy_mean":
        steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", DummyRegressor(strategy="mean")),
        ]
    elif model_name == "ridge_cv":
        steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", RidgeCV(alphas=np.logspace(-6, 6, 25))),
        ]
    elif model_name == "random_forest":
        steps = [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=n_estimators,
                    random_state=random_seed,
                    n_jobs=-1,
                ),
            ),
        ]
    elif model_name == "extra_trees":
        steps = [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=n_estimators,
                    random_state=random_seed,
                    n_jobs=-1,
                ),
            ),
        ]
    elif model_name == "hist_gradient_boosting":
        steps = [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingRegressor(
                    max_iter=500,
                    learning_rate=0.04,
                    l2_regularization=0.01,
                    early_stopping=True,
                    random_state=random_seed,
                ),
            ),
        ]
    elif model_name == "tabpfn":
        kwargs = {
            "n_estimators": tabpfn_n_estimators,
            "random_state": random_seed,
            "device": tabpfn_device,
            "inference_precision": "auto",
            "memory_saving_mode": "auto",
            "show_progress_bar": False,
        }
        if tabpfn_kwargs:
            kwargs.update(tabpfn_kwargs)
        steps = [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", TabPFNRegressor(**kwargs)),
        ]
    else:
        raise ValueError(f"Unsupported model: {model_name!r}.")

    return Pipeline(steps)


def predict_in_batches(
    model: Pipeline,
    X_test: pd.DataFrame,
    batch_size: int | None,
) -> np.ndarray:
    """Predict with optional final-estimator batching for GPU memory control."""

    if batch_size is None or batch_size <= 0 or batch_size >= len(X_test):
        return model.predict(X_test)

    preprocessor = model[:-1]
    regressor = model.steps[-1][1]
    X_test_preprocessed = preprocessor.transform(X_test)

    predictions = []
    for start in range(0, len(X_test_preprocessed), batch_size):
        stop = start + batch_size
        predictions.append(regressor.predict(X_test_preprocessed[start:stop]))
    return np.concatenate(predictions)
