"""Retrospective active-learning screening diagnostics for Matbench tasks.

This is an ICAL-inspired extension, not a full reproduction of ICAL. It keeps
the project on the original Matbench data while asking a different question:
given a small labeled set and a remaining candidate pool, which surrogate
strategy finds high-target materials fastest?

The script is intentionally standalone. It reuses the project's existing
feature and model helpers, but does not modify the main notebook, source
modules, or benchmark result tables. Outputs go under:

    results/active_learning_screening/<run_id>/

Smoke test:

    python scripts/run_active_learning_screening.py \
        --tasks matbench_jdft2d \
        --feature-sets magpie \
        --strategies random extra_trees_greedy \
        --folds 0 \
        --initial-fractions 0.1 \
        --max-acquisitions 20 \
        --acquisition-batch-size 10 \
        --repeats 1 \
        --device cpu

Fuller GPU run:

    python scripts/run_active_learning_screening.py \
        --strategies random extra_trees_greedy extra_trees_ucb tabpfn_greedy tabpfn_disagreement_ucb \
        --initial-fractions 0.1 0.2 \
        --max-acquisitions 100 \
        --acquisition-batch-size 10 \
        --repeats 3 \
        --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


DEFAULT_TASKS = ("matbench_jdft2d", "matbench_phonons")
DEFAULT_FEATURE_SETS = ("magpie", "magpie_structure_all")
DEFAULT_STRATEGIES = (
    "random",
    "extra_trees_greedy",
    "extra_trees_ucb",
    "tabpfn_greedy",
    "tabpfn_disagreement_ucb",
)
MODEL_STRATEGIES = {
    "extra_trees_greedy",
    "extra_trees_ucb",
    "tabpfn_greedy",
    "tabpfn_disagreement_ucb",
}
DEFAULT_RANDOM_SEED = 42


@dataclass(frozen=True)
class ActiveLearningConfig:
    tasks: tuple[str, ...]
    feature_sets: tuple[str, ...]
    strategies: tuple[str, ...]
    initial_fractions: tuple[float, ...]
    folds: tuple[int, ...] | None
    repeats: int
    random_seed: int
    objective: str
    initial_selection: str
    acquisition_batch_size: int
    max_acquisitions: int | None
    max_acquisition_fraction: float | None
    top_fraction: float
    ucb_beta: float
    n_estimators: int
    tabpfn_n_estimators: int
    device: str
    tabpfn_predict_batch_size: int | None
    feature_n_jobs: int
    use_feature_cache: bool
    show_feature_progress: bool
    include_oracle: bool


def _parse_args() -> ActiveLearningConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Run retrospective active-learning screening on official Matbench "
            "training folds without touching the main benchmark outputs."
        )
    )
    parser.add_argument("--tasks", nargs="+", default=list(DEFAULT_TASKS))
    parser.add_argument("--feature-sets", nargs="+", default=list(DEFAULT_FEATURE_SETS))
    parser.add_argument("--strategies", nargs="+", default=list(DEFAULT_STRATEGIES))
    parser.add_argument(
        "--initial-fractions",
        nargs="+",
        type=float,
        default=[0.1, 0.2],
        help="Initial labeled fractions of each official train fold.",
    )
    parser.add_argument(
        "--folds",
        nargs="+",
        type=int,
        default=None,
        help="Optional subset of official Matbench fold numbers.",
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--random-seed", type=int, default=DEFAULT_RANDOM_SEED)
    parser.add_argument(
        "--objective",
        choices=["maximize", "minimize"],
        default="maximize",
        help="Whether high or low target values are considered the discovery goal.",
    )
    parser.add_argument(
        "--initial-selection",
        choices=["random", "stratified", "low", "high"],
        default="random",
        help=(
            "How to choose the initial labeled set. random is the most realistic; "
            "low/high can reproduce curated-start sensitivity checks."
        ),
    )
    parser.add_argument("--acquisition-batch-size", type=int, default=10)
    parser.add_argument(
        "--max-acquisitions",
        type=int,
        default=100,
        help="Maximum number of acquired labels after initialization.",
    )
    parser.add_argument(
        "--max-acquisition-fraction",
        type=float,
        default=None,
        help=(
            "Optional budget as a fraction of the candidate pool. If provided, "
            "the smaller of this budget and --max-acquisitions is used."
        ),
    )
    parser.add_argument(
        "--top-fraction",
        type=float,
        default=0.05,
        help="Fraction of each official train fold treated as top candidates.",
    )
    parser.add_argument(
        "--ucb-beta",
        type=float,
        default=0.5,
        help="Exploration weight for UCB-style acquisition proxies.",
    )
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--tabpfn-n-estimators", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--tabpfn-predict-batch-size", type=int, default=128)
    parser.add_argument("--feature-n-jobs", type=int, default=1)
    parser.add_argument("--no-feature-cache", action="store_true")
    parser.add_argument("--show-feature-progress", action="store_true")
    parser.add_argument(
        "--include-oracle",
        action="store_true",
        help="Include an oracle upper-bound strategy that knows true labels.",
    )
    args = parser.parse_args()

    initial_fractions = tuple(sorted(set(float(value) for value in args.initial_fractions)))
    if not initial_fractions or any(value <= 0 or value >= 1 for value in initial_fractions):
        raise ValueError("All --initial-fractions must be in the interval (0, 1).")
    if args.repeats < 1:
        raise ValueError("--repeats must be at least 1.")
    if args.acquisition_batch_size < 1:
        raise ValueError("--acquisition-batch-size must be at least 1.")
    if args.max_acquisitions is not None and args.max_acquisitions < 1:
        raise ValueError("--max-acquisitions must be positive when provided.")
    if args.max_acquisition_fraction is not None and not (0 < args.max_acquisition_fraction <= 1):
        raise ValueError("--max-acquisition-fraction must be in the interval (0, 1].")
    if not (0 < args.top_fraction < 1):
        raise ValueError("--top-fraction must be in the interval (0, 1).")

    valid_strategies = set(DEFAULT_STRATEGIES) | {"oracle"}
    unknown = sorted(set(args.strategies).difference(valid_strategies))
    if unknown:
        raise ValueError(f"Unsupported strategy or strategies: {unknown}")

    strategies = list(args.strategies)
    if args.include_oracle and "oracle" not in strategies:
        strategies.append("oracle")

    return ActiveLearningConfig(
        tasks=tuple(args.tasks),
        feature_sets=tuple(args.feature_sets),
        strategies=tuple(strategies),
        initial_fractions=initial_fractions,
        folds=tuple(args.folds) if args.folds is not None else None,
        repeats=args.repeats,
        random_seed=args.random_seed,
        objective=args.objective,
        initial_selection=args.initial_selection,
        acquisition_batch_size=args.acquisition_batch_size,
        max_acquisitions=args.max_acquisitions,
        max_acquisition_fraction=args.max_acquisition_fraction,
        top_fraction=args.top_fraction,
        ucb_beta=args.ucb_beta,
        n_estimators=args.n_estimators,
        tabpfn_n_estimators=args.tabpfn_n_estimators,
        device=args.device,
        tabpfn_predict_batch_size=args.tabpfn_predict_batch_size,
        feature_n_jobs=args.feature_n_jobs,
        use_feature_cache=not args.no_feature_cache,
        show_feature_progress=args.show_feature_progress,
        include_oracle=args.include_oracle,
    )


def _iter_requested_folds(task, folds: Iterable[int] | None) -> list[int]:
    official = list(task.folds_nums)
    if folds is None:
        return official
    requested = list(folds)
    unknown = sorted(set(requested).difference(official))
    if unknown:
        raise ValueError(f"Requested folds are not official Matbench folds: {unknown}")
    return requested


def _objective_values(y: pd.Series | np.ndarray, objective: str) -> np.ndarray:
    values = np.asarray(y, dtype=float)
    if objective == "maximize":
        return values
    if objective == "minimize":
        return -values
    raise ValueError(f"Unsupported objective: {objective!r}")


def _target_from_objective(objective_values: np.ndarray, objective: str) -> np.ndarray:
    if objective == "maximize":
        return objective_values
    if objective == "minimize":
        return -objective_values
    raise ValueError(f"Unsupported objective: {objective!r}")


def _normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros_like(values, dtype=float)
    clean = values.copy()
    fill = np.nanmedian(clean[finite])
    clean[~finite] = fill
    lo = np.min(clean)
    hi = np.max(clean)
    if math.isclose(float(lo), float(hi)):
        return np.zeros_like(clean, dtype=float)
    return (clean - lo) / (hi - lo)


def _stable_seed_offset(*parts: object, modulus: int = 9_973) -> int:
    text = "|".join(str(part) for part in parts)
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) % modulus


def _select_initial_indices(
    y_train: pd.Series,
    *,
    fraction: float,
    objective: str,
    method: str,
    rng: np.random.Generator,
) -> pd.Index:
    n_initial = max(1, int(round(len(y_train) * fraction)))
    n_initial = min(n_initial, len(y_train) - 1)
    objective_y = pd.Series(_objective_values(y_train, objective), index=y_train.index)

    if method == "random":
        selected = rng.choice(y_train.index.to_numpy(), size=n_initial, replace=False)
        return pd.Index(selected)

    if method == "high":
        return pd.Index(objective_y.nlargest(n_initial).index)

    if method == "low":
        return pd.Index(objective_y.nsmallest(n_initial).index)

    if method == "stratified":
        n_bins = min(5, max(1, y_train.nunique()), len(y_train))
        try:
            bins = pd.qcut(objective_y, q=n_bins, labels=False, duplicates="drop")
        except ValueError:
            selected = rng.choice(y_train.index.to_numpy(), size=n_initial, replace=False)
            return pd.Index(selected)

        selected: list[object] = []
        grouped = pd.Series(y_train.index, index=y_train.index).groupby(bins)
        for _, group in grouped:
            group_values = group.to_numpy()
            take = max(1, int(round(n_initial * len(group_values) / len(y_train))))
            take = min(take, len(group_values))
            selected.extend(rng.choice(group_values, size=take, replace=False).tolist())

        if len(selected) > n_initial:
            selected = rng.choice(np.asarray(selected, dtype=object), size=n_initial, replace=False).tolist()
        elif len(selected) < n_initial:
            remaining = y_train.index.difference(pd.Index(selected)).to_numpy()
            add = rng.choice(remaining, size=n_initial - len(selected), replace=False)
            selected.extend(add.tolist())
        return pd.Index(selected)

    raise ValueError(f"Unsupported initial selection method: {method!r}")


def _top_candidate_sets(
    y_train: pd.Series,
    *,
    objective: str,
    top_fraction: float,
) -> tuple[set[object], object, float, float]:
    objective_y = pd.Series(_objective_values(y_train, objective), index=y_train.index)
    n_top = max(1, int(math.ceil(len(y_train) * top_fraction)))
    top_index = set(objective_y.nlargest(n_top).index)
    best_idx = objective_y.idxmax()
    best_objective = float(objective_y.loc[best_idx])
    best_target = float(y_train.loc[best_idx])
    return top_index, best_idx, best_objective, best_target


def _tree_prediction_std(model, X_pool: pd.DataFrame) -> np.ndarray:
    """Return per-candidate tree prediction std for a fitted sklearn pipeline."""

    preprocessor = model[:-1]
    regressor = model.steps[-1][1]
    X_preprocessed = preprocessor.transform(X_pool)
    if not hasattr(regressor, "estimators_"):
        return np.zeros(len(X_pool), dtype=float)
    per_tree = np.vstack([tree.predict(X_preprocessed) for tree in regressor.estimators_])
    return per_tree.std(axis=0)


def _predict_strategy_scores(
    strategy: str,
    X_labeled: pd.DataFrame,
    y_labeled: pd.Series,
    X_pool: pd.DataFrame,
    *,
    objective: str,
    random_seed: int,
    n_estimators: int,
    tabpfn_n_estimators: int,
    device: str,
    tabpfn_predict_batch_size: int | None,
    ucb_beta: float,
) -> tuple[np.ndarray, dict[str, float | str]]:
    from matbench_tabpfn.models import build_model, predict_in_batches

    fit_seconds = 0.0
    predict_seconds = 0.0
    metadata: dict[str, float | str] = {}

    if strategy.startswith("extra_trees"):
        model = build_model(
            "extra_trees",
            random_seed=random_seed,
            n_estimators=n_estimators,
        )
        fit_start = time.time()
        model.fit(X_labeled, y_labeled)
        fit_seconds = time.time() - fit_start

        predict_start = time.time()
        y_pred = predict_in_batches(model, X_pool, None)
        predict_seconds = time.time() - predict_start
        pred_objective = _objective_values(y_pred, objective)

        if strategy == "extra_trees_greedy":
            scores = pred_objective
            metadata["uncertainty_proxy"] = "none"
        elif strategy == "extra_trees_ucb":
            uncertainty = _tree_prediction_std(model, X_pool)
            scores = _normalize(pred_objective) + ucb_beta * _normalize(uncertainty)
            metadata["uncertainty_proxy"] = "tree_prediction_std"
            metadata["mean_uncertainty_proxy"] = float(np.mean(uncertainty))
        else:
            raise ValueError(f"Unsupported ExtraTrees strategy: {strategy!r}")

        metadata["fit_seconds"] = fit_seconds
        metadata["predict_seconds"] = predict_seconds
        return scores, metadata

    if strategy.startswith("tabpfn"):
        tabpfn_model = build_model(
            "tabpfn",
            random_seed=random_seed,
            n_estimators=n_estimators,
            tabpfn_n_estimators=tabpfn_n_estimators,
            tabpfn_device=device,
        )
        fit_start = time.time()
        tabpfn_model.fit(X_labeled, y_labeled)
        fit_seconds = time.time() - fit_start

        predict_start = time.time()
        tabpfn_pred = predict_in_batches(tabpfn_model, X_pool, tabpfn_predict_batch_size)
        predict_seconds = time.time() - predict_start
        tabpfn_objective = _objective_values(tabpfn_pred, objective)

        if strategy == "tabpfn_greedy":
            scores = tabpfn_objective
            metadata["uncertainty_proxy"] = "none"
        elif strategy == "tabpfn_disagreement_ucb":
            extra_trees_model = build_model(
                "extra_trees",
                random_seed=random_seed,
                n_estimators=n_estimators,
            )
            et_fit_start = time.time()
            extra_trees_model.fit(X_labeled, y_labeled)
            fit_seconds += time.time() - et_fit_start

            et_predict_start = time.time()
            et_pred = predict_in_batches(extra_trees_model, X_pool, None)
            predict_seconds += time.time() - et_predict_start

            disagreement = np.abs(tabpfn_pred - et_pred)
            scores = _normalize(tabpfn_objective) + ucb_beta * _normalize(disagreement)
            metadata["uncertainty_proxy"] = "tabpfn_extra_trees_disagreement"
            metadata["mean_uncertainty_proxy"] = float(np.mean(disagreement))
        else:
            raise ValueError(f"Unsupported TabPFN strategy: {strategy!r}")

        metadata["fit_seconds"] = fit_seconds
        metadata["predict_seconds"] = predict_seconds
        return scores, metadata

    raise ValueError(f"Unsupported model strategy: {strategy!r}")


def _choose_acquisition(
    strategy: str,
    X_labeled: pd.DataFrame,
    y_labeled: pd.Series,
    X_pool: pd.DataFrame,
    y_pool: pd.Series,
    *,
    objective: str,
    batch_size: int,
    rng: np.random.Generator,
    random_seed: int,
    n_estimators: int,
    tabpfn_n_estimators: int,
    device: str,
    tabpfn_predict_batch_size: int | None,
    ucb_beta: float,
) -> tuple[pd.Index, dict[str, float | str]]:
    batch_size = min(batch_size, len(X_pool))

    if strategy == "random":
        selected = rng.choice(X_pool.index.to_numpy(), size=batch_size, replace=False)
        return pd.Index(selected), {
            "fit_seconds": 0.0,
            "predict_seconds": 0.0,
            "uncertainty_proxy": "none",
        }

    if strategy == "oracle":
        objective_pool = pd.Series(_objective_values(y_pool, objective), index=y_pool.index)
        return pd.Index(objective_pool.nlargest(batch_size).index), {
            "fit_seconds": 0.0,
            "predict_seconds": 0.0,
            "uncertainty_proxy": "oracle_true_label",
        }

    scores, metadata = _predict_strategy_scores(
        strategy,
        X_labeled,
        y_labeled,
        X_pool,
        objective=objective,
        random_seed=random_seed,
        n_estimators=n_estimators,
        tabpfn_n_estimators=tabpfn_n_estimators,
        device=device,
        tabpfn_predict_batch_size=tabpfn_predict_batch_size,
        ucb_beta=ucb_beta,
    )
    score_series = pd.Series(scores, index=X_pool.index)
    selected = score_series.sort_values(ascending=False).head(batch_size).index
    return pd.Index(selected), metadata


def _trace_state(
    *,
    task_name: str,
    target_col: str,
    unit: str | None,
    input_type: str,
    feature_source: str,
    feature_set: str,
    feature_set_display: str,
    strategy: str,
    fold: int,
    repeat: int,
    initial_fraction: float,
    initial_size: int,
    acquired_count: int,
    cycle: int,
    y_train: pd.Series,
    labeled_idx: pd.Index,
    top_idx: set[object],
    best_idx: object,
    global_best_objective: float,
    global_best_target: float,
    objective: str,
    selected_this_cycle: int,
    fit_seconds: float,
    predict_seconds: float,
    uncertainty_proxy: str,
    mean_uncertainty_proxy: float | None,
) -> dict:
    y_labeled = y_train.loc[labeled_idx]
    labeled_objective = pd.Series(_objective_values(y_labeled, objective), index=labeled_idx)
    best_found_idx = labeled_objective.idxmax()
    best_found_objective = float(labeled_objective.loc[best_found_idx])
    best_found_target = float(y_train.loc[best_found_idx])
    top_found = len(set(labeled_idx).intersection(top_idx))
    n_top = len(top_idx)
    percentile_rank = float((pd.Series(_objective_values(y_train, objective), index=y_train.index) <= best_found_objective).mean())

    return {
        "task": task_name,
        "target": target_col,
        "unit": unit,
        "matbench_input_type": input_type,
        "feature_source": feature_source,
        "feature_set": feature_set,
        "feature_set_display": feature_set_display,
        "strategy": strategy,
        "fold": fold,
        "repeat": repeat,
        "initial_fraction": initial_fraction,
        "initial_size": initial_size,
        "train_fold_size": len(y_train),
        "candidate_pool_initial_size": len(y_train) - initial_size,
        "cycle": cycle,
        "acquired_count": acquired_count,
        "labeled_size": len(labeled_idx),
        "selected_this_cycle": selected_this_cycle,
        "best_found_index": best_found_idx,
        "best_found_target": best_found_target,
        "best_found_objective": best_found_objective,
        "global_best_index": best_idx,
        "global_best_target": global_best_target,
        "global_best_objective": global_best_objective,
        "objective_gap": global_best_objective - best_found_objective,
        "objective_regret_fraction": (
            (global_best_objective - best_found_objective) / abs(global_best_objective)
            if not math.isclose(global_best_objective, 0.0)
            else np.nan
        ),
        "best_objective_percentile_found": percentile_rank,
        "top_fraction_hit_count": top_found,
        "top_fraction_total": n_top,
        "top_fraction_hit_rate": top_found / n_top,
        "found_global_best": best_idx in set(labeled_idx),
        "found_any_top_fraction": top_found > 0,
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "uncertainty_proxy": uncertainty_proxy,
        "mean_uncertainty_proxy": mean_uncertainty_proxy,
    }


def _simulate_one_run(
    *,
    task_name: str,
    target_col: str,
    unit: str | None,
    input_type: str,
    feature_source: str,
    feature_set: str,
    feature_set_display: str,
    strategy: str,
    fold: int,
    repeat: int,
    initial_fraction: float,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: ActiveLearningConfig,
) -> tuple[list[dict], list[dict]]:
    seed = (
        config.random_seed
        + 100_000 * repeat
        + 10_000 * int(fold)
        + int(round(initial_fraction * 1_000))
        + _stable_seed_offset(task_name, feature_set, strategy)
    )
    rng = np.random.default_rng(seed)

    initial_idx = _select_initial_indices(
        y_train,
        fraction=initial_fraction,
        objective=config.objective,
        method=config.initial_selection,
        rng=rng,
    )
    labeled_idx = pd.Index(initial_idx)
    pool_idx = y_train.index.difference(labeled_idx)
    top_idx, best_idx, global_best_objective, global_best_target = _top_candidate_sets(
        y_train,
        objective=config.objective,
        top_fraction=config.top_fraction,
    )

    fraction_budget = None
    if config.max_acquisition_fraction is not None:
        fraction_budget = int(math.ceil(len(pool_idx) * config.max_acquisition_fraction))
    if config.max_acquisitions is None:
        max_acquisitions = fraction_budget if fraction_budget is not None else len(pool_idx)
    elif fraction_budget is None:
        max_acquisitions = config.max_acquisitions
    else:
        max_acquisitions = min(config.max_acquisitions, fraction_budget)
    max_acquisitions = min(max_acquisitions, len(pool_idx))

    trace_rows: list[dict] = [
        _trace_state(
            task_name=task_name,
            target_col=target_col,
            unit=unit,
            input_type=input_type,
            feature_source=feature_source,
            feature_set=feature_set,
            feature_set_display=feature_set_display,
            strategy=strategy,
            fold=fold,
            repeat=repeat,
            initial_fraction=initial_fraction,
            initial_size=len(labeled_idx),
            acquired_count=0,
            cycle=0,
            y_train=y_train,
            labeled_idx=labeled_idx,
            top_idx=top_idx,
            best_idx=best_idx,
            global_best_objective=global_best_objective,
            global_best_target=global_best_target,
            objective=config.objective,
            selected_this_cycle=0,
            fit_seconds=0.0,
            predict_seconds=0.0,
            uncertainty_proxy="none",
            mean_uncertainty_proxy=None,
        )
    ]
    acquisition_rows: list[dict] = []

    acquired_count = 0
    cycle = 0
    while len(pool_idx) > 0 and acquired_count < max_acquisitions:
        cycle += 1
        batch_size = min(config.acquisition_batch_size, max_acquisitions - acquired_count)
        selected_idx, metadata = _choose_acquisition(
            strategy,
            X_train.loc[labeled_idx],
            y_train.loc[labeled_idx],
            X_train.loc[pool_idx],
            y_train.loc[pool_idx],
            objective=config.objective,
            batch_size=batch_size,
            rng=rng,
            random_seed=seed + cycle,
            n_estimators=config.n_estimators,
            tabpfn_n_estimators=config.tabpfn_n_estimators,
            device=config.device,
            tabpfn_predict_batch_size=config.tabpfn_predict_batch_size,
            ucb_beta=config.ucb_beta,
        )

        selected_y = y_train.loc[selected_idx]
        selected_objective = pd.Series(
            _objective_values(selected_y, config.objective),
            index=selected_idx,
        )
        for idx in selected_idx:
            acquisition_rows.append(
                {
                    "task": task_name,
                    "feature_set": feature_set,
                    "feature_set_display": feature_set_display,
                    "strategy": strategy,
                    "fold": fold,
                    "repeat": repeat,
                    "initial_fraction": initial_fraction,
                    "cycle": cycle,
                    "acquisition_rank_in_cycle": int(list(selected_idx).index(idx) + 1),
                    "acquired_index": idx,
                    "acquired_target": float(y_train.loc[idx]),
                    "acquired_objective": float(selected_objective.loc[idx]),
                    "is_global_best": idx == best_idx,
                    "is_top_fraction": idx in top_idx,
                }
            )

        labeled_idx = labeled_idx.append(selected_idx)
        pool_idx = pool_idx.difference(selected_idx)
        acquired_count += len(selected_idx)

        trace_rows.append(
            _trace_state(
                task_name=task_name,
                target_col=target_col,
                unit=unit,
                input_type=input_type,
                feature_source=feature_source,
                feature_set=feature_set,
                feature_set_display=feature_set_display,
                strategy=strategy,
                fold=fold,
                repeat=repeat,
                initial_fraction=initial_fraction,
                initial_size=len(initial_idx),
                acquired_count=acquired_count,
                cycle=cycle,
                y_train=y_train,
                labeled_idx=labeled_idx,
                top_idx=top_idx,
                best_idx=best_idx,
                global_best_objective=global_best_objective,
                global_best_target=global_best_target,
                objective=config.objective,
                selected_this_cycle=len(selected_idx),
                fit_seconds=float(metadata.get("fit_seconds", 0.0)),
                predict_seconds=float(metadata.get("predict_seconds", 0.0)),
                uncertainty_proxy=str(metadata.get("uncertainty_proxy", "none")),
                mean_uncertainty_proxy=(
                    float(metadata["mean_uncertainty_proxy"])
                    if "mean_uncertainty_proxy" in metadata
                    else None
                ),
            )
        )

    return trace_rows, acquisition_rows


def _summarize_trace(trace_df: pd.DataFrame) -> pd.DataFrame:
    if trace_df.empty:
        return pd.DataFrame()

    group_cols = [
        "task",
        "target",
        "unit",
        "matbench_input_type",
        "feature_source",
        "feature_set",
        "feature_set_display",
        "strategy",
        "fold",
        "repeat",
        "initial_fraction",
    ]
    rows: list[dict] = []
    for keys, group in trace_df.groupby(group_cols, dropna=False):
        group = group.sort_values("acquired_count")
        base = dict(zip(group_cols, keys))
        final = group.iloc[-1]
        found_top = group[group["found_global_best"]]
        found_any_top = group[group["found_any_top_fraction"]]
        rows.append(
            {
                **base,
                "initial_size": int(group.iloc[0]["initial_size"]),
                "train_fold_size": int(group.iloc[0]["train_fold_size"]),
                "candidate_pool_initial_size": int(group.iloc[0]["candidate_pool_initial_size"]),
                "budget_acquired": int(final["acquired_count"]),
                "found_global_best_within_budget": bool(final["found_global_best"]),
                "acquisitions_to_global_best": (
                    int(found_top.iloc[0]["acquired_count"]) if not found_top.empty else np.nan
                ),
                "found_any_top_fraction_within_budget": bool(final["found_any_top_fraction"]),
                "acquisitions_to_first_top_fraction": (
                    int(found_any_top.iloc[0]["acquired_count"])
                    if not found_any_top.empty
                    else np.nan
                ),
                "final_best_found_target": float(final["best_found_target"]),
                "global_best_target": float(final["global_best_target"]),
                "final_objective_gap": float(final["objective_gap"]),
                "final_objective_regret_fraction": float(final["objective_regret_fraction"]),
                "final_best_objective_percentile_found": float(
                    final["best_objective_percentile_found"]
                ),
                "final_top_fraction_hit_count": int(final["top_fraction_hit_count"]),
                "top_fraction_total": int(final["top_fraction_total"]),
                "final_top_fraction_hit_rate": float(final["top_fraction_hit_rate"]),
                "fit_seconds_sum": float(group["fit_seconds"].sum()),
                "predict_seconds_sum": float(group["predict_seconds"].sum()),
            }
        )
    return pd.DataFrame(rows)


def _aggregate_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()

    group_cols = [
        "task",
        "unit",
        "feature_set",
        "feature_set_display",
        "strategy",
        "initial_fraction",
    ]
    return (
        summary_df.groupby(group_cols, dropna=False)
        .agg(
            n_runs=("fold", "count"),
            found_global_best_rate=("found_global_best_within_budget", "mean"),
            acquisitions_to_global_best_mean=("acquisitions_to_global_best", "mean"),
            acquisitions_to_global_best_median=("acquisitions_to_global_best", "median"),
            first_top_fraction_acquisitions_mean=(
                "acquisitions_to_first_top_fraction",
                "mean",
            ),
            final_objective_gap_mean=("final_objective_gap", "mean"),
            final_objective_regret_fraction_mean=(
                "final_objective_regret_fraction",
                "mean",
            ),
            final_best_objective_percentile_mean=(
                "final_best_objective_percentile_found",
                "mean",
            ),
            final_top_fraction_hit_rate_mean=("final_top_fraction_hit_rate", "mean"),
            final_top_fraction_hit_count_mean=("final_top_fraction_hit_count", "mean"),
            fit_seconds_sum=("fit_seconds_sum", "sum"),
            predict_seconds_sum=("predict_seconds_sum", "sum"),
        )
        .reset_index()
        .sort_values(
            [
                "task",
                "initial_fraction",
                "final_objective_regret_fraction_mean",
                "final_top_fraction_hit_rate_mean",
            ],
            ascending=[True, True, True, False],
        )
    )


def _plot_best_found_curves(trace_df: pd.DataFrame, figures_dir: Path) -> Path | None:
    if trace_df.empty:
        return None
    import matplotlib.pyplot as plt
    import seaborn as sns

    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_df = (
        trace_df.groupby(
            [
                "task",
                "feature_set_display",
                "strategy",
                "initial_fraction",
                "acquired_count",
            ],
            dropna=False,
        )
        .agg(
            best_percentile_mean=("best_objective_percentile_found", "mean"),
            best_percentile_std=("best_objective_percentile_found", "std"),
        )
        .reset_index()
    )
    tasks = list(plot_df["task"].unique())
    if not tasks:
        return None

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(
        len(tasks),
        1,
        figsize=(11, max(4.2, 4.2 * len(tasks))),
        squeeze=False,
    )
    for ax, task_name in zip(axes[:, 0], tasks):
        task_df = plot_df.query("task == @task_name")
        sns.lineplot(
            data=task_df,
            x="acquired_count",
            y="best_percentile_mean",
            hue="strategy",
            style="feature_set_display",
            markers=True,
            dashes=False,
            ax=ax,
        )
        ax.set_title(f"{task_name}: active-learning best-found curve")
        ax.set_xlabel("Acquired labels after initialization")
        ax.set_ylabel("Best objective percentile found")
        ax.set_ylim(0, 1.02)
        ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    path = figures_dir / "active_learning_best_found_curve.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_top_fraction_hit_curves(trace_df: pd.DataFrame, figures_dir: Path) -> Path | None:
    if trace_df.empty:
        return None
    import matplotlib.pyplot as plt
    import seaborn as sns

    plot_df = (
        trace_df.groupby(
            [
                "task",
                "feature_set_display",
                "strategy",
                "initial_fraction",
                "acquired_count",
            ],
            dropna=False,
        )
        .agg(hit_rate_mean=("top_fraction_hit_rate", "mean"))
        .reset_index()
    )
    tasks = list(plot_df["task"].unique())
    if not tasks:
        return None

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(
        len(tasks),
        1,
        figsize=(11, max(4.2, 4.2 * len(tasks))),
        squeeze=False,
    )
    for ax, task_name in zip(axes[:, 0], tasks):
        task_df = plot_df.query("task == @task_name")
        sns.lineplot(
            data=task_df,
            x="acquired_count",
            y="hit_rate_mean",
            hue="strategy",
            style="feature_set_display",
            markers=True,
            dashes=False,
            ax=ax,
        )
        ax.set_title(f"{task_name}: top-candidate hit rate during screening")
        ax.set_xlabel("Acquired labels after initialization")
        ax.set_ylabel("Fraction of top candidates discovered")
        ax.set_ylim(0, 1.02)
        ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    path = figures_dir / "active_learning_top_fraction_hit_curve.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_strategy_ranking(aggregate_df: pd.DataFrame, figures_dir: Path) -> Path | None:
    if aggregate_df.empty:
        return None
    import matplotlib.pyplot as plt
    import seaborn as sns

    plot_df = aggregate_df.copy()
    plot_df["condition"] = (
        plot_df["strategy"]
        + "\n"
        + plot_df["feature_set_display"]
        + "\ninit="
        + plot_df["initial_fraction"].map(lambda value: f"{value:.0%}")
    )
    tasks = list(plot_df["task"].unique())
    if not tasks:
        return None

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(
        len(tasks),
        1,
        figsize=(12, max(4.5, 4.5 * len(tasks))),
        squeeze=False,
    )
    for ax, task_name in zip(axes[:, 0], tasks):
        task_df = plot_df.query("task == @task_name").sort_values(
            "final_objective_regret_fraction_mean"
        )
        sns.barplot(
            data=task_df,
            x="condition",
            y="final_objective_regret_fraction_mean",
            color="#4c78a8",
            ax=ax,
        )
        ax.set_title(f"{task_name}: final regret after active-learning budget")
        ax.set_xlabel("")
        ax.set_ylabel("Mean final objective regret fraction, lower is better")
        ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    path = figures_dir / "active_learning_final_regret_ranking.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_markdown_summary(
    *,
    tables_dir: Path,
    config: ActiveLearningConfig,
    aggregate_df: pd.DataFrame,
) -> Path:
    lines = [
        "# Active-Learning Screening Summary",
        "",
        "This extension is inspired by ICAL-style pool-based active learning, "
        "but it is a retrospective Matbench diagnostic rather than a full ICAL reproduction.",
        "",
        "The official Matbench train fold is split into a small labeled set and an "
        "unlabeled candidate pool. Each strategy sequentially acquires labels from "
        "that pool, and success is measured by discovery efficiency rather than MAE.",
        "",
        "## Configuration",
        "",
        f"- Objective direction: `{config.objective}`.",
        f"- Initial labeled fractions: `{', '.join(f'{v:.0%}' for v in config.initial_fractions)}`.",
        f"- Top-candidate definition: top `{config.top_fraction:.0%}` by objective value.",
        f"- Acquisition batch size: `{config.acquisition_batch_size}`.",
        f"- Maximum acquisitions: `{config.max_acquisitions}`.",
        f"- Strategies: `{', '.join(config.strategies)}`.",
        "",
        "## Best Conditions By Final Regret",
        "",
    ]

    if aggregate_df.empty:
        lines.append("No successful active-learning runs were available.")
    else:
        for (task, init_fraction), group in aggregate_df.groupby(
            ["task", "initial_fraction"], dropna=False
        ):
            best = group.sort_values("final_objective_regret_fraction_mean").iloc[0]
            lines.append(
                "- `{}` init={:.0%}: best condition is `{}` with `{}`, "
                "mean final regret={:.4g}, mean top-candidate hit rate={:.3g}.".format(
                    task,
                    init_fraction,
                    best["strategy"],
                    best["feature_set_display"],
                    best["final_objective_regret_fraction_mean"],
                    best["final_top_fraction_hit_rate_mean"],
                )
            )

    lines.extend(
        [
            "",
            "## Interpretation Guide",
            "",
            "- If TabPFN strategies reach high objective percentiles faster than random, "
            "the model is useful for candidate selection, not only static prediction.",
            "- If `Magpie + all structure` improves the curves, structure descriptors help "
            "the sequential discovery workflow as well as the MAE benchmark.",
            "- If UCB/disagreement strategies beat pure greedy selection, model disagreement "
            "is acting as a useful uncertainty proxy. If not, it is still a useful negative "
            "result: simple uncertainty proxies may be insufficient without full probabilistic "
            "ICAL-style uncertainty.",
            "",
            "## Important Scope Note",
            "",
            "This diagnostic does not use the official test fold during acquisition. "
            "It simulates screening only inside the official training fold so the "
            "main benchmark evaluation remains untouched.",
            "",
        ]
    )

    path = tables_dir / "active_learning_screening_summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def run_active_learning_screening(config: ActiveLearningConfig) -> None:
    from matbench_tabpfn.features import build_task_features, load_matbench_task
    from matbench_tabpfn.paths import (
        collect_environment_manifest,
        create_run_paths,
        write_json,
    )

    paths = create_run_paths(
        PROJECT_ROOT,
        run_id=None,
        base_dir="results/active_learning_screening",
        update_latest=False,
    )
    write_json(paths.logs / "config.json", asdict(config))
    write_json(paths.logs / "environment.json", collect_environment_manifest(asdict(config)))

    trace_rows: list[dict] = []
    acquisition_rows: list[dict] = []

    for task_name in config.tasks:
        task = load_matbench_task(task_name)
        target_col = task.metadata["target"]
        unit = task.metadata.get("unit")
        input_type = task.metadata["input_type"]
        folds = _iter_requested_folds(task, config.folds)

        for feature_set in config.feature_sets:
            print(f"Building features: {task_name} | {feature_set}")
            try:
                features, feature_source, feature_set_display = build_task_features(
                    task_name,
                    task,
                    feature_set,
                    paths.features,
                    use_cache=config.use_feature_cache,
                    show_progress=config.show_feature_progress,
                    n_jobs=config.feature_n_jobs,
                )
            except ValueError as exc:
                print(f"Skipping {task_name} | {feature_set}: {exc}")
                continue

            for fold in folds:
                X_train_raw, y_train = task.get_train_and_val_data(fold)
                X_train = features.loc[X_train_raw.index]

                for initial_fraction in config.initial_fractions:
                    for repeat in range(config.repeats):
                        for strategy in config.strategies:
                            print(
                                f"{task_name} | {feature_set} | {strategy} | "
                                f"fold {fold} | init {initial_fraction:.0%} | repeat {repeat}"
                            )
                            run_trace, run_acquisitions = _simulate_one_run(
                                task_name=task_name,
                                target_col=target_col,
                                unit=unit,
                                input_type=input_type,
                                feature_source=feature_source,
                                feature_set=feature_set,
                                feature_set_display=feature_set_display,
                                strategy=strategy,
                                fold=fold,
                                repeat=repeat,
                                initial_fraction=initial_fraction,
                                X_train=X_train,
                                y_train=y_train,
                                config=config,
                            )
                            trace_rows.extend(run_trace)
                            acquisition_rows.extend(run_acquisitions)

    trace_df = pd.DataFrame(trace_rows)
    acquisition_df = pd.DataFrame(acquisition_rows)
    summary_df = _summarize_trace(trace_df)
    aggregate_df = _aggregate_summary(summary_df)

    trace_df.to_csv(paths.predictions / "active_learning_trace.csv", index=False)
    acquisition_df.to_csv(paths.predictions / "active_learning_acquisitions.csv", index=False)
    summary_df.to_csv(paths.tables / "active_learning_run_summary.csv", index=False)
    aggregate_df.to_csv(paths.tables / "active_learning_aggregate_summary.csv", index=False)

    _plot_best_found_curves(trace_df, paths.figures)
    _plot_top_fraction_hit_curves(trace_df, paths.figures)
    _plot_strategy_ranking(aggregate_df, paths.figures)
    _write_markdown_summary(
        tables_dir=paths.tables,
        config=config,
        aggregate_df=aggregate_df,
    )

    print(f"\nActive-learning screening complete: {paths.root}")
    print(f"Trace: {paths.predictions / 'active_learning_trace.csv'}")
    print(f"Summary: {paths.tables / 'active_learning_aggregate_summary.csv'}")


def main() -> None:
    config = _parse_args()
    run_active_learning_screening(config)


if __name__ == "__main__":
    main()
