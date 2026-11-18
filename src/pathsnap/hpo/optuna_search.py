"""Bayesian hyperparameter optimisation of the selected classifier (LightGBM).

Why Bayesian (Optuna TPE) and not grid search: each trial costs a full cross-validation,
so we want every new trial to be informed by the earlier ones. TPE models which regions
of the search space gave good scores and samples more from them.

Objective  : mean candidate-level Macro F1 (threshold 0.5) over GroupKFold folds split by
             trip. This is the same metric Member 2 used to select LightGBM, so the tuned
             and untuned models are judged identically.
Search data: the 80 train trips only. student_test is never used during the search.
Storage    : SQLite, so an interrupted study can be resumed (load_if_exists=True).
Trial 0    : the untuned configuration is enqueued first, so the study always contains
             the baseline it has to beat.
"""

from __future__ import annotations

import time

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score

#: Member 2's untuned LightGBM configuration (src/pathsnap/models/registry.py).
UNTUNED_PARAMS: dict = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=20,
                            subsample=0.8, colsample_bytree=0.9, reg_lambda=0.0)

#: Search space, with the reason for each range.
SEARCH_SPACE_DOC = {
    "num_leaves": "15-255 (log): tree complexity; 63 untuned, both simpler and richer trees tried",
    "learning_rate": "0.01-0.2 (log): step size; smaller needs more trees",
    "n_estimators": "100-800 (step 50): number of boosting rounds",
    "min_child_samples": "10-300 (log): minimum rows per leaf; larger = smoother, less overfitting",
    "subsample": "0.5-1.0: share of rows per tree (bagging), reduces variance",
    "colsample_bytree": "0.5-1.0: share of features per tree, decorrelates trees",
    "reg_lambda": "1e-3-10 (log): L2 penalty on leaf values",
}


def suggest_params(trial: optuna.Trial) -> dict:
    return dict(
        num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        n_estimators=trial.suggest_int("n_estimators", 100, 800, step=50),
        min_child_samples=trial.suggest_int("min_child_samples", 10, 300, log=True),
        subsample=trial.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
    )


def make_lgbm(params: dict, seed: int = 42) -> LGBMClassifier:
    return LGBMClassifier(**params, subsample_freq=1, n_jobs=-1, random_state=seed, verbose=-1)


def make_objective(df: pd.DataFrame, folds: list, features: list[str], seed: int = 42):
    """Objective = mean validation Macro F1 over the given trip-grouped folds.
    Reports the running mean after each fold so hopeless trials can be pruned early."""

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial)
        scores = []
        t0 = time.perf_counter()
        for k, (tri, vai) in enumerate(folds):
            model = make_lgbm(params, seed)
            model.fit(df[features].iloc[tri], df["y"].iloc[tri])
            p = model.predict_proba(df[features].iloc[vai])[:, 1]
            scores.append(f1_score(df["y"].iloc[vai], (p >= 0.5).astype(int), average="macro"))
            trial.report(float(np.mean(scores)), step=k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("fold_scores", [float(s) for s in scores])
        trial.set_user_attr("seconds", time.perf_counter() - t0)
        return float(np.mean(scores))

    return objective


def run_study(df, folds, features, n_trials: int, storage: str, study_name: str, seed: int = 42) -> optuna.Study:
    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner, storage=storage,
                                study_name=study_name, load_if_exists=True)
    if len(study.trials) == 0:
        # reg_lambda=0 lies outside the log-scale range, so the untuned trial uses its
        # lower bound 1e-3 (a negligible L2 penalty). Everything else is identical.
        study.enqueue_trial({**UNTUNED_PARAMS, "reg_lambda": 1e-3})
    remaining = max(0, n_trials - len(study.trials))
    if remaining:
        study.optimize(make_objective(df, folds, features, seed), n_trials=remaining)
    return study
