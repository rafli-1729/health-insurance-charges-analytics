# src/tuning.py
from typing import Callable, Dict, Any, Type
import optuna
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.base import BaseEstimator
import numpy as np
from .utils import logger

def tune_model(model_cls: Type[BaseEstimator],
               search_space_fn: Callable[[optuna.Trial], Dict[str, Any]],
               X, y, preprocessor,
               n_trials: int = 50, cv: int = 3, scoring: str = "neg_mean_absolute_error") -> Dict[str, Any]:
    """
    Generic Optuna wrapper that tunes hyperparameters for given model class.
    Returns best params (with added safe defaults).
    """
    if X.shape[0] < cv:
        raise ValueError("Not enough rows to perform CV. Reduce cv or provide more data.")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    logger.info("Starting Optuna tuning...")

    def objective(trial: optuna.Trial):
        params = search_space_fn(trial)
        model = model_cls(**params)
        pipeline = Pipeline([("preprocessor", preprocessor), ("model", model)])
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        return -scores.mean()  # minimize error

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params.copy()
    # set safe defaults
    best.update({"random_state": 42})
    if "n_jobs" in model_cls().get_params().keys():
        best.setdefault("n_jobs", -1)
    logger.info(f"Optuna finished. Best value: {study.best_value:.4f}")
    return best
