# src/evaluation.py
from typing import Dict, Any, Optional
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate, KFold
import numpy as np
import pandas as pd
from .utils import logger, clean_feature_names
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_final_model(model_cls, best_params: Dict[str, Any], X, y,
                         preprocessor, n_splits: int = 5, n_jobs: int = 1) -> Pipeline:
    """
    Build final Pipeline and run cross_validate with multiple metrics.
    Returns trained pipeline (unfitted) so caller can fit on full data if desired.
    """
    from sklearn.pipeline import Pipeline as SKPipeline
    from sklearn.model_selection import cross_validate, KFold

    if X.shape[0] < n_splits:
        raise ValueError("Not enough rows for requested n_splits.")

    pipeline = SKPipeline([("preprocessor", preprocessor), ("regressor", model_cls(**best_params))])
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scoring = {"r2": "r2", "mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error"}

    logger.info(f"Running {n_splits}-Fold CV for final evaluation...")
    results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs, return_train_score=False)

    # Print per-fold and average
    for i in range(n_splits):
        r2 = results["test_r2"][i]
        mae = -results["test_mae"][i]
        rmse = -results["test_rmse"][i]
        logger.info(f"Fold {i+1} -> R2: {r2:.4f} | MAE: {mae:.2f} | RMSE: {rmse:.2f}")

    logger.info(f"AVG R2: {results['test_r2'].mean():.4f} | AVG MAE: {-results['test_mae'].mean():.4f}")

    return pipeline

def plot_feature_importance_from_pipeline(pipeline: Pipeline, X: pd.DataFrame, y, top_n: Optional[int] = 20, show: bool = True):
    """
    Fit pipeline on X,y and plot feature importances if model supports it (e.g., tree-based).
    Returns DataFrame of importances.
    """
    pipeline.fit(X, y)
    model = pipeline.named_steps.get("regressor")
    preprocessor = pipeline.named_steps.get("preprocessor")

    raw_names = preprocessor.get_feature_names_out()
    feature_names = clean_feature_names(raw_names)

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
    else:
        raise ValueError("Model does not expose feature importances or coef_.")
    df = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False)
    if top_n:
        df = df.head(top_n)

    plt.figure(figsize=(8, max(4, len(df)*0.25)))
    sns.barplot(x="importance", y="feature", data=df)
    plt.title("Feature importances")
    plt.tight_layout()
    if show: plt.show()
    plt.close()
    return df
