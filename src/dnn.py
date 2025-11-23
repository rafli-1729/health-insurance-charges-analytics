# src/dnn.py
from typing import List, Tuple, Any, Dict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Activation, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import KFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import pandas as pd
from .utils import logger, clean_feature_names

def build_dnn_model(input_dim: int, learning_rate: float = 1e-2) -> Sequential:
    """Return a compiled Keras Sequential model for regression."""
    model = Sequential([
        Dense(128, input_dim=input_dim),
        BatchNormalization(),
        Activation("relu"),
        Dropout(0.3),

        Dense(64),
        BatchNormalization(),
        Activation("relu"),
        Dropout(0.2),

        Dense(16, activation="relu"),
        Dense(1, activation="linear")
    ])
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse", metrics=["mae"])
    return model

def run_dnn_cv(X: pd.DataFrame, y: pd.Series, num_cols: List[str], cat_cols: List[str],
               n_splits: int = 5, epochs: int = 100, batch_size: int = 32) -> Dict[str, float]:
    """
    Run KFold CV for DNNs. Preprocessing fitted per fold to avoid leakage.
    Returns average metrics across folds.
    """
    if X.shape[0] < n_splits:
        raise ValueError("Not enough rows for requested n_splits.")
    logger.info(f"Running DNN {n_splits}-Fold CV | epochs={epochs} | batch_size={batch_size}")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = {"r2": [], "rmse": [], "mae": []}

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        preproc = ColumnTransformer([
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                              ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first"))]), cat_cols),
            ("num", Pipeline([("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]), num_cols)
        ], remainder="drop")

        X_tr_proc = preproc.fit_transform(X_tr)
        X_val_proc = preproc.transform(X_val)

        model = build_dnn_model(input_dim=X_tr_proc.shape[1])
        callbacks = [
            EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5, verbose=0)
        ]
        model.fit(X_tr_proc, y_tr, validation_data=(X_val_proc, y_val),
                  epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=0)

        y_pred = model.predict(X_val_proc, verbose=0).flatten()
        scores["r2"].append(r2_score(y_val, y_pred))
        scores["rmse"].append(mean_squared_error(y_val, y_pred)**0.5)
        scores["mae"].append(mean_absolute_error(y_val, y_pred))
        logger.info(f"Fold {fold+1} -> R2={scores['r2'][-1]:.4f} MAE={scores['mae'][-1]:.2f} RMSE={scores['rmse'][-1]:.2f}")

    return {"R2": np.mean(scores["r2"]), "MAE": np.mean(scores["mae"]), "RMSE": np.mean(scores["rmse"])}

def train_final_dnn(X: pd.DataFrame, y: pd.Series, num_cols: List[str], cat_cols: List[str],
                    epochs: int = 100, batch_size: int = 32) -> Tuple[Any, np.ndarray, List[str]]:
    """
    Train DNN on full dataset and return (model, processed_X, feature_names).
    """
    preproc = ColumnTransformer([
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="first"))]), cat_cols),
        ("num", Pipeline([("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler())]), num_cols)
    ], remainder="drop")

    X_proc = preproc.fit_transform(X)
    raw_names = preproc.get_feature_names_out()
    feature_names = clean_feature_names(raw_names)

    model = build_dnn_model(input_dim=X_proc.shape[1])
    model.fit(X_proc, y, epochs=epochs, batch_size=batch_size, verbose=0)
    logger.info("Final DNN trained on full data.")
    return model, X_proc, feature_names

def explain_dnn_feature_importance(model, X_processed: np.ndarray, y_true: np.ndarray, feature_names: List[str], n_repeats: int = 1):
    """
    Permutation importance for a trained DNN.
    Note: expensive O(n_features * n_repeats).
    """
    baseline = mean_absolute_error(y_true, model.predict(X_processed, verbose=0).flatten())
    importances = []
    rng = np.random.default_rng(42)
    for i in range(X_processed.shape[1]):
        incrs = []
        for _ in range(n_repeats):
            Xp = X_processed.copy()
            rng.shuffle(Xp[:, i])
            pred = model.predict(Xp, verbose=0).flatten()
            incrs.append(mean_absolute_error(y_true, pred) - baseline)
        importances.append(float(np.mean(incrs)))

    df = pd.DataFrame({"feature": feature_names, "importance_mae": importances}).sort_values("importance_mae", ascending=False)
    # plotting
    plt.figure(figsize=(8, max(4, len(df)*0.25)))
    sns.barplot(data=df.head(30), x="importance_mae", y="feature")
    plt.title(f"DNN permutation importances (baseline MAE: {baseline:.2f})")
    plt.tight_layout()
    plt.show()
    plt.close()
    return df
