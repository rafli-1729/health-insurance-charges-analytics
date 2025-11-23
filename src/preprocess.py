# src/preprocess.py
from typing import List, Tuple
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import pandas as pd

def build_preprocessor(num_features: List[str], cat_features: List[str], 
                       num_imputer=None, num_scaler=None
                       cat_imputer=None, cat_encoder=None, 
) -> ColumnTransformer:
    """
    Build a ColumnTransformer handling numeric and categorical columns.
    drop_first_for_ohe: useful for linear models to avoid dummy trap, not needed for tree models.
    """
    # Validate feature lists
    if not isinstance(num_features, list) or not isinstance(cat_features, list):
        raise TypeError("num_features and cat_features must be lists of column names.")

    if num_imputer is None:
        num_imputer = SimpleImputer(strategy="mean")
    if num_scaler is None:
        num_scaler = StandardScaler()
    if cat_imputer is None:
        cat_imputer = SimpleImputer(strategy="most_frequent")
    if cat_encoder is None:
        cat_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    numeric_pipeline = Pipeline([
        ("imputer", num_imputer),
        ("scaler", num_scaler)
    ])

    categorical_pipeline = Pipeline([
        ("imputer", cat_imputer),
        ("encoder", cat_encoder)
    ])

    transformers = []

    if cat_features:
        transformers.append(("cat", categorical_pipeline, cat_features))
    if num_features:
        transformers.append(("num", numeric_pipeline, num_features))

    if not transformers:
        raise ValueError("Both num_features and cat_features are empty. Nothing to transform.")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False
    )

def fit_transform_df(
    preprocessor: ColumnTransformer,
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Fit the preprocessor on a DataFrame and return a transformed DataFrame
    along with output feature names.

    Returns
    -------
    transformed_df : pd.DataFrame
    feature_names : list of str
    """
    arr = preprocessor.fit_transform(df)
    try:
        feature_names = list(preprocessor.get_feature_names_out())
    except Exception:
        feature_names = [f"feature_{i}" for i in range(arr.shape[1])]

    transformed_df = pd.DataFrame(arr, columns=feature_names)
    return transformed_df, feature_names
