from __future__ import annotations
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

CATEGORICAL = ["protocol_type", "service", "flag"]
TARGET = "class"


def _rare_bucket(series: pd.Series, min_count: int = 50) -> pd.Series:
    counts = series.value_counts(dropna=False)
    rare_vals = counts[counts < min_count].index
    return series.where(~series.isin(rare_vals), other="_RARE_")


def preprocess(df: pd.DataFrame) -> Tuple:
    """
    Return (X_train, X_test, y_train, y_test, preproc)
    - Stratified 80/20 split
    - Rare-category bucketing for stability
    - Numeric: median impute + scaler
    - Categorical: most_frequent impute + OHE
    """
    y = df[TARGET].astype("int64")
    X = df.drop(columns=[TARGET]).copy()

    # Rare bucketing only on known categoricals
    for c in CATEGORICAL:
        if c in X.columns:
            X[c] = _rare_bucket(X[c], min_count=50)

    # Identify columns
    num_cols = [c for c in X.columns if c not in CATEGORICAL]
    cat_cols = [c for c in CATEGORICAL if c in X.columns]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    preproc = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler(with_mean=False)),  # sparse-friendly
            ]), num_cols),
            ("cat", Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
            ]), cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )

    return X_train, X_test, y_train, y_test, preproc
