"""Structured feature engineering and preprocessing pipelines."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def infer_structured_columns(
    df: pd.DataFrame,
    target_col: str,
    text_col: str,
    id_col: str,
    explicit_drop_cols: Iterable[str] | None = None,
) -> list[str]:
    """Infer structured columns by excluding id/target/text and explicit drops.

    Also excludes columns containing list/dict/set values, which are not valid
    scalar tabular features for the default sklearn preprocessing pipeline.
    """
    drop = {target_col, text_col, id_col}
    if explicit_drop_cols:
        drop.update(explicit_drop_cols)

    candidate_cols = [c for c in df.columns if c not in drop]
    safe_cols: list[str] = []
    for col in candidate_cols:
        s = df[col]
        if s.dtype == "object":
            sample = s.dropna().head(200)
            has_unhashable = sample.apply(lambda v: isinstance(v, (list, dict, set))).any()
            if has_unhashable:
                continue
        safe_cols.append(col)

    return safe_cols


def _one_hot_encoder() -> OneHotEncoder:
    """Build OneHotEncoder compatible with sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def build_structured_preprocessor(df_struct: pd.DataFrame) -> ColumnTransformer:
    """Create preprocessing pipeline for numeric + categorical structured inputs."""
    num_cols = df_struct.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df_struct.columns if c not in num_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
    )


def fit_transform_structured(
    train_struct: pd.DataFrame,
    valid_struct: pd.DataFrame | None = None,
) -> Tuple[ColumnTransformer, np.ndarray, np.ndarray | None]:
    """Fit structured preprocessor and transform train/(optional) valid sets."""
    preprocessor = build_structured_preprocessor(train_struct)
    x_train = preprocessor.fit_transform(train_struct)
    x_valid = preprocessor.transform(valid_struct) if valid_struct is not None else None
    return preprocessor, x_train, x_valid
