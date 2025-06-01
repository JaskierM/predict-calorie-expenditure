import numpy as np
import pandas as pd

from typing import List, Dict, Tuple
from pandas.api.types import CategoricalDtype
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer


def _encode_cat(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()
    for col in X.columns:
        if X[col].dtype.name == "object":
            X[col] = X[col].astype("category")
        if X[col].dtype.name == "category":
            X[col] = X[col].cat.codes.astype(int)
    return X


def _encode_ord(
    df: pd.DataFrame, feature_ord_levels: Dict[str, List[str]]
) -> pd.DataFrame:
    for feature, levels in feature_ord_levels.items():
        if feature in df.columns:
            cat_type = CategoricalDtype(categories=levels, ordered=True)
            df[feature] = df[feature].astype(str).astype(cat_type).cat.codes
    return df


def _build_pipelines(
    feature_ord_levels: Dict[str, List[str]],
) -> Tuple[Pipeline, Pipeline, FunctionTransformer]:
    ord_pipeline = Pipeline(
        [
            (
                "ord_codes",
                FunctionTransformer(
                    lambda X: _encode_ord(X, feature_ord_levels),
                    validate=False,
                    feature_names_out="one-to-one",
                ),
            )
        ]
    )
    cat_pipeline = Pipeline(
        [
            (
                "cat_codes",
                FunctionTransformer(
                    _encode_cat, validate=False, feature_names_out="one-to-one"
                ),
            )
        ]
    )
    num_pipeline = FunctionTransformer(
        lambda X: X, validate=False, feature_names_out="one-to-one"
    )
    return ord_pipeline, cat_pipeline, num_pipeline


def fit_preprocessor_and_transform_train(
    train: pd.DataFrame,
    feature_ord_levels: Dict[str, List[str]],
    target_column: str = "Calories",
):
    y_train = train[target_column]
    X_train = train.drop(columns=[target_column])

    ord_cols = [col for col in feature_ord_levels.keys() if col in X_train.columns]

    cat_cols = [
        col
        for col in X_train.select_dtypes(["object", "category"]).columns
        if col not in ord_cols
    ]

    num_cols = [
        col
        for col in X_train.select_dtypes(["int64", "float64"]).columns
        if col not in ord_cols and col not in cat_cols
    ]

    ord_pipeline, cat_pipeline, num_pipeline = _build_pipelines(feature_ord_levels)

    transformers = []
    if ord_cols:
        transformers.append(("ord", ord_pipeline, ord_cols))
    if cat_cols:
        transformers.append(("cat", cat_pipeline, cat_cols))
    if num_cols:
        transformers.append(("num", num_pipeline, num_cols))

    preprocessor = ColumnTransformer(transformers)

    X_train_arr = preprocessor.fit_transform(X_train)
    X_train_df = pd.DataFrame(
        X_train_arr,
        columns=preprocessor.get_feature_names_out(),
        index=train.index,
    )
    y_train_log = np.log1p(y_train)

    return X_train_df, y_train_log, preprocessor


def transform_with_preprocessor(
    df: pd.DataFrame,
    preprocessor: ColumnTransformer,
    target_column: str = "Calories",
) -> pd.DataFrame:
    X = df.drop(columns=[target_column], errors="ignore")
    X_arr = preprocessor.transform(X)
    X_df = pd.DataFrame(
        X_arr, columns=preprocessor.get_feature_names_out(), index=df.index
    )
    return X_df
