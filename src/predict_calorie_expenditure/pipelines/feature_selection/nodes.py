import pandas as pd

from typing import Optional, List, Dict


def _drop_features(
    df: pd.DataFrame, features_to_drop: Optional[List[str]] = None
) -> pd.DataFrame:
    if features_to_drop:
        df = df.drop(columns=features_to_drop, errors="ignore")
    return df


def _get_outlier_indexes(
    df: pd.DataFrame, feature_outlier_thresholds: Dict[str, Dict[str, float]]
) -> List[int]:

    outlier_indexes = set()

    for feature, thresholds in feature_outlier_thresholds.items():
        if feature in df.columns:
            if "upper" in thresholds.keys():
                outlier_indexes = outlier_indexes.union(
                    set(df[df[feature] > thresholds["upper"]].index)
                )
            if "lower" in thresholds.keys():
                outlier_indexes = outlier_indexes.union(
                    set(df[df[feature] < thresholds["lower"]].index)
                )

    return list(outlier_indexes)


def _drop_outliers(
    df: pd.DataFrame,
    feature_outlier_thresholds: Optional[Dict[str, Dict[str, float]]] = None,
) -> pd.DataFrame:

    if feature_outlier_thresholds:
        outlier_indexes = _get_outlier_indexes(
            df, feature_outlier_thresholds=feature_outlier_thresholds
        )
        return df.drop(index=outlier_indexes)

    return df


def select_and_preprocess_features(
    df: pd.DataFrame,
    features_to_drop: Optional[List[str]] = None,
    feature_outlier_thresholds: Optional[Dict[str, Dict[str, float]]] = None,
    drop_duplicates: Optional[bool] = True,
) -> pd.DataFrame:

    df = df.copy()
    df = _drop_features(df, features_to_drop=features_to_drop)
    df = _drop_outliers(df, feature_outlier_thresholds=feature_outlier_thresholds)

    if drop_duplicates:
        df = df.drop_duplicates()

    return df
