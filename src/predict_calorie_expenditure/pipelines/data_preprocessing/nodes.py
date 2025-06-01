import pandas as pd

from typing import Optional, List


def _clean_data(
    df: pd.DataFrame,
    drop_duplicates: Optional[bool] = True,
    id_column: Optional[str] = "id",
) -> pd.DataFrame:

    if id_column in df.columns:
        df.set_index(id_column, inplace=True)

    if drop_duplicates:
        df = df.drop_duplicates()

    return df


def _cast_feature_types(
    df: pd.DataFrame,
    features_cat: Optional[List[str]] = None,
    features_date: Optional[List[str]] = None,
    datetime_format: Optional[str] = "%d/%m/%Y",
) -> pd.DataFrame:

    if features_cat:
        for col in features_cat:
            if col in df.columns:
                df[col] = df[col].astype("category")

    if features_date:
        for col in features_date:
            if col in df.columns:
                df[col] = pd.to_datetime(
                    df[col], format=datetime_format, errors="coerce"
                )

    return df


def preprocess_data(
    df: pd.DataFrame,
    drop_duplicates: Optional[bool] = True,
    id_column: Optional[str] = "id",
    features_cat: Optional[List[str]] = None,
    features_date: Optional[List[str]] = None,
    datetime_format: Optional[str] = "%d/%m/%Y",
) -> pd.DataFrame:

    df = df.copy()
    df = _clean_data(df, drop_duplicates=drop_duplicates, id_column=id_column)
    df = _cast_feature_types(
        df,
        features_cat=features_cat,
        features_date=features_date,
        datetime_format=datetime_format,
    )

    return df
