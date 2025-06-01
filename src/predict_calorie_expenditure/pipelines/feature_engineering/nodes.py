import numpy as np
import pandas as pd

from typing import Optional


def _feature_binning(df: pd.DataFrame) -> pd.DataFrame:

    if "Age" in df.columns:
        df["Age_Group"] = pd.cut(
            df["Age"],
            bins=[0, 25, 40, 60, 100],
            labels=["until25", "25-40", "40-60", "60-100"],
        )

    if "Heart_Rate" in df.columns:
        df["Heart_Rate_Group"] = pd.cut(
            df["Heart_Rate"],
            bins=[0, 90, 100, 200],
            labels=[
                "low",
                "normal",
                "high",
            ],
        )

    if "Duration" in df.columns:
        df["Duration_Group"] = pd.qcut(
            df["Duration"], q=4, labels=["short", "medium", "long", "very_long"]
        )

    return df


def _feature_transformations(df: pd.DataFrame) -> pd.DataFrame:

    if "Body_Temp" in df.columns:
        df["Body_Temp_Deviation"] = df["Body_Temp"] - 36.5
        df["Is_Fever"] = (df["Body_Temp"] >= 40.0).astype(int)

    if "Heart_Rate" in df.columns:
        df["Heart_Rate_Deviation"] = df["Heart_Rate"] - 60

    if "Duration" in df.columns:
        df["Is_Short_Session"] = (df["Duration"] < 10).astype(int)

    return df


def _feature_interactions(df: pd.DataFrame) -> pd.DataFrame:

    if "Weight" in df.columns and "Height" in df.columns:
        df["BMI"] = df["Weight"] / ((df["Height"] / 100) ** 2)

    if (
        "Duration" in df.columns
        and "Heart_Rate" in df.columns
        and "Weight" in df.columns
    ):
        df["Calories_Est"] = df["Duration"] * df["Heart_Rate"] * df["Weight"] / 10000

    if "Duration" in df.columns and "Heart_Rate" in df.columns:
        df["Effort"] = df["Duration"] * df["Heart_Rate"]

    if "Heart_Rate" in df.columns and "Age" in df.columns:
        df["Intensity_Index"] = df["Heart_Rate"] / (220 - df["Age"])

    if "Duration" in df.columns and "Weight" in df.columns:
        df["Effort_Per_Kg"] = df["Duration"] * df["Heart_Rate"] / df["Weight"]

    if "Duration" in df.columns and "Age" in df.columns:
        df["Duration_Per_Age"] = df["Duration"] / df["Age"]

    return df


def _feature_logs(df: pd.DataFrame) -> pd.DataFrame:

    if "Duration" in df.columns:
        df["Log_Duration"] = np.log1p(df["Duration"])

    if "Weight" in df.columns:
        df["Log_Weight"] = np.log1p(df["Weight"])

    if "Height" in df.columns:
        df["Log_Height"] = np.log1p(df["Height"])

    if "Age" in df.columns:
        df["Log_Age"] = np.log1p(df["Age"])

    if "Calories_Est" in df.columns:
        df["Log_Calories_Est"] = np.log1p(df["Calories_Est"])

    if "Effort" in df.columns:
        df["Log_Effort"] = np.log1p(df["Effort"])

    if "Effort_Per_Kg" in df.columns:
        df["Log_Effort_Per_Kg"] = np.log1p(df["Effort_Per_Kg"])

    return df


def create_features(
    df: pd.DataFrame,
    use_binning: Optional[bool] = True,
    use_transformations: Optional[bool] = True,
    use_interactions: Optional[bool] = True,
    use_logs: Optional[bool] = True,
) -> pd.DataFrame:

    df = df.copy()

    if use_binning:
        df = _feature_binning(df)

    if use_transformations:
        df = _feature_transformations(df)

    if use_interactions:
        df = _feature_interactions(df)

    if use_logs:
        df = _feature_logs(df)

    return df
