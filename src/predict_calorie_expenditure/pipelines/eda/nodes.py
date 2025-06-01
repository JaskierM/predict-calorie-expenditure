import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from ydata_profiling import ProfileReport
from sklearn.feature_selection import mutual_info_regression
from typing import Optional, Union


def _scale_df(
    df: pd.DataFrame,
    scale_factor: Optional[float] = 1.0,
    random_state: Optional[Union[int, None]] = None,
) -> pd.DataFrame:
    if scale_factor and 0 < scale_factor < 1.0:
        df = df.sample(frac=scale_factor, random_state=random_state)
    return df


def generate_profile_report(
    df: pd.DataFrame,
    dataset_name: str,
    output_path: str,
    scale_factor: Optional[float] = 1.0,
    random_state: Optional[Union[int, None]] = None,
) -> None:

    df = _scale_df(df, scale_factor=scale_factor, random_state=random_state)

    report = ProfileReport(
        df, title=f"Profile Report: {dataset_name}", explorative=True
    )
    output_path = Path(output_path) / f"profile_report_{dataset_name}.html"
    report.to_file(output_path)
    print(f"Saved report: {output_path}")


def compute_mi_scores_plot(
    df: pd.DataFrame,
    target_column: str,
    output_path: str,
    scale_factor: Optional[float] = 1.0,
    random_state: Optional[Union[int, None]] = None,
) -> None:

    df = _scale_df(df, scale_factor=scale_factor, random_state=random_state)

    X = df.drop(columns=[target_column])
    y = df[target_column]

    for col in X.select_dtypes(include=["object", "category", "datetime64[ns]"]):
        X[col], _ = X[col].factorize()

    discrete_mask = [pd.api.types.is_integer_dtype(dtype) for dtype in X.dtypes]
    mi = mutual_info_regression(X, y, discrete_features=discrete_mask, random_state=0)
    mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=True)

    plt.figure(figsize=(10, max(6, len(mi_series) * 0.25)))
    plt.barh(mi_series.index, mi_series.values)
    plt.title(f"Mutual Information Scores (sampled {len(df):,} rows)")
    plt.tight_layout()

    output_path = Path(output_path) / "mi_scores.png"
    plt.savefig(output_path)
    plt.close()
