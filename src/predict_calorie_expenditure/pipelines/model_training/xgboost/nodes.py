import numpy as np
import pandas as pd
import xgboost as xgb
import optuna

from typing import Dict, Tuple
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import make_scorer, mean_squared_error
from kedro_mlflow.io.metrics import MlflowMetricDataset


def _make_rmse_scorer():
    return make_scorer(
        lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
        greater_is_better=False,
    )


def optimize_xgboost_params(
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int,
    cv_folds: int,
    random_state: int,
) -> dict:
    X = X.apply(pd.to_numeric, errors="coerce")

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 0, 10),
        }

        model = xgb.XGBRegressor(**params, random_state=random_state)
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        rmse_scorer = _make_rmse_scorer()

        scores = cross_val_score(model, X, y, scoring=rmse_scorer, cv=cv)
        return -scores.mean()

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    return study.best_params


def train_xgboost_model(
    X: pd.DataFrame,
    y: pd.Series,
    params: Dict,
    cv_folds: int,
    random_state: int,
) -> Tuple[xgb.XGBRegressor, float]:

    X = X.apply(pd.to_numeric, errors="coerce")
    model = xgb.XGBRegressor(**params, random_state=random_state)
    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    rmse_scorer = _make_rmse_scorer()

    rmse_scores = cross_val_score(
        model,
        X,
        y,
        scoring=rmse_scorer,
        cv=cv,
        error_score="raise",
        verbose=True,
    )

    mlflow_rmse = MlflowMetricDataset(key="rmse")
    mlflow_rmse.save(-rmse_scores.mean())

    model.fit(X, y)

    return model


def predict_xgboost_model(model: xgb.XGBRegressor, X_test: pd.DataFrame) -> pd.Series:
    X_test = X_test.apply(pd.to_numeric, errors="raise")
    preds_log = model.predict(X_test)
    preds = np.expm1(preds_log)
    return pd.Series(preds, index=X_test.index, name="preds")
