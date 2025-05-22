from kedro.pipeline import Pipeline, pipeline, node
from kedro.pipeline.node import Node

from predict_calorie_expenditure.pipelines.model_training.xgboost.nodes import (
    train_xgboost_model,
    optimize_xgboost_params,
    predict_xgboost_model,
)


def _optimize_xgboost_params_node() -> Node:
    return node(
        func=optimize_xgboost_params,
        inputs=dict(
            X="model_inputs_trees_X_train",
            y="model_inputs_trees_y_train",
            n_trials="params:xgboost_optuna_n_trials",
            cv_folds="params:cv_folds",
            random_state="params:random_state",
        ),
        outputs="xgboost_best_params",
        tags=["train"],
        name="optimize_xgboost_params",
    )


def _train_xgboost_model_node() -> Node:
    return node(
        func=train_xgboost_model,
        inputs=dict(
            X="model_inputs_trees_X_train",
            y="model_inputs_trees_y_train",
            params="xgboost_best_params",
            cv_folds="params:cv_folds",
            random_state="params:random_state",
        ),
        outputs="xgboost_model",
        tags=["train"],
        name="train_xgboost_model",
    )


def _predict_xgboost_model_node() -> Node:
    return node(
        func=predict_xgboost_model,
        inputs=dict(
            model="xgboost_model",
            X_test="model_inputs_trees_X_test",
        ),
        outputs="xgboost_preds",
        tags=["test"],
        name="predict_xgboost_model",
    )


def create_pipeline(**_) -> Pipeline:
    core_nodes = [
        _optimize_xgboost_params_node(),
        _train_xgboost_model_node(),
        _predict_xgboost_model_node(),
    ]
    return pipeline(core_nodes)
