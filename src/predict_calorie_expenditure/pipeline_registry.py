from typing import Dict
from kedro.pipeline import Pipeline

from predict_calorie_expenditure.pipelines import (
    eda as eda_pipeline,
    data_preprocessing as data_preprocessing_pipeline,
    feature_engineering as feature_engineering_pipeline,
    feature_selection as feature_selection_pipeline,
)
from predict_calorie_expenditure.pipelines.model_inputs import (
    trees as model_inputs_trees_pipeline,
)
from predict_calorie_expenditure.pipelines.model_training import (
    xgboost as model_training_xbgoost_pipeline,
)


def register_pipelines() -> Dict[str, Pipeline]:
    return {
        "eda": eda_pipeline.create_pipeline(),
        "data_preprocessing": data_preprocessing_pipeline.create_pipeline(),
        "feature_engineering": feature_engineering_pipeline.create_pipeline(),
        "feature_selection": feature_selection_pipeline.create_pipeline(),
        "full_preprocessing": (
            eda_pipeline.create_pipeline()
            + data_preprocessing_pipeline.create_pipeline()
            + feature_engineering_pipeline.create_pipeline()
            + feature_selection_pipeline.create_pipeline()
        ),
        "full_xboost": (
            eda_pipeline.create_pipeline()
            + data_preprocessing_pipeline.create_pipeline()
            + feature_engineering_pipeline.create_pipeline()
            + feature_selection_pipeline.create_pipeline()
            + model_inputs_trees_pipeline.create_pipeline()
            + model_training_xbgoost_pipeline.create_pipeline()
        ),
        "__default__": eda_pipeline.create_pipeline(),
    }
