from kedro.pipeline import Pipeline, node, pipeline
from kedro.pipeline.node import Node
from predict_calorie_expenditure.pipelines.model_inputs.trees.nodes import (
    fit_preprocessor_and_transform_train,
    transform_with_preprocessor,
)
from predict_calorie_expenditure.pipelines.eda.pipeline import (
    generate_profile_report_node,
)


def _fit_and_transform_train_node() -> Node:
    return node(
        func=fit_preprocessor_and_transform_train,
        inputs=dict(
            train="feature_train",
            feature_ord_levels="params:feature_ord_levels",
            target_column="params:target_column",
        ),
        outputs=[
            "model_inputs_trees_X_train",
            "model_inputs_trees_y_train",
            "model_inputs_trees_preprocessor",
        ],
        tags=["train"],
        name="fit_preprocessor_and_transform_train",
    )


def _transform_test_node() -> Node:
    return node(
        func=transform_with_preprocessor,
        inputs=dict(
            df="feature_test",
            preprocessor="model_inputs_trees_preprocessor",
            target_column="params:target_column",
        ),
        outputs="model_inputs_trees_X_test",
        tags=["test"],
        name="transform_test_with_preprocessor",
    )


def create_pipeline(**_) -> Pipeline:
    output_path = "params:report_path_model_inputs_trees"

    core_nodes = [
        _fit_and_transform_train_node(),
        _transform_test_node(),
        *[
            generate_profile_report_node(
                data_part="model_inputs_trees_X",
                split=split,
                output_path=output_path,
                stage="model_inputs",
            )
            for split in ("train", "test")
        ],
    ]
    return pipeline(core_nodes)
