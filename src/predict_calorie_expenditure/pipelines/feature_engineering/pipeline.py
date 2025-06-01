from kedro.pipeline import Pipeline, node, pipeline
from kedro.pipeline.node import Node

from predict_calorie_expenditure.pipelines.feature_engineering.nodes import (
    create_features,
)
from predict_calorie_expenditure.pipelines.eda.pipeline import (
    generate_profile_report_node,
    compute_mi_scores_plot_node,
)


def _create_features_node(split: str) -> Node:
    return node(
        func=create_features,
        inputs=dict(
            df=f"intermediate_{split}",
            use_binning="params:use_binning",
            use_transformations="params:use_transformations",
            use_interactions="params:use_interactions",
            use_logs="params:use_logs",
        ),
        outputs=f"primary_{split}",
        tags=[split],
        name=f"create_features_{split}",
    )


def create_pipeline(**_) -> Pipeline:
    splits = ("train", "test")
    output_path = "params:report_path_feature_engineering"

    core_nodes = (
        [_create_features_node(split=split) for split in splits]
        + [
            generate_profile_report_node(
                data_part="primary",
                split=split,
                output_path=output_path,
                stage="feature_engineering",
            )
            for split in splits
        ]
        + [
            compute_mi_scores_plot_node(
                data_part="primary",
                output_path=output_path,
                stage="feature_engineering",
            )
        ]
    )
    return pipeline(core_nodes)
