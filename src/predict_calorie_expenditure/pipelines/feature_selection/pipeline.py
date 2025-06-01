from kedro.pipeline import Pipeline, node, pipeline
from kedro.pipeline.node import Node

from predict_calorie_expenditure.pipelines.feature_selection.nodes import (
    select_and_preprocess_features,
)
from predict_calorie_expenditure.pipelines.eda.pipeline import (
    generate_profile_report_node,
    compute_mi_scores_plot_node,
)


def _select_and_preprocess_features_node(split: str) -> Node:
    return node(
        func=select_and_preprocess_features,
        inputs=dict(
            df=f"primary_{split}",
            features_to_drop="params:features_to_drop",
            feature_outlier_thresholds="params:feature_outlier_thresholds",
            drop_duplicates=f"params:drop_duplicates_{split}",
        ),
        outputs=f"feature_{split}",
        tags=[split],
        name=f"select_and_preprocess_features_{split}",
    )


def create_pipeline(**_) -> Pipeline:
    splits = ("train", "test")
    output_path = "params:report_path_feature_selection"

    core_nodes = (
        [_select_and_preprocess_features_node(split=split) for split in splits]
        + [
            generate_profile_report_node(
                data_part="feature",
                split=split,
                output_path=output_path,
                stage="feature_selection",
            )
            for split in splits
        ]
        + [
            compute_mi_scores_plot_node(
                data_part="feature",
                output_path=output_path,
                stage="feature_selection",
            )
        ]
    )
    return pipeline(core_nodes)
