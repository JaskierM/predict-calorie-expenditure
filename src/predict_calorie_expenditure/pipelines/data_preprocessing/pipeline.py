from kedro.pipeline import Pipeline, node, pipeline
from kedro.pipeline.node import Node

from predict_calorie_expenditure.pipelines.eda.pipeline import (
    generate_profile_report_node,
    compute_mi_scores_plot_node,
)
from predict_calorie_expenditure.pipelines.data_preprocessing.nodes import (
    preprocess_data,
)


def _preprocess_data_node(split: str) -> Node:
    return node(
        func=preprocess_data,
        inputs=dict(
            df=f"raw_{split}",
            drop_duplicates=f"params:drop_duplicates_{split}",
            id_column="params:id_column",
            features_cat="params:features_cat",
            features_date="params:features_date",
            datetime_format="params:datetime_format",
        ),
        outputs=f"intermediate_{split}",
        tags=[split],
        name=f"preprocess_data_{split}",
    )


def create_pipeline(**_) -> Pipeline:
    splits = ("train", "test")
    output_path = "params:report_path_data_preprocessing"

    core_nodes = (
        [_preprocess_data_node(split=split) for split in splits]
        + [
            generate_profile_report_node(
                data_part="intermediate",
                split=split,
                output_path=output_path,
                stage="data_preprocessing",
            )
            for split in splits
        ]
        + [
            compute_mi_scores_plot_node(
                data_part="intermediate",
                output_path=output_path,
                stage="data_preprocessing",
            )
        ]
    )
    return pipeline(core_nodes)
