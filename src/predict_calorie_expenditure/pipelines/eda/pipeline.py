from kedro.pipeline import Pipeline, node, pipeline
from kedro.pipeline.node import Node

from predict_calorie_expenditure.pipelines.eda.nodes import (
    generate_profile_report,
    compute_mi_scores_plot,
)


def generate_profile_report_node(
    data_part: str, split: str, output_path: str, stage: str
) -> Node:
    return node(
        func=generate_profile_report,
        inputs=dict(
            df=f"{data_part}_{split}",
            dataset_name=f"params:report_dataset_name_{split}",
            output_path=output_path,
            scale_factor="params:scale_factor_eda",
            random_state="params:random_state",
        ),
        outputs=None,
        tags=[f"report_{split}", "report"],
        name=f"generate_profile_report_{split}_{stage}",
    )


def compute_mi_scores_plot_node(data_part: str, output_path: str, stage: str) -> Node:
    return node(
        func=compute_mi_scores_plot,
        inputs=dict(
            df=f"{data_part}_train",
            target_column="params:target_column",
            output_path=output_path,
            scale_factor="params:scale_factor_eda",
            random_state="params:random_state",
        ),
        outputs=None,
        tags=["report_train", "report"],
        name=f"compute_mi_scores_plot_{stage}",
    )


def create_pipeline(**_) -> Pipeline:
    core_nodes = [
        generate_profile_report_node(
            data_part="raw",
            split=split,
            output_path="params:report_path_eda",
            stage="eda",
        )
        for split in ("train", "test")
    ]
    return pipeline(core_nodes)
