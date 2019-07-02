"""Pipeline construction."""

from kedro.pipeline import Pipeline, node
from twitter_network.nodes.setup import *
from twitter_network.nodes.features import *
from twitter_network.nodes.models import *
from kedro.pipeline.decorators import log_time

# Create modelling pipeline
def create_pipeline(**kwargs):
    """Create the project's pipeline.

    Args:
        kwargs: Ignore any additional arguments added in the future.

    Returns:
        Pipeline: The resulting pipeline.

    """

    ####################################################################################
    # 01: Setup
    ####################################################################################
    setup = Pipeline(
        [
            # Create undirected graph from adjacency list
            node(
                create_train_graph,
                inputs="adj_train_full",
                outputs="G_train",
                tags=["graph", "undirected"],
            ),
            # Create directed graph from adjacency list
            node(
                create_train_digraph,
                inputs="adj_train_full",
                outputs="DiG_train",
                tags=["graph", "directed"],
            ),
            # Extract test edges from public test data
            node(
                get_test_edges,
                inputs="test_public",
                outputs="edges_test",
                tags=["edges", "edges_test"],
            ),
            # Extract edge list from directed graph
            node(
                get_all_edges,
                inputs="DiG_train",
                outputs="edges_all",
                tags=["edges", "edges_all"],
            ),
        ],
        name="setup",
    )

    ####################################################################################
    # 02: Preprocessing
    ####################################################################################
    SAMPLE = "train_master"

    preprocessing = Pipeline(
        [
            # # Preprocess sample from CSV
            # node(
            #     preprocess_sample,
            #     inputs=dict(df=SAMPLE, test="edges_test"),
            #     outputs="sample",
            #     tags=["sample"],
            # ),
            # Create undirected subgraph by removing sample edges
            node(
                create_subgraph,
                inputs=dict(G="G_train", edges=SAMPLE),
                outputs="subG_train",
                tags=["sub", "subG"],
            ),
            # Create directed subgraph by removing sample edges
            node(
                create_subgraph,
                inputs=dict(G="DiG_train", edges=SAMPLE),
                outputs="subDiG_train",
                tags=["sub", "subDiG"],
            ),
            # Split sample into training and validation sets
            node(
                split_sample,
                inputs=dict(sample=SAMPLE, parameters="parameters"),
                outputs=["sample_train", "sample_valid"],
                tags=["sample", "split"],
            ),
            # Extract
            node(
                extract_classes,
                inputs="sample_train",
                outputs=dict(edges="edges_train", classes="y_train"),
                tags=["sample", "classes"],
            ),
            node(
                extract_classes,
                inputs="sample_valid",
                outputs=dict(edges="edges_valid", classes="y_valid"),
                tags=["sample", "classes"],
            ),
        ],
        name="preprocess",
    ).decorate(log_time)

    ####################################################################################
    # 03: Feature engineering
    ####################################################################################
    features = Pipeline(
        [
            node(
                compile_features,
                inputs=dict(
                    G="subG_train",
                    DiG="subDiG_train",
                    edges_train="edges_train",
                    edges_valid="edges_valid",
                    edges_test="edges_test",
                    page_rank="page_rank",
                    katz="katz",
                    parameters="parameters",
                ),
                outputs=["non_stand_X_train", "non_stand_X_valid", "non_stand_X_test"],
                tags=["compile"],
            ),
            node(
                transform_features,
                inputs=[
                    "non_stand_X_train",
                    "non_stand_X_valid",
                    "non_stand_X_test",
                    "parameters",
                ],
                outputs=[
                    "non_stand_log_X_train",
                    "non_stand_log_X_valid",
                    "non_stand_log_X_test",
                ],
                tags=["transform", "log"],
            ),
            node(
                rescale_features,
                inputs=[
                    "non_stand_log_X_train",
                    "non_stand_log_X_valid",
                    "non_stand_log_X_test",
                ],
                outputs=["stand_log_X_train", "stand_log_X_valid", "stand_log_X_test"],
                tags=["transform", "scale"],
            ),
            node(
                select_features,
                inputs=[
                    "stand_log_X_train",
                    "stand_log_X_valid",
                    "stand_log_X_test",
                    "y_train",
                    "parameters",
                ],
                outputs=dict(
                    X_train_reduced="X_train",
                    X_valid_reduced="X_valid",
                    X_test_reduced="X_test",
                    included="included_features",
                    excluded="excluded_features",
                ),
                tags=["select"],
            ),
        ],
        name="features",
    ).decorate(log_time)

    ####################################################################################
    # 04: Model training & evaluation
    ####################################################################################
    models = Pipeline(
        [
            node(
                train_Jacob,
                inputs=dict(parameters="parameters"),
                outputs="clf_Jacob",
                tags=["clf", "clf_Jacob"],
            ),
            node(
                train_NB,
                inputs=dict(
                    X_train="X_train", y_train="y_train", parameters="parameters"
                ),
                outputs="clf_NB",
                tags=["clf", "clf_NB"],
            ),
            node(
                train_LR,
                inputs=dict(
                    X_train="X_train", y_train="y_train", parameters="parameters"
                ),
                outputs="clf_LR",
                tags=["clf", "clf_LR"],
            ),
            # node(
            #     train_SVM,
            #     inputs=dict(
            #         X_train="X_train", y_train="y_train", parameters="parameters"
            #     ),
            #     outputs="clf_SVM",
            #     tags=["clf", "clf_SVM"],
            # ),
            node(
                train_RF,
                inputs=dict(
                    X_train="X_train", y_train="y_train", parameters="parameters"
                ),
                outputs="clf_RF",
                tags=["clf", "clf_RF"],
            ),
            node(
                train_ET,
                inputs=dict(
                    X_train="X_train", y_train="y_train", parameters="parameters"
                ),
                outputs="clf_ET",
                tags=["clf", "clf_ET"],
            ),
            node(
                train_GB,
                inputs=dict(
                    X_train="X_train", y_train="y_train", parameters="parameters"
                ),
                outputs="clf_GB",
                tags=["clf", "clf_GB"],
            ),
            node(
                train_IF,
                inputs=dict(
                    X_train="X_train", y_train="y_train", parameters="parameters"
                ),
                outputs="clf_IF",
                tags=["clf", "clf_IF"],
            ),
            node(
                train_HGB,
                inputs=dict(
                    X_train="X_train", y_train="y_train", parameters="parameters"
                ),
                outputs="clf_HGB",
                tags=["clf", "clf_HGB"],
            ),
            node(
                train_AB,
                inputs=dict(
                    X_train="X_train", y_train="y_train", parameters="parameters"
                ),
                outputs="clf_AB",
                tags=["clf", "clf_AB"],
            ),
            node(
                train_XG,
                inputs=dict(
                    X_train="X_train", y_train="y_train", parameters="parameters"
                ),
                outputs="clf_XG",
                tags=["clf", "clf_XG"],
            ),
            node(
                train_NN,
                inputs=dict(
                    X_train="X_train", y_train="y_train", parameters="parameters"
                ),
                outputs="clf_NN",
                tags=["clf", "clf_NN"],
            ),
            # node(
            #     train_voting,
            #     inputs=[
            #         # Training data
            #         "X_train",
            #         "y_train",
            #         # Panel members
            #         "clf_LR",
            #         "clf_SVM",
            #         "clf_RF",
            #         "clf_ET",
            #         "clf_HGB",
            #         "clf_AB",
            #         "clf_XG",
            #     ],
            #     outputs=["vote_hard", "vote_soft"],
            #     tags=["clf", "meta", "vote"],
            # ),
            node(
                evaluate_models,
                inputs=[
                    ## Validation data
                    "X_valid",
                    "y_valid",
                    ## Non-standardised
                    "non_stand_X_valid",
                    # ## Models
                    # "clf_Jacob",
                    # "clf_NB",
                    # "clf_LR",
                    # # "clf_SVM",
                    # "clf_RF",
                    # "clf_ET",
                    # "clf_GB",
                    # # "clf_IF",
                    "clf_HGB",
                    # "clf_AB",
                    # "clf_XG",
                    # "clf_NN",
                    # # "vote_hard",
                    # # "vote_soft",
                ],
                outputs=None,
                tags=["eval"],
            ),
        ],
        name="models",
    ).decorate(log_time)

    ####################################################################################
    # 05: Link prediction
    ####################################################################################
    CHAMPION = "clf_HGB"

    predictions = Pipeline(
        [
            node(
                make_predictions,
                inputs=dict(
                    clf=CHAMPION,
                    edges_test="edges_test",
                    X_test="X_test",
                    non_stand_X_test="non_stand_X_test",
                    parameters="parameters",
                ),
                outputs="predictions",
            )
        ],
        name="predict",
    ).decorate(log_time)

    return setup + preprocessing + features + models + predictions
