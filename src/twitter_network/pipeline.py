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
    SAMPLE = "train_100"

    setup = Pipeline(
        [
            ## NOT RUN
            # Create undirected graph from adjacency list
            # node(
            #     create_train_graph,
            #     inputs="adj_train_full",
            #     outputs="G_train",
            #     tags=["graph", "undirected"],
            # ),
            # Create directed graph from adjacency list
            # node(
            #     create_train_digraph,
            #     inputs="adj_train_full",
            #     outputs="DiG_train",
            #     tags=["graph", "directed"],
            # ),
            # Extract test edges from public test data
            node(
                get_test_edges,
                inputs="test_public",
                outputs="edges_test",
                tags=["sample", "test"],
            ),
            # Preprocess the sample files, converting from CSV to Pandas dataframe
            node(
                preprocess_sample,
                inputs=SAMPLE,
                outputs="sample",
                tags=["sample", "import"],
            ),
            # Split sample into training and validation sets
            node(
                split_sample,
                inputs=dict(sample="sample", parameters="parameters"),
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
        name="setup",
    ).decorate(log_time)

    ####################################################################################
    # 02: Feature generation
    ####################################################################################
    features = Pipeline(
        [
            node(
                calculate_similarity,
                inputs=dict(
                    G="G_train",
                    edges_train="edges_train",
                    edges_valid="edges_valid",
                    edges_test="edges_test",
                ),
                outputs=["non_stand_X_train", "non_stand_X_valid", "non_stand_X_test"],
                tags=["similarity"],
            ),
            node(
                rescale_features,
                inputs=["non_stand_X_train", "non_stand_X_valid", "non_stand_X_test"],
                outputs=["X_train", "X_valid", "X_test"],
                tags=["scale"],
            ),
        ],
        name="features",
    ).decorate(log_time)

    ####################################################################################
    # 03: Modelling
    ####################################################################################
    models = Pipeline(
        [
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
            #         "clf_AB",
            #         "clf_XG",
            #     ],
            #     outputs=["vote_hard", "vote_soft"],
            #     tags=["clf", "meta", "vote"],
            # ),
            node(
                evaluate_models,
                inputs=[
                    # Test data
                    "X_valid",
                    "y_valid",
                    # Models
                    "clf_NB",
                    "clf_LR",
                    # "clf_SVM",
                    "clf_RF",
                    "clf_AB",
                    "clf_XG",
                    "clf_NN",
                    # "vote_hard",
                    # "vote_soft",
                ],
                outputs=None,
                tags=["eval"],
            ),
        ],
        name="models",
    ).decorate(log_time)

    ####################################################################################
    # 04: Prediction
    ####################################################################################
    CHAMPION = "clf_XG"

    predictions = Pipeline(
        [
            node(
                make_predictions,
                inputs=dict(
                    clf=CHAMPION,
                    edges_test="edges_test",
                    X_test="X_test",
                    parameters="parameters",
                ),
                outputs="predictions",
            )
        ],
        name="predict",
    ).decorate(log_time)

    return setup + features + models + predictions
