"""Pipeline construction."""

from kedro.pipeline import Pipeline, node
from twitter_network.nodes.setup import *
from twitter_network.nodes.features import *
from twitter_network.nodes.models import *
from kedro.pipeline.decorators import log_time


def create_pipeline(**kwargs):
    """Create the project's pipeline.

    Args:
        kwargs: Ignore any additional arguments added in the future.

    Returns:
        Pipeline: The resulting pipeline.

    """

    # Data setup and pre-processing

    setup1 = Pipeline(
        [
            node(
                create_train_graph,
                inputs="adj_train",
                outputs="G_train",
                tags=["graph"],
            ),
            node(
                create_train_digraph,
                inputs="adj_train",
                outputs="DiG_train",
                tags=["digraph"],
            ),
            node(
                get_test_edges,
                inputs="test_public",
                outputs="edges_test",
                tags=["edges", "edges_test"],
            ),
            node(
                hide_edges,
                inputs=dict(G="G_train", targets="adj_train", parameters="parameters"),
                outputs=dict(subG="subG_train", hidden="edges_hidden"),
                tags=["edges", "edges_hide"],
            ),
            node(
                generate_fake_edges,
                inputs=dict(
                    targets="adj_train",
                    subG="subG_train",
                    hidden="edges_hidden",
                    test="edges_test",
                    parameters="parameters",
                ),
                outputs="edges_fake",
                tags=["edges", "edges_fake"],
            ),
            node(
                create_sample,
                inputs=dict(hidden="edges_hidden", fakes="edges_fake"),
                outputs="sample",
                tags=["edges", "sample"],
            ),
            node(
                split_sample,
                inputs=dict(sample="sample", parameters="parameters"),
                outputs=["sample_train", "sample_valid"],
                tags=["edges", "sample"],
            ),
            node(
                extract_classes,
                inputs="sample_train",
                outputs=dict(edges="edges_train", classes="y_train"),
                tags=["edges", "sample"],
            ),
            node(
                extract_classes,
                inputs="sample_valid",
                outputs=dict(edges="edges_valid", classes="y_valid"),
                tags=["edges", "sample"],
            ),
        ],
        name="setup1",
    )

    setup2 = Pipeline(
        [
            node(
                create_sample2,
                inputs="small_train",
                outputs="sample2",
                tags=["edges", "sample2"],
            ),
            node(
                split_sample,
                inputs=dict(sample="sample2", parameters="parameters"),
                outputs=["sample_train2", "sample_valid2"],
                tags=["edges", "sample2"],
            ),
            node(
                extract_classes,
                inputs="sample_train2",
                outputs=dict(edges="edges_train2", classes="y_train2"),
                tags=["edges", "sample2"],
            ),
            node(
                extract_classes,
                inputs="sample_valid2",
                outputs=dict(edges="edges_valid2", classes="y_valid2"),
                tags=["edges", "sample2"],
            ),
        ],
        name="setup2",
    )

    setup = Pipeline([setup1, setup2], name="setup").decorate(log_time)

    # Feature generation

    features1 = Pipeline(
        [
            node(
                calculate_similarity,
                inputs=dict(
                    G="subG_train",
                    edges_train="edges_train",
                    edges_valid="edges_valid",
                    edges_test="edges_test",
                ),
                outputs=["non_stand_X_train", "non_stand_X_valid", "non_stand_X_test"],
                tags=["feat", "similarity"],
            ),
            node(
                rescale_features,
                inputs=["non_stand_X_train", "non_stand_X_valid", "non_stand_X_test"],
                outputs=["X_train", "X_valid", "X_test"],
                tags=["feat", "scale"],
            ),
        ],
        name="features1",
    )

    features2 = Pipeline(
        [
            node(
                calculate_similarity,
                inputs=dict(
                    G="G_train",
                    edges_train="edges_train2",
                    edges_valid="edges_valid2",
                    edges_test="edges_test",
                ),
                outputs=[
                    "non_stand_X_train2",
                    "non_stand_X_valid2",
                    "non_stand_X_test2",
                ],
                tags=["feat2", "similarity2"],
            ),
            node(
                rescale_features,
                inputs=[
                    "non_stand_X_train2",
                    "non_stand_X_valid2",
                    "non_stand_X_test2",
                ],
                outputs=["X_train2", "X_valid2", "X_test2"],
                tags=["feat2", "scale2"],
            ),
        ],
        name="features2",
    )

    features = Pipeline([features1, features2], name="features").decorate(log_time)

    # Model fitting and evaluation

    models1 = Pipeline(
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
            node(
                train_SVM,
                inputs=dict(
                    X_train="X_train", y_train="y_train", parameters="parameters"
                ),
                outputs="clf_SVM",
                tags=["clf", "clf_SVM"],
            ),
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
            node(
                train_voting,
                inputs=[
                    # Training data
                    "X_train",
                    "y_train",
                    # Panel members
                    "clf_LR",
                    "clf_SVM",
                    "clf_RF",
                    "clf_AB",
                    "clf_XG",
                ],
                outputs=["vote_hard", "vote_soft"],
                tags=["clf", "meta", "vote"],
            ),
            node(
                evaluate_models,
                inputs=[
                    # Test data
                    "X_valid",
                    "y_valid",
                    # Models
                    "clf_NB",
                    "clf_LR",
                    "clf_SVM",
                    "clf_RF",
                    "clf_AB",
                    "clf_XG",
                    "clf_NN",
                    "vote_hard",
                    "vote_soft",
                ],
                outputs=None,
                tags=["eval"],
            ),
        ],
        name="models1",
    )

    models2 = Pipeline(
        [
            node(
                train_NB,
                inputs=dict(
                    X_train="X_train2", y_train="y_train2", parameters="parameters"
                ),
                outputs="clf_NB2",
                tags=["clf", "clf_NB2"],
            ),
            node(
                train_LR,
                inputs=dict(
                    X_train="X_train2", y_train="y_train2", parameters="parameters"
                ),
                outputs="clf_LR2",
                tags=["clf", "clf_LR2"],
            ),
            node(
                train_SVM,
                inputs=dict(
                    X_train="X_train2", y_train="y_train2", parameters="parameters"
                ),
                outputs="clf_SVM2",
                tags=["clf", "clf_SVM2"],
            ),
            node(
                train_RF,
                inputs=dict(
                    X_train="X_train2", y_train="y_train2", parameters="parameters"
                ),
                outputs="clf_RF2",
                tags=["clf", "clf_RF2"],
            ),
            node(
                train_AB,
                inputs=dict(
                    X_train="X_train2", y_train="y_train2", parameters="parameters"
                ),
                outputs="clf_AB2",
                tags=["clf", "clf_AB2"],
            ),
            node(
                train_XG,
                inputs=dict(
                    X_train="X_train2", y_train="y_train2", parameters="parameters"
                ),
                outputs="clf_XG2",
                tags=["clf", "clf_XG2"],
            ),
            node(
                train_NN,
                inputs=dict(
                    X_train="X_train2", y_train="y_train2", parameters="parameters"
                ),
                outputs="clf_NN2",
                tags=["clf", "clf_NN2"],
            ),
            node(
                train_voting,
                inputs=[
                    # Training data
                    "X_train2",
                    "y_train2",
                    # Panel members
                    "clf_LR2",
                    "clf_SVM2",
                    "clf_RF2",
                    "clf_AB2",
                    "clf_XG2",
                ],
                outputs=["vote_hard2", "vote_soft2"],
                tags=["clf", "meta", "vote2"],
            ),
            node(
                evaluate_models,
                inputs=[
                    # Test data
                    "X_valid2",
                    "y_valid2",
                    # Models
                    "clf_NB2",
                    "clf_LR2",
                    "clf_SVM2",
                    "clf_RF2",
                    "clf_AB2",
                    "clf_XG2",
                    "clf_NN2",
                    "vote_hard2",
                    "vote_soft2",
                ],
                outputs=None,
                tags=["eval2"],
            ),
        ],
        name="models2",
    )

    models = Pipeline([models1, models2], name="models").decorate(log_time)

    # Test predictions

    predictions1 = Pipeline(
        [
            node(
                make_predictions,
                inputs=dict(
                    clf="clf_XG",
                    edges_test="edges_test",
                    X_test="X_test",
                    parameters="parameters",
                ),
                outputs="predictions",
            )
        ],
        name="predict1",
    )

    predictions2 = Pipeline(
        [
            node(
                make_predictions,
                inputs=dict(
                    clf="vote_soft2",
                    edges_test="edges_test",
                    X_test="X_test2",
                    parameters="parameters",
                ),
                outputs="predictions2",
            )
        ],
        name="predict2",
    )

    predictions = Pipeline([predictions1, predictions2], name="predict")

    return setup + features + models + predictions
