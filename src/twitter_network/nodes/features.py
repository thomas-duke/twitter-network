"""Edge feature creation."""

import pandas as pd
import numpy as np
import networkx as nx
from collections import defaultdict
from networkx.classes.graph import Graph
from networkx.algorithms import link_prediction as lp
from sklearn.preprocessing import StandardScaler


def local_similarity(G, edges) -> pd.DataFrame:
    """Calculates local similarity measures for a list of edges on an undirected graph.

    Args:
        G: a NetworkX graph object.

        edges: a list of edges.

    Returns:
        Pandas dataframe with the following similarity statistics for each edge:
            ra: Resource Allocation index
            jc: Jaccard Coefficient
            aa: Adamic Adar index
            pa: Preferential Attachment
    """

    return pd.DataFrame(
        {
            "ra": [x for u, v, x in lp.resource_allocation_index(G, edges)],
            "jc": [x for u, v, x in lp.jaccard_coefficient(G, edges)],
            # "aa": [x for u, v, x in lp.adamic_adar_index(G, edges)],
            "pa": [x for u, v, x in lp.preferential_attachment(G, edges)],
        }
    )


def calculate_similarity(
    G: Graph, edges_train: list, edges_valid: list, edges_test: list
) -> pd.DataFrame:
    """Calculates local similarity statistics for training, validation and test sets.

    Args:
        G: a NetworkX graph object.

        edges_train: list of training edges.

        edges_valid: list of validation edges.

        edges_test: list of test edges.

    Returns:
        A list of Pandas dataframes with the statistics for each list of edges.
    """
    return [
        local_similarity(G, edges) for edges in [edges_train, edges_valid, edges_test]
    ]


def rescale_features(
    X_train: pd.DataFrame, X_valid: pd.DataFrame, X_test: pd.DataFrame
) -> list:
    """Standardises the scale of the training and test features using the mean and
    standard deviation of the training data.

    Args:
        X_train: training data.

        X_test: test data.

    Returns:
        A list containing the standardised training and test data.
    """
    non_stand_X_train = X_train
    non_stand_X_valid = X_valid
    non_stand_X_test = X_test
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test = scaler.transform(X_test)
    return [X_train, X_valid, X_test]
