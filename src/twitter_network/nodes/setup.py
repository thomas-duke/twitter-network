"""Network graph setup."""

import pandas as pd
import numpy as np
import networkx as nx
from networkx.classes.graph import Graph
from networkx.classes.digraph import DiGraph
from itertools import product
from copy import deepcopy
from sklearn.model_selection import train_test_split
from ast import literal_eval


def create_train_graph(targets: dict) -> Graph:
    """Initialise an undirected NetworkX graph from an adjacency list.

        Args:
            targets: training adjacency list as a dictionary, with source nodes as keys
            and target nodes as values.

        Returns:
            A NetworkX graph object.

    """
    sources = list(targets.keys())
    G = nx.Graph()
    for s in sources:
        edges = list(product([s], targets[s]))
        G.add_edges_from(edges)
    return G


def create_train_digraph(targets: dict) -> DiGraph:
    """Initialise an directed NetworkX graph from an adjacency list.

        Args:
            targets: training adjacency list as a dictionary, with source nodes as keys
            and target nodes as values.

        Returns:
            A NetworkX graph object.

    """
    sources = list(targets.keys())
    G = nx.DiGraph()
    for s in sources:
        edges = list(product([s], targets[s]))
        G.add_edges_from(edges)
    return G


def get_test_edges(edges: pd.DataFrame) -> list:
    """Extracts edges from test dataset.

        Args:
            edges: a Pandas dataframe with three columns (Id, Source and Sink), where:
                - Id is a unique identifier for the edge.
                - Source is the source node of the edge.
                - Sink is the target node for the edge.

        Returns:
            A list of (Source, Sink) tuples representing the edges.

    """
    return [(str(x.Source), str(x.Sink)) for x in edges.itertuples()]


def preprocess_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Import a training sample and format it for data processing.

    Args:
        Pandas dataframe with two columns:
            label: a binary variable taking the value 1 if the edge was hidden and 0
            if it was fake.
            edge: the edge as a tuple in the form (source, sink).

    Returns:
        Pre-processed Pandas dataframe.

    """
    df = df.sample(frac=1)
    df = df.rename(columns={"nodes": "edge"})
    df.edge = [literal_eval(x) for x in df.edge]
    df.edge = [(str(u), str(v)) for u, v in df.edge]
    return df


def split_sample(sample: pd.DataFrame, parameters: dict) -> list:
    """Splits sample data into training and validation sets.

    Args:
        sample: Pandas dataframe containing sample edges and class labels.

    Returns:
        A list containing split data.

    """
    return train_test_split(sample, test_size=parameters["sample"]["valid_size"])


def extract_classes(classes: pd.DataFrame) -> dict:
    """Extracts edges and class labels from labelled training data.

    Args:
        classes: Pandas dataframe with two columns:
            edge: the edge as a tuple in the form (source, sink).
            valid: a binary variable taking the value 1 if the edge was hidden and 0
            if it was fake.

    Returns:
        A dictionary containing lists for the edges and class labels.
    """
    return dict(edges=list(classes.edge), classes=classes.label)
