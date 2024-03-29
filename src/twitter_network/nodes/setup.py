"""Network graph setup."""

import logging
import pandas as pd
import numpy as np
import networkx as nx
from networkx.classes.graph import Graph
from networkx.classes.digraph import DiGraph
from typing import Union
import random
from itertools import product
from copy import deepcopy
from sklearn.model_selection import train_test_split
from ast import literal_eval

# Logging text colors
def red(text: str):
    return u"\u001b[31m{}\u001b[0m".format(text)


def blue(text: str):
    return u"\u001b[34m{}\u001b[0m".format(text)


# Graph initialisation
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


# Edge extraction
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


def get_all_edges(G: Union[Graph, DiGraph]) -> list:
    """Extracts all edges from a NetworkX graph..

        Args:
            G: a NetworkX undirected or directed graph object.

        Returns:
            A list of (Source, Sink) tuples representing the edges.

    """
    return list(G.edges)


def create_subgraph(
    G: Union[Graph, DiGraph], edges: Union[list, pd.DataFrame]
) -> Union[Graph, DiGraph]:
    """Create a subgraph of a NetworkX graph by hiding a list of edges.

    Args:
        G: a NetworkX undirected or directed graph object.

        edges: a list of edges to be hidden, OR a Pandas dataframe containing a column
        labelled "edge", which will form the edge list.

    Returns:
        The corresponding subgraph.

    """
    if type(edges) == pd.DataFrame:
        edges = edges.edge
    subG = deepcopy(G)
    subG.remove_edges_from(edges)
    return subG


# Sample generation
def hide_edges(G: Union[Graph, DiGraph], edges: list, parameters: dict) -> dict:
    """Remove a random subset of edges from a NetworkX graph.

        Args:
            G: a NetworkX undirected or directed graph object.

            edges: a list of edges, of which a random subset will be hidden.

            parameters: parameters defined in parameters.yml.

        Returns:
            A dictionary containing:
                subG: the subgraph of G generated by hiding edges.
                hidden: list of hidden edges.

    """
    # Generate a random list of edges to hide
    random.seed(parameters["seed"])
    hidden = random.sample(edges, k=parameters["setup"]["N_hidden"])

    # Create a subgraph by removing edges
    subG = create_subgraph(G, hidden)

    return dict(subG=subG, hidden=hidden)


def fake_edges(
    targets: dict, subG: Graph, hidden: list, test: list, parameters: dict
) -> list:
    """Randomly generates fake edges from source nodes to target nodes in a subgraph.

        Args:
            targets: training adjacency list as a dictionary, with source nodes as keys
            and target nodes as values.

            subG: a NetworkX graph object, which is a subgraph of the training network.

            hidden: list of hidden edges.

            test: list of edges in the test set.

            parameters: parameters defined in parameters.yml.

        Returns:
            List of source nodes.

    """
    fakes = []
    sources = list(targets.keys())
    targets = subG.nodes
    np.random.seed(parameters["seed"])
    while len(fakes) < parameters["setup"]["N_fake"]:
        s = np.random.choice(sources)
        t = np.random.choice(targets)
        if (
            s != t  # no self loops
            and not subG.has_edge(s, t)  # edge does not exist in subgraph
            and not (s, t) in hidden  # edge has not been hidden
            and not (s, t) in test  # edge is not in the test set
            and not (s, t) in fakes  # edge has not already been created
        ):
            fakes.append((s, t))
    return fakes


def create_sample(hidden: list, fakes: list) -> pd.DataFrame:
    """Creates a Pandas DataFrame with a random ordering of hidden edges and fake edges.

    Args:
        hidden: list of hidden edges.

        fakes:  list of fake edges.

    Returns:
        Pandas dataframe with two columns:
            edge: the edge as a tuple in the form (source, sink).
            label: a binary variable taking the value 1 if the edge was hidden and 0
            if it was fake.

    """
    sample = [(x, 1) for x in hidden] + [(x, 0) for x in fakes]
    np.random.shuffle(sample)
    return pd.DataFrame(sample, columns=["edge", "label"])


def self_loop(edge):
    source, sink = edge
    return int(source == sink)


def preprocess_sample(df: pd.DataFrame, test: list) -> pd.DataFrame:
    """Import a training sample and format it for data processing.

    Args:
        Pandas dataframe with two columns:
            label: a binary variable taking the value 1 if the edge was hidden and 0
            if it was fake.
            edge: the edge as a tuple in the form (source, sink).

        test: list of test edges.

    Returns:
        Pre-processed Pandas dataframe.

    """
    log = logging.getLogger(__name__)

    # Randomly shuffle the sample and fix formatting
    df = df.sample(frac=1)
    df = df.rename(columns={"nodes": "edge"})
    df.edge = [literal_eval(x) for x in df.edge]
    df.edge = [(str(u), str(v)) for u, v in df.edge]

    # Remove test edges
    df["test"] = df.edge.apply(lambda x: x in test)
    if any(df["test"]):
        log.warning(
            red("Dropping {} test edge(s) from sample.").format(sum(df["test"]))
        )
    df = df[df.test == False]

    # Drop duplicates
    df_nodups = df.drop_duplicates()
    n_dups = len(df) - len(df_nodups)
    if n_dups > 0:
        log.warning(red("Dropping {} duplicate(s) from sample.").format(n_dups))
        df = df_nodups

    # Check for self loops
    df["self_loop"] = df.edge.apply(self_loop)
    n_loops = sum(df.self_loop)
    if n_loops:
        log.warning(red("Dropping {} self-loop(s) from sample.").format(n_loops))
        df = df[df.self_loop == 0]
        df = df.drop("self_loop", axis=1)

    return df[["edge", "label"]]


def split_sample(sample: pd.DataFrame, parameters: dict) -> list:
    """Splits sample data into training and validation sets.

    Args:
        sample: Pandas dataframe containing sample edges and class labels.

    Returns:
        A list containing split data.

    """
    if parameters["setup"]["train_size"]:
        return train_test_split(sample, train_size=parameters["setup"]["train_size"])
    elif parameters["setup"]["valid_size"]:
        return train_test_split(sample, test_size=parameters["setup"]["valid_size"])


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
