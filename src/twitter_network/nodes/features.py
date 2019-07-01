"""Edge feature creation."""

import logging
import gc
import math
import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
from collections import defaultdict
from networkx.classes.graph import Graph
from networkx.classes.digraph import DiGraph
from networkx.algorithms import link_prediction as lp
from networkx.algorithms.approximation.connectivity import local_node_connectivity
from networkx.algorithms.assortativity import *
from networkx.algorithms.boundary import *
from networkx.algorithms.centrality import *
from networkx.algorithms.cluster import *
from networkx.algorithms.distance_measures import *
from networkx.algorithms.efficiency import *
from networkx.algorithms.reciprocity import *
from networkx.algorithms.shortest_paths import *
from networkx.algorithms.structuralholes import *
from sklearn.preprocessing import StandardScaler, normalize
from twitter_network.nodes.setup import red, blue


def add_degree_features(G: DiGraph, df: pd.DataFrame) -> pd.DataFrame:
    source_in_degree = []
    source_out_degree = []
    source_bi_degree = []
    source_nbrs = []
    sink_in_degree = []
    sink_out_degree = []
    sink_nbrs = []
    sink_bi_degree = []
    cosine_sim = []
    for i, row in tqdm(df.iterrows()):
        source, sink = row["edge"]
        try:
            s_in = set(G.predecessors(source))
            s_out = set(G.successors(source))
            s_bi = set(s_in.intersection(s_out))
            s_nbrs = set(s_in.union(s_out))
        except:
            s_in = set()
            s_out = set()
            s_bi = set()
            s_nbrs = set()
        try:
            d_in = set(G.predecessors(sink))
            d_out = set(G.successors(sink))
            d_bi = set(d_in.intersection(d_out))
            d_nbrs = set(d_in.union(d_out))
        except:
            d_in = set()
            d_out = set()
            d_bi = set()
            d_nbrs = set()

        source_in_degree.append(len(s_in))
        source_out_degree.append(len(s_out))
        source_bi_degree.append(len(s_bi))
        source_nbrs.append(len(s_nbrs))

        sink_in_degree.append(len(d_in))
        sink_out_degree.append(len(d_out))
        sink_bi_degree.append(len(d_bi))
        sink_nbrs.append(len(d_nbrs))

    return pd.DataFrame(
        {
            "edge": df.edge,
            "source_in_degree": source_in_degree,
            "source_out_degree": source_out_degree,
            "source_bi_degree": source_bi_degree,
            "source_nbrs": source_nbrs,
            "sink_in_degree": sink_in_degree,
            "sink_out_degree": sink_out_degree,
            "sink_bi_degree": sink_out_degree,
            "sink_nbrs": sink_nbrs,
        }
    )


# Similarity indices
# def cosine_similarity(source_in, source_out, sink_in, sink_out):
#     try:
#         x1 = [source_in, source_out]
#         x2 = [sink_in, sink_out]
#         X = normalize([x1, x2])
#         return np.dot(X[0], X[1])
#     except:
#         return 0


def jaccard_coeff(source_neighbors, sink_neighbors):
    try:
        if len(source_neighbors) == 0 or len(sink_neighbors) == 0:
            return 0

        shared = source_neighbors.intersection(sink_neighbors)
        total = source_neighbors.union(sink_neighbors)

        jc = len(shared) / len(total)
        return jc
    except:
        return 0


def jc_followees(edge, G):
    source, sink = edge
    try:
        source_succesors = set(G.successors(source))
        sink_successors = set(G.successors(sink))
    except:
        return 0

    return jaccard_coeff(source_succesors, sink_successors)


def jc_followers(edge, G):
    source, sink = edge
    try:
        source_pred = set(G.predecessors(source))
        sink_pred = set(G.predecessors(sink))
    except:
        return 0

    return jaccard_coeff(source_pred, sink_pred)


def adjusted_adamic_adar_index(edge, G):
    source, sink = edge
    try:
        source_successors = set(G.successors(source))
        sink_successors = set(G.successors(sink))
    except:
        return 0
    sum = 0.0
    shared = source_successors.intersection(sink_successors)
    try:
        if len(shared) > 0:
            for n in shared:
                predecessors = list(G.predecessors(n))
                sum += 1 / math.log10(1 + len(predecessors))
            return sum
        else:
            return 0
    except:
        return 0


def resource_allocation(edge, G):
    source, sink = edge
    try:
        source_succesors = set(G.successors(source))
        sink_successors = set(G.successors(sink))
    except:
        return 0
    sum = 0.0
    shared = source_succesors.intersection(sink_successors)
    try:
        if len(shared) > 0:
            for n in shared:
                predecessors = list(G.predecessors(n))
                sum += 1 / len(predecessors)
            return sum
        else:
            return 0
    except:
        return 0


def preferential_attachment(edge, G):
    source, sink = edge
    try:
        source_succesors = set(G.successors(source))
        sink_successors = set(G.successors(sink))
    except:
        return 0

    return len(source_succesors) * len(sink_successors)


# Connectivity
def local_connectivity(edge, G, cutoff):
    source, sink = edge
    return local_node_connectivity(G, source, sink, cutoff=cutoff)


def shortest_path(edge, G):
    source, sink = edge
    dis = 100
    try:
        if G.has_edge(source, sink):
            # temporarily remove edge
            G.remove_edge(source, sink)
            # calculate shortest path length
            dis = nx.shortest_path_length(G, source=source, target=sink)
            # add back edge
            G.add_edge(source, sink)
        else:
            dis = nx.shortest_path_length(G, source=source, target=sink)
        return dis
    except:
        return dis


# Assortativity
def source_avg_nbr_degree(edge, G):
    source, sink = edge
    return average_neighbor_degree(G, nodes=[source])[source]


def sink_avg_nbr_degree(edge, G):
    source, sink = edge
    return average_neighbor_degree(G, nodes=[sink])[sink]


# Boundary size
def node_boundary_size(edge, G):
    return len(list(node_boundary(G, edge)))


# Centrality
def link_dispersion(edge, G):
    source, sink = edge
    return dispersion(G, source, sink)


# Clustering
def shared_neighbors(edge, G):
    source, sink = edge
    return len(list(nx.common_neighbors(G, source, sink)))


def source_triangles(edge, G):
    source, sink = edge
    return triangles(G, source)


def sink_triangles(edge, G):
    source, sink = edge
    return triangles(G, sink)


def source_clustering(edge, G):
    source, sink = edge
    return clustering(G, [source])[source]


def sink_clustering(edge, G):
    source, sink = edge
    return clustering(G, [sink])[sink]


def avg_clustering(edge, G):
    return average_clustering(G, edge)


# Efficiency
def link_efficiency(edge, G):
    source, sink = edge
    return efficiency(G, source, sink)


# Reciprocity
def is_followed_back(edge, G):
    source, sink = edge
    return int(G.has_edge(sink, source))


def source_reciprocity(edge, G):
    source, sink = edge
    return reciprocity(G, [source])[source]


def sink_reciprocity(edge, G):
    source, sink = edge
    r = reciprocity(G, [sink])[sink]
    if r == np.nan:
        return 0.0
    else:
        return r


# Extract features
def extract_features(
    edge_list: list, G: Graph, DiG: DiGraph, parameters: dict
) -> pd.DataFrame:
    """Extracts features for a list of edges on an undirected graph.

    Args:
        edges: a list of edges.

        G: a NetworkX undirected graph.

        DiG: a NetworkX directed graph.

    Returns:
        Pandas dataframe with edge features.
    """

    # Initialise logger and progress bar
    log = logging.getLogger(__name__)
    tqdm.pandas()

    # DEBUG ONLY: calculate features for subset
    subset = parameters["features"]["subset"]
    if subset:
        log.warning(red("Calculating features on first {} edges.".format(subset)))
        edges = edge_list[:subset]
    else:
        log.info(blue("Calculating features on {} edges.".format(len(edges))))
        edges = edge_list

    # Calculate edge features
    try:

        # Initialise feature matrix
        log.info(blue("Initialising feature matrix..."))
        df = pd.DataFrame(dict(edge=edges))

        # Degree features
        log.info(blue("Calculating degree features..."))
        df = add_degree_features(DiG, df)

        # Similarity indices
        log.info(blue("Calculating Jaccard coefficient..."))
        df["JC_followees"] = df.edge.progress_apply(jc_followees, G=DiG)
        df["JC_followers"] = df.edge.progress_apply(jc_followers, G=DiG)
        log.info(blue("Calculating Adamic-Adar index..."))
        df["AA_adj"] = df.edge.progress_apply(adjusted_adamic_adar_index, G=DiG)
        log.info(blue("Calculating resource allocation..."))
        df["RA"] = df.edge.progress_apply(resource_allocation, G=DiG)
        log.info(blue("Calculating preferential attachment..."))
        df["PA"] = df.edge.progress_apply(preferential_attachment, G=DiG)

        # Connectivity
        log.info(blue("Calculating connectivity..."))
        df["local_connectivity"] = df.edge.apply(
            local_connectivity,
            G=DiG,
            cutoff=parameters["features"]["connectivity"]["cutoff"],
        )
        # df["shortest_path"] = df.edge.progress_apply(shortest_path, G=DiG)

        # Assortativity
        log.info(blue("Calculating average neighbor degree..."))
        df["source_avg_nbr_degree"] = df.edge.progress_apply(
            source_avg_nbr_degree, G=DiG
        )
        df["sink_avg_nbr_degree"] = df.edge.progress_apply(sink_avg_nbr_degree, G=DiG)

        # # Boundary size
        # log.info(blue("Calculating boundary size..."))
        # df["node_boundary_size"] = df.edge.progress_apply(node_boundary_size, G=DiG)

        # # Centrality
        # log.info(blue("Calculating centrality..."))
        # centrality = degree_centrality(DiG)
        # df["source_centrality"] = df.edge.progress_apply(lambda e: centrality[e[0]])
        # df["sink_centrality"] = df.edge.progress_apply(lambda e: centrality[e[1]])
        # log.info(blue("Calculating in-degree centrality..."))
        # in_centrality = in_degree_centrality(DiG)
        # df["source_in_centrality"] = df.edge.progress_apply(
        #     lambda e: in_centrality[e[0]]
        # )
        # df["sink_in_centrality"] = df.edge.progress_apply(lambda e: in_centrality[e[1]])
        # log.info(blue("Calculating out-degree centrality..."))
        # out_centrality = out_degree_centrality(DiG)
        # df["source_out_centrality"] = df.edge.progress_apply(
        #     lambda e: out_centrality[e[0]]
        # )
        # df["sink_out_centrality"] = df.edge.progress_apply(
        #     lambda e: out_centrality[e[1]]
        # )
        log.info(blue("Calculating link dispersion..."))
        df["link_dispersion"] = df.edge.progress_apply(link_dispersion, G=G)

        # # Clustering
        # log.info(blue("Calculating shared neighbors..."))
        # df["shared_neighbors"] = df.edge.progress_apply(shared_neighbors, G=G)
        log.info(blue("Calculating clustering coefficients..."))
        df["source_clustering"] = df.edge.progress_apply(source_clustering, G=DiG)
        df["sink_clustering"] = df.edge.progress_apply(sink_clustering, G=DiG)

        # Efficiency
        log.info(blue("Calculating link efficiency..."))
        df["link_efficiency"] = df.edge.progress_apply(link_efficiency, G=G)

        # # Reciprocity
        log.info(blue("Calculating reciprocity metrics..."))
        df["follow_back"] = df.edge.progress_apply(is_followed_back, G=DiG)
        df["source_reciprocity"] = df.edge.progress_apply(source_reciprocity, G=DiG)
        df["sink_reciprocity"] = df.edge.progress_apply(sink_reciprocity, G=DiG)

        # Remove edge column
        df = df.drop("edge", axis=1)

    except:
        del df
        gc.collect()
        raise

    return df


def compile_features(
    G: Graph,
    DiG: DiGraph,
    edges_train: list,
    edges_valid: list,
    edges_test: list,
    parameters: dict,
) -> pd.DataFrame:
    """Compiles features for training, validation and test sets.

    Args:
        G: a NetworkX undirected graph.

        DiG: a NetworkX directed graph.

        edges_train: list of training edges.

        edges_valid: list of validation edges.

        edges_test: list of test edges.

    Returns:
        A list of Pandas dataframes with the features for each edge list.
    """
    return [
        extract_features(edges, G, DiG, parameters)
        for edges in [edges_train, edges_valid, edges_test]
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
    # Fill NA values with zeros
    non_stand_X_train = X_train.fillna(0.0)
    non_stand_X_valid = X_valid.fillna(0.0)
    non_stand_X_test = X_test.fillna(0.0)

    # Rescale features by training values
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(non_stand_X_train)
    X_valid = scaler.transform(non_stand_X_valid)
    X_test = scaler.transform(non_stand_X_test)

    return [X_train, X_valid, X_test]
