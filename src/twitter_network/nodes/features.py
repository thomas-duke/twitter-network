"""Edge feature creation."""

import sys
import logging
import gc
import math
import pandas as pd
import numpy as np
import networkx as nx
from tqdm import tqdm
from collections import defaultdict
from itertools import compress

from networkx.classes.graph import Graph
from networkx.classes.digraph import DiGraph
from networkx.algorithms.approximation.connectivity import local_node_connectivity
from networkx.algorithms.assortativity import *
from networkx.algorithms.boundary import *
from networkx.algorithms.centrality import *
from networkx.algorithms.cluster import *
from networkx.algorithms.components import *
from networkx.algorithms.distance_measures import *
from networkx.algorithms.efficiency import *
from networkx.algorithms.reciprocity import *
from networkx.algorithms.shortest_paths import *
from networkx.algorithms.structuralholes import *

from sklearn.preprocessing import StandardScaler, normalize
import sklearn.feature_selection as fs
from sklearn.svm import LinearSVC
from twitter_network.nodes.setup import red, blue


def add_degree_features(G: DiGraph, df: pd.DataFrame) -> pd.DataFrame:
    source_in_degree = []
    source_out_degree = []
    source_bi_degree = []
    source_nbrs = []
    # source_nbh_links = []
    # # source_ego_nbh_links = []
    # source_nbh_scc = []
    # source_nbh_wcc = []
    # # source_ego_nbh_scc = []
    # # source_ego_nbh_wcc = []

    sink_in_degree = []
    sink_out_degree = []
    sink_bi_degree = []
    sink_nbrs = []
    # sink_nbh_links = []
    # # sink_ego_nbh_links = []
    # sink_nbh_scc = []
    # sink_nbh_wcc = []
    # # sink_ego_nbh_scc = []
    # # sink_ego_nbh_wcc = []

    shared_neighbors = []
    total_neighbors = []
    transitive_links = []

    # inner_nbh_links = []
    # inner_nbh_scc = []
    # inner_nbh_wcc = []

    jc_predecessors = []
    jc_successors = []
    jc_neighbors = []

    cos_predecessors = []
    cos_successors = []
    cos_neighbors = []

    pa_predecessors = []
    pa_successors = []
    pa_neighbors = []

    pref_attachment = []

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

        shared_neighbors.append(len(s_nbrs.intersection(d_nbrs)))
        total_neighbors.append(len(s_nbrs.union(d_nbrs)))
        transitive_links.append(len(s_out.intersection(d_in)))

        jc_predecessors.append(jaccard_coeff(s_in, d_in))
        jc_successors.append(jaccard_coeff(s_out, d_out))
        jc_neighbors.append(jaccard_coeff(s_nbrs, d_nbrs))

        cos_predecessors.append(cosine_distance(s_in, d_in))
        cos_successors.append(cosine_distance(s_out, d_out))
        cos_neighbors.append(cosine_distance(s_nbrs, d_nbrs))

        pa_predecessors.append(preferential_attachment(s_in, d_in))
        pa_successors.append(preferential_attachment(s_out, d_out))
        pa_neighbors.append(preferential_attachment(s_nbrs, d_nbrs))

        # source_nbh = G.subgraph(s_nbrs)
        # # source_ego_nbh = G.subgraph(s_nbrs.union(source))
        # source_nbh_links.append(source_nbh.number_of_edges())
        # # source_ego_nbh_links.append(source_ego_nbh.number_of_edges())
        # source_nbh_scc.append(number_strongly_connected_components(source_nbh))
        # source_nbh_wcc.append(number_weakly_connected_components(source_nbh))
        # # source_ego_nbh_scc.append(source_ego_nbh.number_strongly_connected_components())
        # # source_ego_nbh_wcc.append(source_ego_nbh.number_weakly_connected_components())

        # sink_nbh = G.subgraph(d_nbrs)
        # # sink_ego_nbh = G.subgraph(d_nbrs.union(sink))
        # sink_nbh_links.append(sink_nbh.number_of_edges())
        # # sink_ego_nbh_links.append(sink_ego_nbh.number_of_edges())
        # sink_nbh_scc.append(number_strongly_connected_components(sink_nbh))
        # sink_nbh_wcc.append(number_weakly_connected_components(source_nbh))
        # # sink_ego_nbh_scc.append(sink_ego_nbh.number_strongly_connected_components())
        # # sink_ego_nbh_wcc.append(sink_ego_nbh.number_weakly_connected_components())
        #
        # inner_nbh = G.subgraph(s_nbrs.union(d_nbrs))
        # inner_nbh_links.append(inner_nbh.number_of_edges())
        # inner_nbh_scc.append(number_strongly_connected_components(inner_nbh))
        # inner_nbh_wcc.append(number_weakly_connected_components(inner_nbh))

    df = pd.DataFrame(
        {
            "edge": df.edge,
            "source_in_degree": source_in_degree,
            "source_out_degree": source_out_degree,
            "source_bi_degree": source_bi_degree,
            "source_neighbors": source_nbrs,
            # "source_nbh_links": source_nbh_links,
            # # "source_ego_nbh_links": source_ego_nbh_links,
            # "source_nbh_scc": source_nbh_scc,
            # "source_nbh_wcc": source_nbh_wcc,
            # # "source_ego_nbh_scc": source_ego_nbh_scc,
            # # "source_ego_nbh_wcc": source_ego_nbh_wcc,
            "sink_in_degree": sink_in_degree,
            "sink_out_degree": sink_out_degree,
            "sink_bi_degree": sink_out_degree,
            "sink_neighbors": sink_nbrs,
            # "sink_nbh_links": sink_nbh_links,
            # # "sink_ego_nbh_links": sink_ego_nbh_links,
            # "sink_nbh_scc": sink_nbh_scc,
            # "sink_nbh_wcc": sink_nbh_wcc,
            # # "sink_ego_nbh_scc": sink_ego_nbh_scc,
            # # "sink_ego_nbh_wcc": sink_ego_nbh_wcc,
            "shared_neighbors": shared_neighbors,
            "total_neighbors": total_neighbors,
            "transitive_links": transitive_links,
            # "inner_nbh_links": inner_nbh_links,
            # "inner_nbh_scc": inner_nbh_scc,
            # "inner_nbh_wcc": inner_nbh_wcc,
            "jc_predecessors": jc_predecessors,
            "jc_successors": jc_successors,
            "jc_neighbors": jc_neighbors,
            "cos_predecessors": cos_predecessors,
            "cos_successors": cos_successors,
            "cos_neighbors": cos_neighbors,
            "pa_predecessors": pa_predecessors,
            "pa_successors": pa_successors,
            "pa_neighbors": pa_neighbors,
        }
    )

    # Calculate degree densities
    df["source_in_density"] = df["source_in_degree"] / df["source_neighbors"]
    df["source_out_density"] = df["source_out_degree"] / df["source_neighbors"]
    df["source_bi_density"] = df["source_bi_degree"] / df["source_neighbors"]
    # df["source_nbh_density"] = df["source_nbrs"] / df["source_nbh_links"]
    # # df["source_ego_nbh_density"] = df["source_nbrs"] / df["source_ego_nbh_links"]
    # df["source_avg_scc"] = df["source_nbrs"] / df["source_nbh_scc"]
    # # df["source_avg_ego_scc"] = df["source_nbrs"] / df["source_ego_nbh_scc"]
    # df["source_avg_wcc"] = df["source_nbrs"] / df["source_nbh_wcc"]
    # # df["source_avg_ego_wcc"] = df["source_nbrs"] / df["source_ego_nbh_wcc"]

    df["sink_in_density"] = df["sink_in_degree"] / df["sink_neighbors"]
    df["sink_out_density"] = df["sink_out_degree"] / df["sink_neighbors"]
    df["sink_bi_density"] = df["sink_bi_degree"] / df["sink_neighbors"]
    # df["sink_nbh_density"] = df["sink_nbrs"] / df["sink_nbh_links"]
    # # df["sink_ego_nbh_density"] = df["source_nbrs"] / df["sink_ego_nbh_links"]
    # df["sink_avg_scc"] = df["sink_nbrs"] / df["sink_nbh_scc"]
    # # df["sink_avg_ego_scc"] = df["sink_nbrs"] / df["sink_ego_nbh_scc"]
    # df["sink_avg_wcc"] = df["sink_nbrs"] / df["sink_nbh_wcc"]
    # # df["sink_avg_ego_wcc"] = df["sink_nbrs"] / df["sink_ego_nbh_wcc"]

    return df


# Similarity indices
def cosine_distance(x1, x2):
    try:
        if len(x1) == 0 | len(x2) == 0:
            return 0
        intersect = len(set(x1.intersection(x2)))
        denom = math.sqrt(len(x1) * len(x2))
        sim = intersect / denom
        return sim
    except:
        return 0


def jaccard_coeff(x1, x2):
    try:
        if len(x1) == 0 or len(x2) == 0:
            return 0

        shared = x1.intersection(x2)
        total = x1.union(x2)

        jc = len(shared) / len(total)
        return jc
    except:
        return 0


def preferential_attachment(x1, x2):
    return len(x1) * len(x2)


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
                sum += 1 / math.log(1 + len(predecessors))
            return sum
        else:
            return 0
    except:
        return 0


def resource_allocation(edge, G):
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
                sum += 1 / len(predecessors)
            return sum
        else:
            return 0
    except:
        return 0


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


def source_page_rank(edge, page_rank):
    source, sink = edge
    try:
        pr = page_rank[int(source)]
    except:
        pr = 0
    return pr


def sink_page_rank(edge, page_rank):
    source, sink = edge
    try:
        pr = page_rank[int(sink)]
    except:
        pr = 0
    return pr


def source_katz(edge, katz):
    source, sink = edge
    try:
        k = katz[int(source)]
    except:
        k = 0
    return k


def sink_katz(edge, katz):
    source, sink = edge
    try:
        k = katz[int(sink)]
    except:
        k = 0
    return k


# Clustering
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
    edge_list: list,
    G: Graph,
    DiG: DiGraph,
    page_rank: dict,
    katz: dict,
    parameters: dict,
) -> pd.DataFrame:
    """Extracts features for a list of edges on an undirected graph.

    Args:
        edges: a list of edges.

        G: a NetworkX undirected graph.

        DiG: a NetworkX directed graph.

        page_rank: dictionary containing page_rank measures.

        katz: dictionary containing katz centrality measures.

        parameters: parameters defined in parameters.yml.

    Returns:
        Pandas dataframe with edge features.
    """

    # Initialise logger and progress bar
    log = logging.getLogger(__name__)
    tqdm.pandas()

    # DEBUG ONLY: calculate features for subset
    subset = parameters["features"]["subset"]
    if subset:
        edges = edge_list[:subset]
        log.warning(red("Calculating features on first {} edges.".format(subset)))
    else:
        edges = edge_list
        log.info(blue("Calculating features on {} edges.".format(len(edges))))

    # Calculate edge features
    try:

        # Initialise feature matrix
        log.info(blue("Initialising feature matrix..."))
        df = pd.DataFrame(dict(edge=edges))

        # Degree features
        log.info(blue("Calculating degree features..."))
        df = add_degree_features(DiG, df)

        # Similarity indices
        log.info(blue("Calculating Adamic-Adar index..."))
        df["adjusted_adamic_adar"] = df.edge.progress_apply(
            adjusted_adamic_adar_index, G=DiG
        )
        log.info(blue("Calculating resource allocation..."))
        df["resource_allocation"] = df.edge.progress_apply(resource_allocation, G=DiG)
        # TOO SLOW
        # Connectivity
        # log.info(blue("Calculating connectivity..."))
        # df["local_connectivity"] = df.edge.apply(
        #     local_connectivity,
        #     G=DiG,
        #     cutoff=parameters["features"]["connectivity"]["cutoff"],
        # )
        log.info(blue("Calculating shortest path..."))
        df["shortest_path"] = df.edge.progress_apply(shortest_path, G=DiG)

        # Assortativity
        log.info(blue("Calculating average neighbor degree..."))
        df["source_avg_nbr_degree"] = df.edge.progress_apply(
            source_avg_nbr_degree, G=DiG
        )
        df["sink_avg_nbr_degree"] = df.edge.progress_apply(sink_avg_nbr_degree, G=DiG)

        # Boundary size
        log.info(blue("Calculating boundary size..."))
        df["node_boundary_size"] = df.edge.progress_apply(node_boundary_size, G=DiG)

        # Centrality
        log.info(blue("Calculating centrality..."))
        centrality = degree_centrality(DiG)
        df["source_centrality"] = df.edge.progress_apply(lambda e: centrality[e[0]])
        df["sink_centrality"] = df.edge.progress_apply(lambda e: centrality[e[1]])
        log.info(blue("Calculating in-degree centrality..."))
        in_centrality = in_degree_centrality(DiG)
        df["source_in_centrality"] = df.edge.progress_apply(
            lambda e: in_centrality[e[0]]
        )
        df["sink_in_centrality"] = df.edge.progress_apply(lambda e: in_centrality[e[1]])
        log.info(blue("Calculating out-degree centrality..."))
        out_centrality = out_degree_centrality(DiG)
        df["source_out_centrality"] = df.edge.progress_apply(
            lambda e: out_centrality[e[0]]
        )
        df["sink_out_centrality"] = df.edge.progress_apply(
            lambda e: out_centrality[e[1]]
        )

        # High order centrality measures
        log.info(blue("Calculating PageRank..."))
        df["source_page_rank"] = df.edge.progress_apply(
            source_page_rank, page_rank=page_rank
        )
        df["sink_page_rank"] = df.edge.progress_apply(
            sink_page_rank, page_rank=page_rank
        )

        log.info(blue("Calculating Katz centrality..."))
        df["source_katz"] = df.edge.progress_apply(source_katz, katz=katz)
        df["sink_katz"] = df.edge.progress_apply(sink_katz, katz=katz)

        # TOO SLOW
        # Dispersion
        # log.info(blue("Calculating link dispersion..."))
        # df["link_dispersion"] = df.edge.progress_apply(link_dispersion, G=G)

        # TOO SLOW
        # Clustering
        # log.info(blue("Calculating clustering coefficients..."))
        # df["source_clustering"] = df.edge.progress_apply(source_clustering, G=DiG)
        # df["sink_clustering"] = df.edge.progress_apply(sink_clustering, G=DiG)

        # Efficiency
        log.info(blue("Calculating link efficiency..."))
        df["link_efficiency"] = df.edge.progress_apply(link_efficiency, G=G)

        # Reciprocity
        log.info(blue("Calculating reciprocity metrics..."))
        df["is_followed_back"] = df.edge.progress_apply(is_followed_back, G=DiG)
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
    page_rank: dict,
    katz: dict,
    parameters: dict,
) -> pd.DataFrame:
    """Compiles features for training, validation and test sets.

    Args:
        G: a NetworkX undirected graph.

        DiG: a NetworkX directed graph.

        edges_train: list of training edges.

        edges_valid: list of validation edges.

        edges_test: list of test edges.

        page_rank: dictionary containing page_rank measures.

        katz: dictionary containing katz centrality measures.

        parameters: parameters defined in parameters.yml.

    Returns:
        A list of Pandas dataframes with the features for each edge list.
    """
    return [
        extract_features(
            edge_list=edges,
            G=G,
            DiG=DiG,
            page_rank=page_rank,
            katz=katz,
            parameters=parameters,
        )
        for edges in [edges_train, edges_valid, edges_test]
    ]


def log_offset(X):
    return np.log(1 + X)


def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == "":
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' " "(or 'y' or 'n').\n")


def pause():
    if not query_yes_no("Proceed?"):
        exit()


def transform_features(
    X_train: pd.DataFrame, X_valid: pd.DataFrame, X_test: pd.DataFrame, parameters: dict
) -> list:
    """Apply log transformations to skewed features

    Args:
        X_train: training data.

        X_valid: validation data.

        X_test: test data.

    Returns:
        A list containing the transformed training, validation and test data.
    """

    log = logging.getLogger(__name__)
    paras = parameters["features"]["transformation"]
    threshold = paras["skew_threshold"]  # threshold for applying log transformation
    inclusions = paras["inclusions"]  # strings to include in the transformation
    exclusions = paras["exclusions"]  # strings to exclude from the transformation

    # Select highly skewed variables
    feature_skew = X_train.skew().sort_values(ascending=False)
    skewed = list(feature_skew[feature_skew > threshold].index)
    log.info(
        blue(
            "{} feature(s) exceed skew threshold of {}.".format(len(skewed), threshold)
        )
    )
    if inclusions:
        n_skewed = len(skewed)
        skewed = [var for var in skewed if any(str in var for str in inclusions)]
        n_excluded = n_skewed - len(skewed)
        log.warning(
            red(
                "Including {} variable(s) containing:\n{}.".format(
                    n_excluded, inclusions
                )
            )
        )
        pause()
    if exclusions:
        n_skewed = len(skewed)
        skewed = [var for var in skewed if all(str not in var for str in exclusions)]
        n_excluded = n_skewed - len(skewed)
        log.warning(
            red(
                "Excluding {} variable(s) containing:\n{}.".format(
                    n_excluded, exclusions
                )
            )
        )
        pause()

    # Check variables are correct
    log.warning(red("Transforming {} variable(s): {}.".format(len(skewed), skewed)))
    pause()

    # Apply log transformation to skewed variables
    log.info(blue("Applying log transformation to {} feature(s).".format(len(skewed))))
    log_X_train = X_train
    log_X_valid = X_valid
    log_X_test = X_test
    for var in skewed:
        var_name = "log_" + var
        log_X_train[var_name] = log_offset(log_X_train[var])
        log_X_valid[var_name] = log_offset(log_X_valid[var])
        log_X_test[var_name] = log_offset(log_X_test[var])
        if paras["drop_vars"]:
            log_X_train = log_X_train.drop(var, axis=1)
            log_X_valid = log_X_valid.drop(var, axis=1)
            log_X_test = log_X_test.drop(var, axis=1)

    return [log_X_train, log_X_valid, log_X_test]


def rescale_features(
    X_train: pd.DataFrame, X_valid: pd.DataFrame, X_test: pd.DataFrame
) -> list:
    """Standardises features using the mean and standard deviation of the training data.

    Args:
        X_train: training data.

        X_valid: validation data.

        X_test: test data.

    Returns:
        A list containing the standardised training, validation and test data.
    """
    # Fill NA values with zeros
    non_stand_X_train = X_train.fillna(0.0)
    non_stand_X_valid = X_valid.fillna(0.0)
    non_stand_X_test = X_test.fillna(0.0)

    # Get feature names
    features = list(X_train.columns)

    # Rescale features by training values
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(non_stand_X_train)
    X_valid = scaler.transform(non_stand_X_valid)
    X_test = scaler.transform(non_stand_X_test)

    # Recreate dataframes
    X_train = pd.DataFrame(X_train, columns=features)
    X_valid = pd.DataFrame(X_valid, columns=features)
    X_test = pd.DataFrame(X_test, columns=features)

    return [X_train, X_valid, X_test]


def select_features(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    parameters: dict,
) -> list:
    """Extracts the most relevent features from the sample by fitting a linear SVM on
    the training data.

    Args:
        X_train: training data.

        X_valid: validation data.

        X_test: test data.

        parameters: parameters defined in parameters.yml.

    Returns:
        A list containing the trimmed training, validation and test data.
    """

    log = logging.getLogger(__name__)
    paras = parameters["features"]["selection"]

    if paras["skip"]:
        log.warning(red("Skipping feature selection."))
        pause()
        return dict(
            X_train_reduced=X_train,
            X_valid_reduced=X_valid,
            X_test_reduced=X_test,
            selected=X_train.columns,
            excluded=[],
        )

    # Get feature names
    features = list(X_train.columns)

    # Score functions
    funcs = {"chi2": fs.chi2, "f": fs.f_classif, "mi": fs.mutual_info_classif}

    # Instantiate the selector and fit it to the training data
    selector = fs.GenericUnivariateSelect(
        score_func=funcs[paras["score_func"]], mode=paras["mode"], param=paras["param"]
    )
    if paras["subset"]:
        subset = paras["subset"]
    else:
        subset = len(X_train)
    selector.fit(X_train[:subset], y_train[:subset])

    # Get feature mask
    mask = selector.get_support()
    included = list(compress(features, mask))
    excluded = list(set(features) - set(included))
    log.info(blue("Included {} variables: {}".format(len(included), included)))
    log.info(blue("Excluded {} variables: {}".format(len(excluded), excluded)))

    # Transform the datasets using the fitted selector
    X_train_reduced = selector.transform(X_train)
    X_valid_reduced = selector.transform(X_valid)
    X_test_reduced = selector.transform(X_test)

    return dict(
        X_train_reduced=X_train_reduced,
        X_valid_reduced=X_valid_reduced,
        X_test_reduced=X_test_reduced,
        included=included,
        excluded=excluded,
    )
