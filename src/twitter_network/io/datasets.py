from os.path import isfile
from typing import Any, Union, Dict
from collections import defaultdict
from tensorflow.keras.models import Sequential, load_model

import pandas as pd

from kedro.io import AbstractDataSet


class AdjListDataSet(AbstractDataSet):
    """Loads and saves data to a tab-delimited text file representing an adjacency list
    for a directed graph, where the first element in each row is a source node
    and the remaining elements are target nodes.

    Example:

    1   2
    2   3   4

    represents the network specified by the vertex set V = {1,2,3,4} and the directed
    edge set E = {(1,2), (2,3), (2,4)}.

    """

    def _describe(self) -> Dict[str, Any]:
        return dict(filepath=self._filepath, n_rows=self._n_rows)

    def __init__(self, filepath: str, n_rows: int) -> None:
        """Creates a new instance of ``AdjListDataSet`` pointing to a concrete
        filepath.

        Args:

            filepath: path to a edge list file.

            n_rows: number of lines to read from the edge list file.

        """
        self._filepath = filepath
        self._n_rows = n_rows

    def _load(self) -> dict:
        edge_list = defaultdict(int)
        with open(self._filepath) as f:
            for i in range(self._n_rows):
                if f:
                    line = next(f)
                    nodes = line.strip("\n").split("\t")
                    source = nodes[0]
                    targets = nodes[1:]
                    edge_list[source] = targets
            f.close()
        return dict(edge_list)

    def _save(self, data: dict) -> None:
        with open(self._filepath, "w") as f:
            lines = ["\t".join([str(node), data[node]]) for node in data.keys()]
            output = "\n".join(lines)
            f.write(output)
            f.close()

    def _exists(self) -> bool:
        return isfile(self._filepath)


class KerasDataSet(AbstractDataSet):
    """Loads and saves Keras neural network models to disk.
    """

    def _describe(self) -> Dict[str, Any]:
        return dict(filepath=self._filepath)

    def __init__(self, filepath: str) -> None:
        """Creates a new instance of ``KerasDataSet`` pointing to a concrete
        filepath.

        Args:

            filepath: path to a edge list file.

        """
        self._filepath = filepath

    def _load(self) -> Sequential:
        model = load_model(self._filepath)
        return model

    def _save(self, model: Sequential) -> None:
        model.save(self._filepath)
        del model

    def _exists(self) -> bool:
        return isfile(self._filepath)
