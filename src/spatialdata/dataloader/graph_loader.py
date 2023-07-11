from typing import Literal

import geopandas as gpd
import networkx as nx


def build_graph(
    gdf: gpd.GeoDataFrame,
    method: Literal["knn", "distance", "expansion"] = "knn",
    neighbors: int | None = None,  # type: ignore[syntax]
    threshold: float | None = None,  # type: ignore[syntax]
    distance: float | None = None,  # type: ignore[syntax]
    self_loops: bool = False,
    # **kwargs,
) -> nx.Graph:
    """
    Build a graph from a GeoDataFrame.

    Parameters
    ----------
    gdf
        The GeoDataFrame to build the graph from.
    method
        The method to use to build the graph. One of "knn", "distance", or "expansion".
    neighbors
        The number of neighbors to use for the knn method.
    threshold
        The threshold to use for the distance method.
    distance
        The distance to use for the expansion method.

    Returns
    -------
    The graph.
    """
    # check method is valid
    assert method in ["knn", "distance", "expansion"]
    # only one of neighbors, threshold, or distance can be specified, the one corresponding to the method selected
    if method == "knn":
        assert neighbors is not None
        assert threshold is None
        assert distance is None
    elif method == "distance":
        assert neighbors is None
        assert threshold is not None
        assert distance is None
    else:
        assert method == "expansion"
        assert neighbors is None
        assert threshold is None
        assert distance is not None
