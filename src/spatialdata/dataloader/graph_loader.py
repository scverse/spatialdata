from __future__ import annotations

from typing import Any, Literal

import geopandas as gpd
import numpy as np
from scipy.spatial import KDTree

from spatialdata._logging import logger
from spatialdata._types import ArrayLike


def _compute_percentile(tree: KDTree, percentile: int, centroids: ArrayLike) -> float:
    logger.warning("computing percentile could take long for large datasets.")
    dd = tree.sparse_distance_matrix(
        tree,
        max_distance=np.sqrt(
            (np.max(centroids[:, 0]) - np.min(centroids[:, 0])) ** 2
            + (np.max(centroids[:, 1]) - np.min(centroids[:, 1])) ** 2
        ),
    ).toarray()
    np.fill_diagonal(dd, np.nan)
    dd = dd[~np.isnan(dd)].reshape(dd.shape[0], dd.shape[1] - 1)
    f = np.percentile(dd, percentile)
    assert isinstance(f, float)
    return f


def build_graph(  # TODO: add rings for grids
    gdf: gpd.GeoDataFrame,
    method: Literal["knn", "radius", "expansion"] = "knn",
    use_centroids: bool = True,
    k: int | None = None,  # type: ignore[syntax]
    max_distance: float | None = None,  # type: ignore[syntax]
    percentile: int | None = None,
    self_loops: bool = False,
    **kwargs: Any,
) -> ArrayLike:
    """
    Build a graph connecting the geometries of a GeoDataFrame.

    Parameters
    ----------
    gdf
        The GeoDataFrame to build the graph from.
    method
        The method to use to build the graph. One of "knn", "radius", or "expansion".
    use_centroids
        Weather to use centroids. Assumed use_centroids=True for method='knn' and 'radius'. Only used when
        method='expansion'.
    k
        The number of k to use for the method='knn', needs to be None for method='radius' or method='expansion'.
    max_distance
        The maximum distance between centroids for method='knn' or method='radius', and the buffer for expansion when
        method='expansion'.
    percentile
        Integer between 0 and 100 (including the endpoints). If max_distance is None, percentile can be used to set
        max_distance, based on the percentile of pair-wise distances between the centroids. Computing the percentile
        could take long for large datasets.
    self_loops
        Whether to keep self loops.

    Returns
    -------
    The graph in the form of edge indices (compatible with PyTorch Geometric).
    """
    # validate the parameters passed on the method
    assert method in ["knn", "radius", "expansion"]

    if max_distance is not None:
        assert max_distance > 0
    if percentile is not None:
        assert percentile >= 0
        assert percentile <= 100

    if method == "knn":
        assert k is not None
        assert k > 0
        assert not (percentile is not None and max_distance is not None)
    if method == "radius":
        assert k is None
        assert (percentile is None) ^ (max_distance is None)
    if method == "expansion":
        assert k is None
        assert percentile is None

    if method in ["knn", "radius"] and not use_centroids:
        logger.warning(f"method {method} only available for centroids, automatically set use_centroids=True.")
        use_centroids = True

    if use_centroids:
        gdf_c = gdf.centroid

    if method == "knn":
        centroids = np.array(gdf_c.apply(lambda g: [g.x, g.y]).tolist())
        tree = KDTree(centroids)

        if max_distance is not None:
            d_tree, idx = tree.query(centroids, k=k, distance_upper_bound=max_distance)
        elif percentile is not None:
            assert percentile is not None
            max_distance = _compute_percentile(tree, percentile, centroids)
            d_tree, idx = tree.query(centroids, k=k, distance_upper_bound=max_distance)
        else:
            d_tree, idx = tree.query(centroids, k=k)

        idx = idx[:, 1:]
        if idx.size == 0:
            return np.array([[]])
        point_numbers = np.arange(len(centroids))
        assert isinstance(k, int)
        point_numbers = np.repeat(point_numbers, k - 1)
        idx_flatten = idx.flatten()
        edge_index = np.vstack((point_numbers, idx_flatten)).T
        # to delete the edges longer than max_distance
        # check for self edges
    elif method == "radius":
        gdf_c = gdf.centroid
        centroids = gdf_c.apply(lambda g: [g.x, g.y]).tolist()
        tree = KDTree(centroids)
        if max_distance is not None:
            neigh_list = tree.query_ball_tree(tree, r=max_distance)
        else:
            assert isinstance(percentile, int)
            max_distance = _compute_percentile(tree, percentile, centroids)
            neigh_list = tree.query_ball_tree(tree, r=max_distance)
        edge_index_list: list[list[int]] = []
        for sublist_index, sublist in enumerate(neigh_list):
            for other_index in sublist:
                edge_index_list.append([sublist_index, other_index])
        edge_index = np.array(edge_index_list)
    else:
        if use_centroids:
            gdf_cent = gdf.centroid
            gdf = gpd.GeoDataFrame({"geometry": gdf_cent, "id": gdf.index})  # check if index is the best way
        gdf = gpd.GeoDataFrame(
            {"geometry": gdf.buffer(max_distance, **kwargs), "id": gdf.index}
        )  # check if index is the best way
        gdf_ret = gdf.overlay(gdf, how="intersection")
        edge_index = gdf_ret[["id_1", "id_2"]].values

    edge_index = np.unique(edge_index, axis=0)
    edge_index = np.delete(edge_index, np.where(edge_index == gdf.shape[0])[0], 0)
    if not self_loops:
        mask = edge_index[:, 0] != edge_index[:, 1]
        edge_index = edge_index[mask]

    return edge_index.T  # transposed to be compatible with pyg
