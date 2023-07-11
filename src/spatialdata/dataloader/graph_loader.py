from typing import Literal
import geopandas as gpd
from shapely import wkt
import networkx as nx
import numpy as np
from scipy.spatial import KDTree

def build_graph(
    gdf: gpd.GeoDataFrame,
    method: Literal["knn", "radius", "expansion"] = "knn",
    use_centroids: bool = False,
    k: int | None = None,  # type: ignore[syntax]
    max_distance: float | None = None,  # type: ignore[syntax]
    percentile: int | None = None,
    self_loops: bool = False,
    **kwargs
) -> np.ndarray:
    """
    Build a graph from a GeoDataceFrame.

    Parameters
    ----------
    gdf
        The GeoDataFrame to build the graph from.
    method
        The method to use to build the graph. One of "knn", "radius", or "expansion".
    use_centroids
        Weather to use centroids. Only used when method='expansion'.
    k
        The number of k to use for the knn method.
    max_distance
        The maximum distance between centroids for method='knn' or method='radius', and the buffer for expantion when method='expansion'.
    percentile
        If max_distance is None, percentile can be used to set max_distance, based on the percentile of pair-wise distances between the centroids.
    self_loops
        Weather to keep self loops.

    Returns
    -------
    The graph in the form of edge indices.

    """
    # check method is valid
    assert method in ["knn", "radius", "expansion"]
    if use_centroids:
        gdf = gdf.centroid
    # only one of k, threshold, or max_distance can be specified, the one corresponding to the method selected
    if method == "knn":
        assert k is not None
        assert k > 0
        assert max_distance is None or max_distance > 0 
        if not use_centroids:
            raise Warning("method knn is only available for centroids, automatically set use_centroids=True.")
        gdf_t     = gdf.centroid
        centroids = np.array(gdf_t.apply(lambda g:[g.x,g.y]).tolist())
        tree      = KDTree(centroids)
        
        if max_distance is not None:
            if percentile is not None:
                raise Warning("both percentile and max_distance are set, will only use max_distance.")
            d_tree, idx   = tree.query(centroids, k=k, distance_upper_bound=max_distance)
        else:
            if percentile is not None:
                raise Warning("computing percentile could take long for large datasets.")
                dd           = tree.sparse_distance_matrix(tree, max_distance=np.sqrt((np.max(centroids[:,0]) - np.min(centroids[:,0]))**2 + (np.max(centroids[:,1]) - np.min(centroids[:,1]))**2)).toarray()
                np.fill_diagonal(dd, np.nan)
                dd           = dd[~np.isnan(dd)].reshape(dd.shape[0], dd.shape[1] - 1)
                max_distance = np.percentile(dd, percentile)
                d_tree, idx  = tree.query(centroids, k=k, distance_upper_bound=max_distance)
            else:
                d_tree, idx   = tree.query(centroids, k=k)
        idx           = idx[:,1:]
        point_numbers = np.arange(len(centroids))
        point_numbers = np.repeat(point_numbers, k-1)
        idx_flatten   = idx.flatten()
        edge_index    = np.vstack((point_numbers,idx_flatten))
        # check for self edges
    elif method == "radius":
        assert k is None
        assert max_distance > 0 or percentile > 0
        if not use_centroids:
            raise Warning("method radius is only available for centroids, automatically set use_centroids=True.")
        gdf_t     = gdf.centroid
        centroids = gdf_t.apply(lambda g:[g.x,g.y]).tolist()
        tree      = KDTree(centroids)
        if max_distance is not None:
            if percentile is not None:
                raise Warning("both percentile and max_distance are set, will only use max_distance.")
            neigh_list    = tree.query_ball_tree(tree, r=max_distance) 
        else:
            raise Warning("computing percentile could take long for large datasets.")
            dd           = tree.sparse_distance_matrix(tree, max_distance=np.sqrt((np.max(centroids[:,0]) - np.min(centroids[:,0]))**2 + (np.max(centroids[:,1]) - np.min(centroids[:,1]))**2)).toarray()
            np.fill_diagonal(dd, np.nan)
            dd           = dd[~np.isnan(dd)].reshape(dd.shape[0], dd.shape[1] - 1)
            max_distance = np.percentile(dd, percentile)
            neigh_list   = tree.query_ball_tree(tree, r=max_distance) 
        edge_index = []
        for sublist_index, sublist in enumerate(neigh_list):
            for index in sublist:
                for other_index in sublist:
                    edge_index.append([sublist_index, other_index])
        edge_index = np.array(edge_index)
    else:
        assert method == "expansion"
        assert k is None
        assert max_distance > 0 or percentile > 0
        gdf        = gpd.GeoDataFrame({'geometry': gdf.buffer(max_distance, **kwargs), 'id': gdf.index}) #check if index is the best way
        gdf_ret    = gdf.overlay(gdf, how='intersection')
        edge_index = gdf_ret[['id_1', 'id_2']].values
    if not self_loops:
        mask = edge_index[:,0] != edge_index[:,1]
        edge_index = edge_index[mask]
    return edge_index.T # transposed to be compatible with pyg 



