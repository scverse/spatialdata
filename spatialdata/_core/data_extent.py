"""This file contains functions to compute the bounding box describing the extent of a spatial element,
or of a specific region in the SpatialElement object."""
import numpy as np
from geopandas import GeoDataFrame

from spatialdata._core.core_utils import get_dims
from spatialdata._types import ArrayLike


def _get_bounding_box_of_circle_elements(self, shapes: GeoDataFrame) -> tuple[ArrayLike, ArrayLike, tuple[str, ...]]:
    """Get the coordinates for the corners of the bounding box of that encompasses a given spot.

    Returns
    -------
    min_coordinate
        The minimum coordinate of the bounding box.
    max_coordinate
        The maximum coordinate of the bounding box.
    """
    spots_element = self.sdata.shapes[self.spots_element_keys[0]]
    spots_dims = get_dims(spots_element)

    centroids = []
    for dim_name in spots_dims:
        centroids.append(getattr(spots_element["geometry"], dim_name).to_numpy())
    centroids_array = np.column_stack(centroids)
    radius = np.expand_dims(spots_element["radius"].to_numpy(), axis=1)

    min_coordinates = (centroids_array - radius).astype(int)
    max_coordinates = (centroids_array + radius).astype(int)

    return min_coordinates, max_coordinates, spots_dims
