from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa

from spatialdata._core.coordinate_system import CoordinateSystem, _get_spatial_axes

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class BaseSpatialRequest:
    """Base class for spatial queries."""

    coordinate_system: CoordinateSystem

    def __post_init__(self) -> None:
        # validate the coordinate system
        spatial_axes = _get_spatial_axes(self.coordinate_system)
        if len(spatial_axes) == 0:
            raise ValueError("No spatial axes in the requested coordinate system")


@dataclass(frozen=True)
class BoundingBoxRequest(BaseSpatialRequest):
    """Query with an axis-aligned bounding box.

    Attributes
    ----------
    coordinate_system : CoordinateSystem
        The coordinate system the coordinates are expressed in.
    min_coordinate : np.ndarray
        The coordinate of the lower left hand corner (i.e., minimum values)
        of the bounding box.
    max_coordiate : np.ndarray
        The coordinate of the upper right hand corner (i.e., maximum values)
        of the bounding box
    """

    min_coordinate: np.ndarray  # type: ignore[type-arg]
    max_coordinate: np.ndarray  # type: ignore[type-arg]


def _bounding_box_query_points(points: pa.Table, request: BoundingBoxRequest) -> pa.Table:
    """Perform a spatial bounding box query on a points element.

    Parameters
    ----------
    points : pa.Table
        The points element to perform the query on.
    request : BoundingBoxRequest
        The request for the query.

    Returns
    -------
    query_result : pa.Table
        The points contained within the specified bounding box.
    """
    spatial_axes = _get_spatial_axes(request.coordinate_system)

    for axis_index, axis_name in enumerate(spatial_axes):
        # filter by lower bound
        min_value = request.min_coordinate[axis_index]
        points = points.filter(pa.compute.greater(points[axis_name], min_value))

        # filter by upper bound
        max_value = request.max_coordinate[axis_index]
        points = points.filter(pa.compute.less(points[axis_name], max_value))

    return points
