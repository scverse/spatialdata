from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pyarrow as pa

from spatialdata._core.coordinate_system import CoordinateSystem

if TYPE_CHECKING:
    from spatialdata import SpatialData


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


class SpatialQueryManager:
    """Perform spatial queries on SpatialData objects"""

    def __init__(self, sdata: SpatialData):
        self._sdata = sdata

    def bounding_box(self, request: BoundingBoxRequest) -> SpatialData:
        """Perform a bounding box query on the SpatialData object.

        Parameters
        ----------
        request : BoundingBoxRequest
            The bounding box request.

        Returns
        -------
        requested_sdata : SpatialData
            The SpatialData object containing the requested data.
            Elements with no valid data are omitted.
        """
        requested_points = {}
        for points_name, points_data in self._sdata.points.items():
            points = _bounding_box_query_points(points_data, request)
            if len(points) > 0:
                # do not include elements with no data
                requested_points[points_name] = points

        return SpatialData(points=requested_points)

    def __call__(self, request: BaseSpatialRequest) -> SpatialData:
        if isinstance(request, BoundingBoxRequest):
            return self.bounding_box(request)
        else:
            raise TypeError("unknown request type")


def _get_spatial_axes(
    coordinate_system: CoordinateSystem,
) -> list[str]:
    """Get the names of the spatial axes in a coordinate system.

    Parameters
    ----------
    coordinate_system : CoordinateSystem
        The coordinate system to get the spatial axes from.

    Returns
    -------
    spatial_axis_names : List[str]
        The names of the spatial axes.
    """
    return [axis.name for axis in coordinate_system._axes if axis.type == "space"]


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
