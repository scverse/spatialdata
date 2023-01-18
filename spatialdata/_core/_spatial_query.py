from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

import numpy as np
import pyarrow as pa
from multiscale_spatial_image.multiscale_spatial_image import MultiscaleSpatialImage
from spatial_image import SpatialImage

from spatialdata._core.coordinate_system import CoordinateSystem, _get_spatial_axes
from spatialdata._core.transformations import Sequence, Translation

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
    max_coordinate : np.ndarray
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


def _bounding_box_query_points_dict(
    points_dict: dict[str, pa.Table], request: BoundingBoxRequest
) -> dict[str, pa.Table]:
    requested_points = {}
    for points_name, points_data in points_dict.items():
        points = _bounding_box_query_points(points_data, request)
        if len(points) > 0:
            # do not include elements with no data
            requested_points[points_name] = points

    return requested_points


def _bounding_box_query_image(
    image: Union[MultiscaleSpatialImage, SpatialImage], request: BoundingBoxRequest
) -> Union[MultiscaleSpatialImage, SpatialImage]:
    """Perform a spatial bounding box query on an Image or Labels element.

    Parameters
    ----------
    image : Union[MultiscaleSpatialImage, SpatialImage]
        The image element to perform the query on.
    request : BoundingBoxRequest
        The request for the query.

    Returns
    -------
    query_result : Union[MultiscaleSpatialImage, SpatialImage]
        The image contained within the specified bounding box.
    """
    spatial_axes = _get_spatial_axes(request.coordinate_system)

    # build the request
    selection = {}
    for axis_index, axis_name in enumerate(spatial_axes):
        # get the min value along the axis
        min_value = request.min_coordinate[axis_index]

        # get max value, slices are open half interval
        max_value = request.max_coordinate[axis_index] + 1

        # add the
        selection[axis_name] = slice(min_value, max_value)

    query_result = image.sel(selection)

    # update the transform
    # currently, this assumes the existing transforms input coordinate system
    # is the intrinsic coordinate system
    # todo: this should be updated when we support multiple transforms
    initial_transform = query_result.transform
    n_axes_intrinsic = len(initial_transform.input_coordinate_system.axes_names)

    coordinate_system = initial_transform.input_coordinate_system
    spatial_indices = [i for i, axis in enumerate(coordinate_system._axes) if axis.type == "space"]

    translation_vector = np.zeros((n_axes_intrinsic,))
    for spatial_axis_index, coordinate_index in enumerate(spatial_indices):
        translation_vector[coordinate_index] = request.min_coordinate[spatial_axis_index]

    translation = Translation(
        translation=translation_vector,
        input_coordinate_system=coordinate_system,
        output_coordinate_system=coordinate_system,
    )

    new_transformation = Sequence(
        [translation, initial_transform],
        input_coordinate_system=coordinate_system,
        output_coordinate_system=initial_transform.output_coordinate_system,
    )

    query_result.attrs["transformation"] = new_transformation

    return query_result
