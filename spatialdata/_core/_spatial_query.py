from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

import numpy as np
import pyarrow as pa
from geopandas import GeoDataFrame
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
    coordinate_system
        The coordinate system the coordinates are expressed in.
    min_coordinate
        The coordinate of the lower left hand corner (i.e., minimum values)
        of the bounding box.
    max_coordinate
        The coordinate of the upper right hand corner (i.e., maximum values)
        of the bounding box
    """

    min_coordinate: np.ndarray  # type: ignore[type-arg]
    max_coordinate: np.ndarray  # type: ignore[type-arg]


def _bounding_box_query_points(points: pa.Table, request: BoundingBoxRequest) -> pa.Table:
    """Perform a spatial bounding box query on a points element.

    Parameters
    ----------
    points
        The points element to perform the query on.
    request
        The request for the query.

    Returns
    -------
    The points contained within the specified bounding box.
    """
    spatial_axes = _get_spatial_axes(request.coordinate_system)

    for axis_index, axis_name in enumerate(spatial_axes):
        # filter by lower bound
        min_value = request.min_coordinate[axis_index]
        points = points[points[axis_name].gt(min_value)]

        # filter by upper bound
        max_value = request.max_coordinate[axis_index]
        points = points[points[axis_name].lt(max_value)]

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
    image
        The image element to perform the query on.
    request
        The request for the query.

    Returns
    -------
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

    query_result.attrs["transform"] = new_transformation

    return query_result


def _bounding_box_query_image_dict(
    image_dict: dict[str, Union[MultiscaleSpatialImage, SpatialImage]], request: BoundingBoxRequest
) -> dict[str, Union[MultiscaleSpatialImage, SpatialImage]]:
    requested_images = {}
    for image_name, image_data in image_dict.items():
        image = _bounding_box_query_image(image_data, request)
        if 0 not in image.shape:
            # do not include elements with no data
            requested_images[image_name] = image

    return requested_images


def _bounding_box_query_shapes(shapes_table: GeoDataFrame, request: BoundingBoxRequest) -> GeoDataFrame:
    """Perform a spatial bounding box query on a shapes element.

    Parameters
    ----------
    shapes_table
        The shapes element to perform the query on.
    request
        The request for the query.

    Returns
    -------
    The shapes contained within the specified bounding box.
    """
    spatial_axes = _get_spatial_axes(request.coordinate_system)

    # get the shapes bounding boxes
    shapes_min_column_keys = [f"min{axis}" for axis in spatial_axes]
    shapes_min_coordinates = shapes_table.bounds[shapes_min_column_keys].values

    shapes_max_column_keys = [f"max{axis}" for axis in spatial_axes]
    shapes_max_coordinates = shapes_table.bounds[shapes_max_column_keys].values

    # check that the min coordinates are inside the bounding box
    min_inside = np.all(request.min_coordinate < shapes_min_coordinates, axis=1)

    # check that the max coordinates are inside the bounding box
    max_inside = np.all(request.max_coordinate > shapes_max_coordinates, axis=1)

    # shapes inside the bounding box satisfy both
    shapes_inside = np.logical_and(min_inside, max_inside)

    return shapes_table.loc[shapes_inside]


def _bounding_box_query_shapes_dict(
    shapes_dict: dict[str, GeoDataFrame], request: BoundingBoxRequest
) -> dict[str, GeoDataFrame]:
    requested_shapes = {}
    for shapes_name, shapes_data in shapes_dict.items():
        shapes_table = _bounding_box_query_shapes(shapes_data, request)
        if len(shapes_table) > 0:
            # do not include elements with no data
            requested_shapes[shapes_name] = shapes_table

    return requested_shapes
