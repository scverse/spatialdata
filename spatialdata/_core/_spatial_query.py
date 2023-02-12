from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

import numpy as np
import pyarrow as pa
from geopandas import GeoDataFrame
from multiscale_spatial_image.multiscale_spatial_image import MultiscaleSpatialImage
from spatial_image import SpatialImage
from dask.dataframe.core import DataFrame as DaskDataFrame

from spatialdata._core.transformations import Sequence, Translation
from spatialdata._core.core_utils import ValidAxis_t, validate_axes, get_spatial_axes


@dataclass(frozen=True)
class BaseSpatialRequest:
    """Base class for spatial queries."""

    axes: tuple[ValidAxis_t, ...]

    def __post_init__(self) -> None:
        # validate the axes
        spatial_axes = get_spatial_axes(self.axes)
        non_spatial_axes = set(self.axes) - set(spatial_axes)
        if len(non_spatial_axes) > 0:
            raise ValueError(f"Non-spatial axes specified: {non_spatial_axes}")


@dataclass(frozen=True)
class BoundingBoxRequest(BaseSpatialRequest):
    """Query with an axis-aligned bounding box.

    Attributes
    ----------
    axes
        The axes the coordinates are expressed in.
    min_coordinate
        The coordinate of the lower left hand corner (i.e., minimum values)
        of the bounding box.
    max_coordinate
        The coordinate of the upper right hand corner (i.e., maximum values)
        of the bounding box
    """

    min_coordinate: np.ndarray  # type: ignore[type-arg]
    max_coordinate: np.ndarray  # type: ignore[type-arg]

    def __post_init__(self) -> None:
        super().__post_init__()

        # validate the axes
        if len(self.axes) != len(self.min_coordinate) or len(self.axes) != len(self.max_coordinate):
            raise ValueError("The number of axes must match the number of coordinates.")

        # validate the coordinates
        if np.any(self.min_coordinate > self.max_coordinate):
            raise ValueError("The minimum coordinate must be less than the maximum coordinate.")


def _bounding_box_query_points(points: DaskDataFrame, request: BoundingBoxRequest) -> DaskDataFrame:
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

    for axis_index, axis_name in enumerate(request.axes):
        # filter by lower bound
        min_value = request.min_coordinate[axis_index]
        points = points[points[axis_name].gt(min_value)]

        # filter by upper bound
        max_value = request.max_coordinate[axis_index]
        points = points[points[axis_name].lt(max_value)]

    return points


def _bounding_box_query_points_dict(
    points_dict: dict[str, DaskDataFrame], request: BoundingBoxRequest
) -> dict[str, DaskDataFrame]:
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
    from spatialdata._core._spatialdata_ops import get_transformation, set_transformation

    # build the request
    selection = {}
    for axis_index, axis_name in enumerate(request.axes):
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
    initial_transform = get_transformation(query_result)

    translation = Translation(
        translation=request.min_coordinate,
        axes=request.axes
    )

    new_transformation = Sequence(
        [translation, initial_transform],
    )

    set_transformation(query_result, new_transformation)

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


def _bounding_box_query_polygons(polygons_table: GeoDataFrame, request: BoundingBoxRequest) -> GeoDataFrame:
    """Perform a spatial bounding box query on a polygons element.

    Parameters
    ----------
    polygons_table
        The polygons element to perform the query on.
    request
        The request for the query.

    Returns
    -------
    The polygons contained within the specified bounding box.
    """
    # get the polygon bounding boxes
    polygons_min_column_keys = [f"min{axis}" for axis in request.axes]
    polygons_min_coordinates = polygons_table.bounds[polygons_min_column_keys].values

    polygons_max_column_keys = [f"max{axis}" for axis in request.axes]
    polygons_max_coordinates = polygons_table.bounds[polygons_max_column_keys].values

    # check that the min coordinates are inside the bounding box
    min_inside = np.all(request.min_coordinate < polygons_min_coordinates, axis=1)

    # check that the max coordinates are inside the bounding box
    max_inside = np.all(request.max_coordinate > polygons_max_coordinates, axis=1)

    # polygons inside the bounding box satisfy both
    polygon_inside = np.logical_and(min_inside, max_inside)

    return polygons_table.loc[polygon_inside]


def _bounding_box_query_polygons_dict(
    polygons_dict: dict[str, GeoDataFrame], request: BoundingBoxRequest
) -> dict[str, GeoDataFrame]:
    requested_polygons = {}
    for polygons_name, polygons_data in polygons_dict.items():
        polygons_table = _bounding_box_query_polygons(polygons_data, request)
        if len(polygons_table) > 0:
            # do not include elements with no data
            requested_polygons[polygons_name] = polygons_table

    return requested_polygons
