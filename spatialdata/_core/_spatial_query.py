from __future__ import annotations

from dataclasses import dataclass
from functools import singledispatch
from typing import Union

import dask.array as da
import numpy as np
from dask.dataframe.core import DataFrame as DaskDataFrame
from geopandas import GeoDataFrame
from multiscale_spatial_image.multiscale_spatial_image import MultiscaleSpatialImage
from spatial_image import SpatialImage

from spatialdata import SpatialData, SpatialElement
from spatialdata._core._query_requests import _dict_query_dispatcher
from spatialdata._core.core_utils import ValidAxis_t, get_spatial_axes
from spatialdata._core.transformations import BaseTransformation, Sequence, Translation
from spatialdata._types import ArrayLike


def get_bounding_box_corners(min_coordinate: ArrayLike, max_coordinate: ArrayLike) -> ArrayLike:
    """From the min and max coordinates of a bounding box, get the coordinates
    of all corners.

    Parameters
    ----------
    min_coordinate : np.ndarray
        The upper left hand corner of the bounding box (i.e., minimum coordinates
        along all dimensions).
    max_coordinate : np.ndarray
        The lower right hand corner of the bounding box (i.e., the maximum coordinates
        along all dimensions

    Returns
    -------
    (N, D) array of coordinates of the corners. N = 4 for 2D and 8 for 3D.
    """
    if len(min_coordinate) == 2:
        # 2D bounding box
        return np.array(
            [
                [min_coordinate[0], min_coordinate[1]],
                [min_coordinate[0], max_coordinate[1]],
                [max_coordinate[0], max_coordinate[1]],
                [max_coordinate[0], min_coordinate[1]],
            ]
        )

    elif len(min_coordinate) == 3:
        # 3D bounding cube
        return np.array(
            [
                [min_coordinate[0], min_coordinate[1], min_coordinate[2]],
                [min_coordinate[0], min_coordinate[1], max_coordinate[2]],
                [min_coordinate[0], max_coordinate[1], max_coordinate[2]],
                [min_coordinate[0], max_coordinate[1], min_coordinate[2]],
                [max_coordinate[0], min_coordinate[1], min_coordinate[2]],
                [max_coordinate[0], min_coordinate[1], max_coordinate[2]],
                [max_coordinate[0], max_coordinate[1], max_coordinate[2]],
                [max_coordinate[0], max_coordinate[1], min_coordinate[2]],
            ]
        )
    else:
        raise ValueError("bounding box must be 2D or 3D")


def _get_bounding_box_corners_in_intrinsic_coordinates(
    element: SpatialElement,
    min_coordinate: ArrayLike,
    max_coordinate: ArrayLike,
    target_coordinate_system: str,
    axes: tuple[str, ...],
) -> tuple[ArrayLike, ArrayLike, tuple[str, ...]]:
    from spatialdata._core._spatialdata_ops import get_transformation
    from spatialdata._core.core_utils import get_dims

    # get the transformation from the element's intrinsic coordinate system
    # to the query coordinate space
    transform_to_query_space = get_transformation(element, to_coordinate_system=target_coordinate_system)

    # get the coordinates of the bounding box corners
    bounding_box_corners = get_bounding_box_corners(
        min_coordinate=min_coordinate,
        max_coordinate=max_coordinate,
    )

    # transform the coordinates to the intrinsic coordinate system
    intrinsic_axes = get_dims(element)
    transform_to_intrinsic = transform_to_query_space.inverse().to_affine_matrix(  # type: ignore[union-attr]
        input_axes=axes, output_axes=intrinsic_axes
    )
    ndim = len(min_coordinate)
    rotation_matrix = transform_to_intrinsic[0:ndim, 0:ndim]
    translation = transform_to_intrinsic[0:ndim, ndim]

    intrinsic_bounding_box_corners = bounding_box_corners @ rotation_matrix.T + translation

    min_coordinate_intrinsic = intrinsic_bounding_box_corners.min(axis=0)
    max_coordinate_intrinsic = intrinsic_bounding_box_corners.max(axis=0)

    return min_coordinate_intrinsic, max_coordinate_intrinsic, intrinsic_axes


@dataclass(frozen=True)
class BaseSpatialRequest:
    """Base class for spatial queries."""

    target_coordinate_system: str
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


def _bounding_box_mask_points(
    points: DaskDataFrame, min_coordinate: ArrayLike, max_coordinate: ArrayLike, axes: tuple[str, ...]
) -> DaskDataFrame:
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

    in_bounding_box_masks = []
    for axis_index, axis_name in enumerate(axes):
        min_value = min_coordinate[axis_index]
        in_bounding_box_masks.append(points[axis_name].gt(min_value).to_dask_array(lengths=True))
    for axis_index, axis_name in enumerate(axes):
        max_value = max_coordinate[axis_index]
        in_bounding_box_masks.append(points[axis_name].lt(max_value).to_dask_array(lengths=True))
    in_bounding_box_masks = da.stack(in_bounding_box_masks, axis=-1)
    return da.all(in_bounding_box_masks, axis=1)


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
    # TODO: if the perforamnce are bad for teh translation + scale case, we can replace the dask_image method with a simple image slicing. We can do this when the transformation, in it's affine form, has only zeros outside the diagonal and outside the translation vector. If it has non-zero elements we need to call dask_image. This reasoning applies also to points, polygons and shapes
    from spatialdata._core._spatialdata_ops import (
        get_transformation,
        set_transformation,
    )

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
    assert isinstance(initial_transform, BaseTransformation)

    translation = Translation(translation=request.min_coordinate, axes=request.axes)

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


@singledispatch
def bounding_box_query(
    element: Union[SpatialElement, SpatialData],
    axes: tuple[str, ...],
    min_coordinate: ArrayLike,
    max_coordinate: ArrayLike,
    target_coordinate_system: str,
) -> Union[SpatialElement, SpatialData]:
    raise NotImplementedError()


@bounding_box_query.register(SpatialData)
def _(
    sdata: SpatialData,
    axes: tuple[str, ...],
    min_coordinate: ArrayLike,
    max_coordinate: ArrayLike,
    target_coordinate_system: str,
) -> SpatialData:
    new_elements = {}
    for element_type in ["points", "images", "polygons", "shapes"]:
        elements = getattr(sdata, element_type)
        queried_elements = _dict_query_dispatcher(
            elements,
            bounding_box_query,
            axes=axes,
            min_coordinate=min_coordinate,
            max_coordinate=max_coordinate,
            target_coordinate_system=target_coordinate_system,
        )
        new_elements[element_type] = queried_elements
    return SpatialData(**new_elements, table=sdata.table)


@bounding_box_query.register(DaskDataFrame)
def _(
    points: DaskDataFrame,
    axes: tuple[str, ...],
    min_coordinate: ArrayLike,
    max_coordinate: ArrayLike,
    target_coordinate_system: str,
) -> DaskDataFrame:
    from spatialdata._core._spatialdata_ops import get_transformation

    # get the four corners of the bounding box
    (
        min_coordinate_intrinsic,
        max_coordinate_intrinsic,
        intrinsic_axes,
    ) = _get_bounding_box_corners_in_intrinsic_coordinates(
        element=points,
        min_coordinate=min_coordinate,
        max_coordinate=max_coordinate,
        target_coordinate_system=target_coordinate_system,
        axes=axes,
    )

    # get the points in the intrinsic coordinate bounding box
    in_intrinsic_bounding_box = _bounding_box_mask_points(
        points=points,
        min_coordinate=min_coordinate_intrinsic,
        max_coordinate=max_coordinate_intrinsic,
        axes=intrinsic_axes,
    )
    points_in_intrinsic_bounding_box = points.loc[in_intrinsic_bounding_box]

    if in_intrinsic_bounding_box.sum() == 0:
        # if there aren't any points, just return
        return None

    # we have to reset the index since we have subset
    # https://stackoverflow.com/questions/61395351/how-to-reset-index-on-concatenated-dataframe-in-dask
    points_in_intrinsic_bounding_box = points_in_intrinsic_bounding_box.assign(idx=1)
    points_in_intrinsic_bounding_box = points_in_intrinsic_bounding_box.set_index(
        points_in_intrinsic_bounding_box.idx.cumsum() - 1
    )
    points_in_intrinsic_bounding_box = points_in_intrinsic_bounding_box.map_partitions(
        lambda df: df.rename(index={"idx": None})
    )
    points_in_intrinsic_bounding_box = points_in_intrinsic_bounding_box.drop(columns=["idx"])

    # transform the element to the query coordinate system
    transform_to_query_space = get_transformation(points, to_coordinate_system=target_coordinate_system)
    points_query_coordinate_system = transform_to_query_space.transform(points_in_intrinsic_bounding_box)  # type: ignore[union-attr]

    # get a mask for the points in the bounding box
    bounding_box_mask = _bounding_box_mask_points(
        points=points_query_coordinate_system,
        min_coordinate=min_coordinate,
        max_coordinate=max_coordinate,
        axes=axes,
    )
    if bounding_box_mask.sum() == 0:
        return None
    else:
        return points_in_intrinsic_bounding_box.loc[bounding_box_mask]

    #
    #
    # # which one is needed?
    # t.inverse().to_affine_matrix(input_axes=axes, output_axes=axes)
    #
    # set_transformation(points, t.inverse(), to_coordinate_system='new_target')
    # set_transformation(points, {'new_target': t.inverse()}, set_all=True)
    # return _bounding_box_query_points(points, request, transformation=t)
