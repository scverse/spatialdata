from __future__ import annotations

from collections import defaultdict
from functools import singledispatch
from typing import Any

import numpy as np
import pandas as pd
from dask.dataframe.core import DataFrame as DaskDataFrame
from geopandas import GeoDataFrame
from multiscale_spatial_image import MultiscaleSpatialImage
from shapely import MultiPolygon, Point, Polygon
from spatial_image import SpatialImage
from xarray import DataArray

from spatialdata._core.operations.transform import transform
from spatialdata._core.spatialdata import SpatialData
from spatialdata._types import ArrayLike
from spatialdata.models import get_axes_names
from spatialdata.models._utils import SpatialElement
from spatialdata.models.models import PointsModel
from spatialdata.transformations.operations import get_transformation
from spatialdata.transformations.transformations import (
    BaseTransformation,
)

BoundingBoxDescription = dict[str, tuple[float, float]]


def _get_extent_of_circles(circles: GeoDataFrame) -> BoundingBoxDescription:
    """
    Compute the extent (bounding box) of a set of circles.

    Parameters
    ----------
    circles

    Returns
    -------
    The bounding box description.
    """
    assert isinstance(circles.geometry.iloc[0], Point)
    assert "radius" in circles.columns, "Circles must have a 'radius' column."
    axes = get_axes_names(circles)

    centroids = []
    for dim_name in axes:
        centroids.append(getattr(circles["geometry"], dim_name).to_numpy())
    centroids_array = np.column_stack(centroids)
    radius = np.expand_dims(circles["radius"].to_numpy(), axis=1)

    min_coordinates = np.min(centroids_array - radius, axis=0)
    max_coordinates = np.max(centroids_array + radius, axis=0)

    extent = {}
    for idx, ax in enumerate(axes):
        extent[ax] = (min_coordinates[idx], max_coordinates[idx])
    return extent


def _get_extent_of_polygons_multipolygons(
    shapes: GeoDataFrame,
) -> BoundingBoxDescription:
    """
    Compute the extent (bounding box) of a set of polygons and/or multipolygons.

    Parameters
    ----------
    shapes

    Returns
    -------
    The bounding box description.
    """
    assert isinstance(shapes.geometry.iloc[0], (Polygon, MultiPolygon))
    axes = get_axes_names(shapes)
    bounds = shapes["geometry"].bounds
    # NOTE: this implies the order x, y (which is probably correct?)
    min_coordinates = np.array((bounds["minx"].min(), bounds["miny"].min()))
    max_coordinates = np.array((bounds["maxx"].max(), bounds["maxy"].max()))
    extent = {}
    for idx, ax in enumerate(axes):
        extent[ax] = (min_coordinates[idx], max_coordinates[idx])
    return extent


def _get_extent_of_data_array(e: DataArray, coordinate_system: str) -> BoundingBoxDescription:
    # lightweight conversion to SpatialImage just to fix the type of the single-dispatch
    _check_element_has_coordinate_system(element=SpatialImage(e), coordinate_system=coordinate_system)
    # also here
    data_axes = get_axes_names(SpatialImage(e))
    min_coordinates = []
    max_coordinates = []
    axes = []
    for ax in ["z", "y", "x"]:
        if ax in data_axes:
            i = data_axes.index(ax)
            axes.append(ax)
            min_coordinates.append(0)
            max_coordinates.append(e.shape[i])
    return _compute_extent_in_coordinate_system(
        # and here
        element=SpatialImage(e),
        coordinate_system=coordinate_system,
        min_coordinates=np.array(min_coordinates),
        max_coordinates=np.array(max_coordinates),
        axes=tuple(axes),
    )


@singledispatch
def get_extent(
    e: SpatialData | SpatialElement, coordinate_system: str = "global", **kwargs: Any
) -> BoundingBoxDescription:
    """
    Get the extent (bounding box) of a SpatialData object or a SpatialElement.

    Parameters
    ----------
    e
        The SpatialData object or SpatialElement to computed the extent of.

    Returns
    -------
    min_coordinate
        The minimum coordinate of the bounding box.
    max_coordinate
        The maximum coordinate of the bounding box.
    axes
        The names of the dimensions of the bounding box
    """
    raise ValueError("The object type is not supported.")


@get_extent.register
def _(
    e: SpatialData,
    coordinate_system: str = "global",
    has_images: bool = True,
    has_labels: bool = True,
    has_points: bool = True,
    has_shapes: bool = True,
    elements: list[Any] | None = None,
) -> BoundingBoxDescription:
    """
    Get the extent (bounding box) of a SpatialData object: the extent of the union of the extents of all its elements.

    Parameters
    ----------
    e
        The SpatialData object.

    Returns
    -------
    The bounding box description.
    """
    new_min_coordinates_dict = defaultdict(list)
    new_max_coordinates_dict = defaultdict(list)
    mask = [has_images, has_labels, has_points, has_shapes]
    include_spatial_elements = ["images", "labels", "points", "shapes"]
    include_spatial_elements = [i for (i, v) in zip(include_spatial_elements, mask) if v]

    if elements is None:  # to shut up ruff
        elements = []
    if not isinstance(elements, list):
        raise ValueError(f"Invalid type of `elements`: {type(elements)}, expected `list`.")

    for element in e._gen_elements():
        plot_element = (len(elements) == 0) or (element[1] in elements)
        plot_element = plot_element and (element[0] in include_spatial_elements)
        if plot_element:
            transformations = get_transformation(element[2], get_all=True)
            assert isinstance(transformations, dict)
            coordinate_systems = list(transformations.keys())
            if coordinate_system in coordinate_systems:
                extent = get_extent(element[2], coordinate_system=coordinate_system)
                min_coordinates = [pair[0] for pair in extent.values()]
                max_coordinates = [pair[1] for pair in extent.values()]
                axes = list(extent.keys())
                for i, ax in enumerate(axes):
                    new_min_coordinates_dict[ax].append(min_coordinates[i])
                    new_max_coordinates_dict[ax].append(max_coordinates[i])
    if len(axes) == 0:
        raise ValueError(
            f"The SpatialData object does not contain any element in the coordinate system {coordinate_system!r}, "
            f"please pass a different coordinate system wiht the argument 'coordinate_system'."
        )
    new_min_coordinates = np.array([min(new_min_coordinates_dict[ax]) for ax in axes])
    new_max_coordinates = np.array([max(new_max_coordinates_dict[ax]) for ax in axes])
    extent = {}
    for idx, ax in enumerate(axes):
        extent[ax] = (new_min_coordinates[idx], new_max_coordinates[idx])
    return extent


@get_extent.register
def _(e: GeoDataFrame, coordinate_system: str = "global") -> BoundingBoxDescription:
    """
    Compute the extent (bounding box) of a set of shapes.

    Returns
    -------
    The bounding box description.
    """
    _check_element_has_coordinate_system(element=e, coordinate_system=coordinate_system)
    # remove potentially empty geometries
    e_temp = e[e["geometry"].apply(lambda geom: not geom.is_empty)]
    if isinstance(e_temp.geometry.iloc[0], Point):
        assert "radius" in e_temp.columns, "Shapes must have a 'radius' column."
        extent = _get_extent_of_circles(e_temp)
    else:
        assert isinstance(e_temp.geometry.iloc[0], (Polygon, MultiPolygon)), "Shapes must be polygons or multipolygons."
        extent = _get_extent_of_polygons_multipolygons(e_temp)
    min_coordinates = [pair[0] for pair in extent.values()]
    max_coordinates = [pair[1] for pair in extent.values()]
    axes = tuple(extent.keys())

    return _compute_extent_in_coordinate_system(
        element=e_temp,
        coordinate_system=coordinate_system,
        min_coordinates=np.array(min_coordinates),
        max_coordinates=np.array(max_coordinates),
        axes=axes,
    )


@get_extent.register
def _(e: DaskDataFrame, coordinate_system: str = "global") -> BoundingBoxDescription:
    _check_element_has_coordinate_system(element=e, coordinate_system=coordinate_system)
    axes = get_axes_names(e)
    min_coordinates = np.array([e[ax].min().compute() for ax in axes])
    max_coordinates = np.array([e[ax].max().compute() for ax in axes])
    return _compute_extent_in_coordinate_system(
        element=e,
        coordinate_system=coordinate_system,
        min_coordinates=min_coordinates,
        max_coordinates=max_coordinates,
        axes=axes,
    )


@get_extent.register
def _(e: SpatialImage, coordinate_system: str = "global") -> BoundingBoxDescription:
    return _get_extent_of_data_array(e, coordinate_system=coordinate_system)


@get_extent.register
def _(e: MultiscaleSpatialImage, coordinate_system: str = "global") -> BoundingBoxDescription:
    _check_element_has_coordinate_system(element=e, coordinate_system=coordinate_system)
    xdata = next(iter(e["scale0"].values()))
    return _get_extent_of_data_array(xdata, coordinate_system=coordinate_system)


def _check_element_has_coordinate_system(element: SpatialElement, coordinate_system: str) -> None:
    transformations = get_transformation(element, get_all=True)
    assert isinstance(transformations, dict)
    coordinate_systems = list(transformations.keys())
    if coordinate_system not in coordinate_systems:
        raise ValueError(
            f"The element does not contain any coordinate system named {coordinate_system!r}, "
            f"please pass a different coordinate system wiht the argument 'coordinate_system'."
        )


def _compute_extent_in_coordinate_system(
    element: SpatialElement | DataArray,
    coordinate_system: str,
    min_coordinates: ArrayLike,
    max_coordinates: ArrayLike,
    axes: tuple[str, ...],
) -> BoundingBoxDescription:
    """
    Transform the extent from the intrinsic coordinates of the element to the given coordinate system.

    Parameters
    ----------
    element
        The SpatialElement.
    coordinate_system
        The coordinate system to transform the extent to.
    min_coordinates
        Min coordinates of the extent in the intrinsic coordinates of the element.
    max_coordinates
        Max coordinates of the extent in the intrinsic coordinates of the element.
    axes
        The min and max coordinates refer to.

    Returns
    -------
    The bounding box description in the specified coordinate system.
    """
    transformation = get_transformation(element, to_coordinate_system=coordinate_system)
    assert isinstance(transformation, BaseTransformation)
    from spatialdata._core.query._utils import get_bounding_box_corners

    corners = get_bounding_box_corners(
        axes=axes,
        min_coordinate=min_coordinates,
        max_coordinate=max_coordinates,
    )
    df = pd.DataFrame(corners.data, columns=corners.axis.data.tolist())
    points = PointsModel.parse(df, coordinates={k: k for k in axes})
    transformed_corners = transform(points, transformation).compute()
    # Make sure min and max values are in the same order as axes
    min_coordinates = transformed_corners.min()[list(axes)].to_numpy()
    max_coordinates = transformed_corners.max()[list(axes)].to_numpy()
    extent = {}
    for idx, ax in enumerate(axes):
        extent[ax] = (min_coordinates[idx], max_coordinates[idx])
    return extent
