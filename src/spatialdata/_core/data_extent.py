from __future__ import annotations

from collections import defaultdict
from functools import singledispatch
from typing import Union

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
    centroids = circles["geometry"].centroid
    bounds = pd.DataFrame(
        {
            "minx": centroids.x,
            "maxx": centroids.x,
            "miny": centroids.y,
            "maxy": centroids.y,
        }
    )
    bounds["minx"] -= circles["radius"]
    bounds["miny"] -= circles["radius"]
    bounds["maxx"] += circles["radius"]
    bounds["maxy"] += circles["radius"]

    extent = {}
    for ax in axes:
        extent[ax] = (bounds[f"min{ax}"].min(), bounds[f"max{ax}"].max())
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
    extent = {}
    for ax in axes:
        extent[ax] = (bounds[f"min{ax}"].min(), bounds[f"max{ax}"].max())
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
def get_extent(e: SpatialData | SpatialElement, coordinate_system: str = "global") -> BoundingBoxDescription:
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
    elements: Union[list[str], None] = None,
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
    new_min_coordinates_dict: dict[str, list[float]] = defaultdict(list)
    new_max_coordinates_dict: dict[str, list[float]] = defaultdict(list)
    mask = [has_images, has_labels, has_points, has_shapes]
    include_spatial_elements = ["images", "labels", "points", "shapes"]
    include_spatial_elements = [i for (i, v) in zip(include_spatial_elements, mask) if v]

    if elements is None:  # to shut up ruff
        elements = []
    if not isinstance(elements, list):
        raise ValueError(f"Invalid type of `elements`: {type(elements)}, expected `list`.")

    for element in e._gen_elements():
        element_type, element_name, element_obj = element
        plot_element = (len(elements) == 0) or (element_name in elements)
        plot_element = plot_element and (element_type in include_spatial_elements)
        if plot_element:
            transformations = get_transformation(element_obj, get_all=True)
            assert isinstance(transformations, dict)
            coordinate_systems = list(transformations.keys())
            if coordinate_system in coordinate_systems:
                extent = get_extent(element_obj, coordinate_system=coordinate_system)
                axes = list(extent.keys())
                for ax in axes:
                    new_min_coordinates_dict[ax] += [extent[ax][0]]
                    new_max_coordinates_dict[ax] += [extent[ax][1]]
                if len(axes) == 0:
                    raise ValueError(
                        f"The SpatialData object does not contain any element in the "
                        f" coordinate system {coordinate_system!r}, "
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

    # separate points from (multi-)polygons
    e_points = e_temp[e_temp["geometry"].apply(lambda geom: isinstance(geom, Point))]
    e_polygons = e_temp[e_temp["geometry"].apply(lambda geom: isinstance(geom, (Polygon, MultiPolygon)))]
    extent = None
    if len(e_points) > 0:
        assert "radius" in e_points.columns, "Shapes that are points must have a 'radius' column."
        extent = _get_extent_of_circles(e_points)
    if len(e_polygons) > 0:
        extent_polygons = _get_extent_of_polygons_multipolygons(e_polygons)
        if extent is None:
            extent = extent_polygons
        else:
            # case when there are points AND (multi-)polygons in the GeoDataFrame
            extent["y"] = (min(extent["y"][0], extent_polygons["y"][0]), max(extent["y"][1], extent_polygons["y"][1]))
            extent["x"] = (min(extent["x"][0], extent_polygons["x"][0]), max(extent["x"][1], extent_polygons["x"][1]))

    if extent is None:
        raise ValueError(
            "Unable to compute extent of GeoDataFrame. It needs to contain at least one non-empty "
            "Point or Polygon or Multipolygon."
        )

    min_coordinates = [extent["y"][0], extent["x"][0]]
    max_coordinates = [extent["y"][1], extent["x"][1]]
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
        Min coordinates of the extent in the intrinsic coordinates of the element, expects [y_min, x_min].
    max_coordinates
        Max coordinates of the extent in the intrinsic coordinates of the element, expects [y_max, x_max].
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
    transformed_corners = pd.DataFrame(transform(points, transformation).compute())
    # Make sure min and max values are in the same order as axes
    extent = {}
    for ax in axes:
        extent[ax] = (transformed_corners[ax].min(), transformed_corners[ax].max())
    return extent
