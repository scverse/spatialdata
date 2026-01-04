# Functions to compute the bounding box describing the extent of a SpatialElement or SpatialData object
from collections import defaultdict
from functools import singledispatch

import numpy as np
import pandas as pd
from dask.dataframe import DataFrame as DaskDataFrame
from geopandas import GeoDataFrame
from shapely import MultiPolygon, Point, Polygon
from xarray import DataArray, DataTree

from spatialdata._core.operations.transform import transform
from spatialdata._core.spatialdata import SpatialData
from spatialdata.models import get_axes_names
from spatialdata.models._utils import SpatialElement
from spatialdata.models.models import PointsModel
from spatialdata.transformations.operations import get_transformation

BoundingBoxDescription = dict[str, tuple[float, float]]


def _get_extent_of_circles(circles: GeoDataFrame) -> BoundingBoxDescription:
    """
    Compute the extent (bounding box) of a set of circles.

    Parameters
    ----------
    circles
        The circles represented as a GeoDataFrame with a `radius` column.

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

    return {ax: (bounds[f"min{ax}"].min(), bounds[f"max{ax}"].max()) for ax in axes}


def _get_extent_of_polygons_multipolygons(
    shapes: GeoDataFrame,
) -> BoundingBoxDescription:
    """
    Compute the extent (bounding box) of a set of polygons and/or multipolygons.

    Parameters
    ----------
    shapes
        The shapes represented as a GeoDataFrame.

    Returns
    -------
    The bounding box description.
    """
    assert isinstance(shapes.geometry.iloc[0], Polygon | MultiPolygon)
    axes = get_axes_names(shapes)
    bounds = shapes["geometry"].bounds
    return {ax: (bounds[f"min{ax}"].min(), bounds[f"max{ax}"].max()) for ax in axes}


def _get_extent_of_points(e: DaskDataFrame) -> BoundingBoxDescription:
    axes = get_axes_names(e)
    mins = dict(e[list(axes)].min().compute())
    maxs = dict(e[list(axes)].max().compute())
    return {ax: (mins[ax], maxs[ax]) for ax in axes}


def _get_extent_of_data_array(e: DataArray, coordinate_system: str) -> BoundingBoxDescription:
    _check_element_has_coordinate_system(element=e, coordinate_system=coordinate_system)
    data_axes = get_axes_names(e)
    extent: BoundingBoxDescription = {}
    for ax in ["z", "y", "x"]:
        if ax in data_axes:
            i = data_axes.index(ax)
            extent[ax] = (0, e.shape[i])
    return _compute_extent_in_coordinate_system(
        element=e,
        coordinate_system=coordinate_system,
        extent=extent,
    )


@singledispatch
def get_extent(
    e: SpatialData | SpatialElement,
    coordinate_system: str = "global",
    exact: bool = True,
    has_images: bool = True,
    has_labels: bool = True,
    has_points: bool = True,
    has_shapes: bool = True,
    elements: list[str] | None = None,  # noqa: UP007 # https://github.com/scverse/spatialdata/pull/318#issuecomment-1755714287
) -> BoundingBoxDescription:
    """
    Get the extent (bounding box) of a SpatialData object or a SpatialElement.

    Parameters
    ----------
    e
        The SpatialData object or SpatialElement to compute the extent of.

    Returns
    -------
    The bounding box description.

    min_coordinate
        The minimum coordinate of the bounding box.
    max_coordinate
        The maximum coordinate of the bounding box.
    axes
        The names of the dimensions of the bounding box.
    exact
        Whether the extent is computed exactly or not.

            - If `True`, the extent is computed exactly.
            - If `False`, an approximation faster to compute is given.

        The approximation is guaranteed to contain all the data, see notes for details.
    has_images
        If `True`, images are included in the computation of the extent.
    has_labels
        If `True`, labels are included in the computation of the extent.
    has_points
        If `True`, points are included in the computation of the extent.
    has_shapes
        If `True`, shapes are included in the computation of the extent.
    elements
        If not `None`, only the elements with the given names are included in the computation of the extent.

    Notes
    -----
    The extent of a `SpatialData` object is the extent of the union of the extents of all its elements.
    The extent of a `SpatialElement` is the extent of the element in the coordinate system
    specified by the argument `coordinate_system`.

    If `exact` is `False`, first the extent of the `SpatialElement` before any transformation is computed.
    Then, the extent is transformed to the target coordinate system. This is faster than computing the extent
    after the transformation, since the transformation is applied to extent of the untransformed data,
    as opposed to transforming the data and then computing the extent.

    The exact and approximate extent are the same if the transformation does not contain any rotation or shear, or in
    the case in which the transformation is affine but all the corners of the extent of the untransformed data
    (bounding box corners) are part of the dataset itself. Note that this is always the case for raster data.

    An extreme case is a dataset composed of the two points `(0, 0)` and `(1, 1)`, rotated anticlockwise by 45 degrees.
    The exact extent is the bounding box `[minx, miny, maxx, maxy] = [0, 0, 0, 1.414]`, while the approximate extent is
    the box `[minx, miny, maxx, maxy] = [-0.707, 0, 0.707, 1.414]`.
    """
    raise ValueError(f"The object type {type(e)} is not supported.")


@get_extent.register
def _(
    e: SpatialData,
    coordinate_system: str = "global",
    exact: bool = True,
    has_images: bool = True,
    has_labels: bool = True,
    has_points: bool = True,
    has_shapes: bool = True,
    elements: list[str] | None = None,
) -> BoundingBoxDescription:
    """
    Get the extent (bounding box) of a SpatialData object.

    The resulting extent is the union of the extents of all its elements.

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
    include_spatial_elements = [i for (i, v) in zip(include_spatial_elements, mask, strict=True) if v]

    if elements is None:  # to shut up ruff
        elements = []
    if not isinstance(elements, list):
        raise ValueError(f"Invalid type of `elements`: {type(elements)}, expected `list`.")

    for element in e._gen_elements():
        element_type, element_name, element_obj = element
        consider_element = (len(elements) == 0) or (element_name in elements)
        consider_element = consider_element and (element_type in include_spatial_elements)
        if consider_element:
            transformations = get_transformation(element_obj, get_all=True)
            assert isinstance(transformations, dict)
            coordinate_systems = list(transformations.keys())
            if coordinate_system in coordinate_systems:
                if isinstance(element_obj, DaskDataFrame | GeoDataFrame):
                    extent = get_extent(element_obj, coordinate_system=coordinate_system, exact=exact)
                else:
                    extent = get_extent(element_obj, coordinate_system=coordinate_system)
                axes = list(extent.keys())
                for ax in axes:
                    new_min_coordinates_dict[ax] += [extent[ax][0]]
                    new_max_coordinates_dict[ax] += [extent[ax][1]]
                if len(axes) == 0:
                    raise ValueError(
                        f"The SpatialData object does not contain any element in the "
                        f" coordinate system {coordinate_system!r}, "
                        f"please pass a different coordinate system with the argument 'coordinate_system'."
                    )
    if len(new_min_coordinates_dict) == 0:
        raise ValueError(
            f"The SpatialData object does not contain any element in the coordinate system {coordinate_system!r}, "
            "please pass a different coordinate system with the argument 'coordinate_system'."
        )
    axes = list(new_min_coordinates_dict.keys())
    new_min_coordinates = np.array([min(new_min_coordinates_dict[ax]) for ax in axes])
    new_max_coordinates = np.array([max(new_max_coordinates_dict[ax]) for ax in axes])
    extent = {}
    for idx, ax in enumerate(axes):
        extent[ax] = (new_min_coordinates[idx], new_max_coordinates[idx])
    return extent


def _get_extent_of_shapes(e: GeoDataFrame) -> BoundingBoxDescription:
    # remove potentially empty geometries
    e_temp = e[e["geometry"].apply(lambda geom: not geom.is_empty)]
    assert len(e_temp) > 0, "Cannot compute extent of an empty collection of geometries."

    # separate points from (multi-)polygons
    first_geometry = e_temp["geometry"].iloc[0]
    if isinstance(first_geometry, Point):
        return _get_extent_of_circles(e)
    assert isinstance(first_geometry, Polygon | MultiPolygon)
    return _get_extent_of_polygons_multipolygons(e)


@get_extent.register
def _(e: GeoDataFrame, coordinate_system: str = "global", exact: bool = True) -> BoundingBoxDescription:
    """
    Get the extent (bounding box) of a SpatialData object.

    The resulting extent is the union of the extents of all its elements.

    Parameters
    ----------
    e
        The SpatialData object.

    Returns
    -------
    The bounding box description.
    """
    _check_element_has_coordinate_system(element=e, coordinate_system=coordinate_system)
    if not exact:
        extent = _get_extent_of_shapes(e)
        return _compute_extent_in_coordinate_system(
            element=e,
            coordinate_system=coordinate_system,
            extent=extent,
        )
    transformed = transform(e, to_coordinate_system=coordinate_system)
    return _get_extent_of_shapes(transformed)


@get_extent.register
def _(e: DaskDataFrame, coordinate_system: str = "global", exact: bool = True) -> BoundingBoxDescription:
    _check_element_has_coordinate_system(element=e, coordinate_system=coordinate_system)
    if not exact:
        extent = _get_extent_of_points(e)
        return _compute_extent_in_coordinate_system(
            element=e,
            coordinate_system=coordinate_system,
            extent=extent,
        )
    transformed = transform(e, to_coordinate_system=coordinate_system)
    return _get_extent_of_points(transformed)


@get_extent.register
def _(e: DataArray, coordinate_system: str = "global") -> BoundingBoxDescription:
    return _get_extent_of_data_array(e, coordinate_system=coordinate_system)


@get_extent.register
def _(e: DataTree, coordinate_system: str = "global") -> BoundingBoxDescription:
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
            f"please pass a different coordinate system with the argument 'coordinate_system'."
        )


def _compute_extent_in_coordinate_system(
    element: SpatialElement | DataArray, coordinate_system: str, extent: BoundingBoxDescription
) -> BoundingBoxDescription:
    """
    Transform the extent from the intrinsic coordinates of the element to the given coordinate system.

    Parameters
    ----------
    element
        The SpatialElement.
    coordinate_system
        The coordinate system to transform the extent to.
    extent
        The extent in the intrinsic coordinates of the element.

    Returns
    -------
    The bounding box description in the specified coordinate system.
    """
    from spatialdata._core.query._utils import get_bounding_box_corners

    axes = get_axes_names(element)
    if "c" in axes:
        axes = tuple(ax for ax in axes if ax != "c")
    min_coordinates = np.array([extent[ax][0] for ax in axes])
    max_coordinates = np.array([extent[ax][1] for ax in axes])
    corners = get_bounding_box_corners(
        axes=axes,
        min_coordinate=min_coordinates,
        max_coordinate=max_coordinates,
    )
    df = pd.DataFrame(corners.data, columns=corners.axis.data.tolist())
    d = get_transformation(element, get_all=True)
    points = PointsModel.parse(df, coordinates={k: k for k in axes}, transformations=d)
    transformed_corners = pd.DataFrame(transform(points, to_coordinate_system=coordinate_system).compute())
    # Make sure min and max values are in the same order as axes
    extent = {}
    for ax in axes:
        extent[ax] = (transformed_corners[ax].min(), transformed_corners[ax].max())
    return extent


def are_extents_equal(extent0: BoundingBoxDescription, extent1: BoundingBoxDescription, atol: float = 0.1) -> bool:
    """
    Check if two data extents, as returned by `get_extent()` are equal up to approximation errors.

    Parameters
    ----------
    extent0
        The first data extent.
    extent1
        The second data extent.
    atol
        The absolute tolerance to use when comparing the extents.

    Returns
    -------
    Whether the extents are equal or not.

    Notes
    -----
    The default value of `atol` is currently high because of a bug of `rasterize()` that makes the extent of the
    rasterized data slightly different from the extent of the original data. This bug is tracked in
    https://github.com/scverse/spatialdata/issues/165
    """
    return all(np.allclose(extent0[k], extent1[k], atol=atol) for k in set(extent0.keys()).union(extent1.keys()))
