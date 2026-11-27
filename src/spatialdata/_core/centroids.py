from __future__ import annotations

from functools import singledispatch

import numpy as np
import pandas as pd
import xarray as xr
from dask.dataframe import DataFrame as DaskDataFrame
from geopandas import GeoDataFrame
from shapely import MultiPolygon, Point, Polygon
from xarray import DataArray, DataTree

from spatialdata._core.operations.transform import transform
from spatialdata.models import get_axes_names
from spatialdata.models._utils import SpatialElement
from spatialdata.models.models import Labels2DModel, Labels3DModel, PointsModel, get_model
from spatialdata.transformations.operations import get_transformation
from spatialdata.transformations.transformations import BaseTransformation

BoundingBoxDescription = dict[str, tuple[float, float]]


def _validate_coordinate_system(e: SpatialElement, coordinate_system: str) -> None:
    d = get_transformation(e, get_all=True)
    assert isinstance(d, dict)
    if coordinate_system not in d:
        raise ValueError(
            f"No transformation to coordinate system {coordinate_system!r} is available for the given element. "
            f"Available coordinate systems: {list(d.keys())}"
        )


def _area_scale_factor(e: SpatialElement, coordinate_system: str) -> float:
    """Factor by which areas (2D) or volumes (3D) of ``e`` scale when mapped into ``coordinate_system``.

    For an affine map, every area/volume scales by ``|det|`` of its linear part, regardless of position.
    """
    t = get_transformation(e, coordinate_system)
    assert isinstance(t, BaseTransformation)
    axes = get_axes_names(e)
    matrix = t.to_affine_matrix(input_axes=axes, output_axes=axes)
    return float(abs(np.linalg.det(matrix[:-1, :-1])))


@singledispatch
def get_centroids(
    e: SpatialElement,
    coordinate_system: str = "global",
    return_background: bool = False,
    return_area: bool = False,
) -> DaskDataFrame:
    """
    Get the centroids of the geometries contained in a SpatialElement, as a new Points element.

    Parameters
    ----------
    e
        The SpatialElement. Only points, shapes (circles, polygons and multipolygons) and labels are supported.
    coordinate_system
        The coordinate system in which the centroids are computed.
    return_background
        If True, the centroid of the background label (0) is included in the output (labels only).
    return_area
        If True, an ``area`` column with the size of each geometry, measured in ``coordinate_system``, is added.
        Labels are measured by counting pixels, circles as ``pi * radius**2`` and polygons by their geometric
        area; the result is then scaled by the absolute determinant of the linear part of the transformation to
        ``coordinate_system``. For 3D labels the voxels are counted, so ``area`` is a volume (as in
        ``skimage.measure.regionprops``). Not supported for points.

    Returns
    -------
    The centroids, as a points element (Dask DataFrame) in the given coordinate system.

    Notes
    -----
    For :class:`~shapely.MultiPolygon` geometries, the centroid is the geometric (area-weighted) centroid computed by
    ``geopandas`` (``GeoDataFrame.centroid``), i.e. sub-polygons contribute in proportion to their area. For
    multiscale labels, centroids and sizes are computed on the full-resolution ``scale0`` level.
    """
    raise ValueError(f"The object type {type(e)} is not supported.")


def _get_centroids_for_labels(xdata: xr.DataArray) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Compute centroids and pixel (voxel) counts for all labels in a DataArray in a single O(n_voxels) pass.

    Works for any number of spatial dimensions (2D and 3D labels).
    """
    arr = xdata.data.compute()
    axes = list(xdata.dims)

    # Map label values to a contiguous range for bincount efficiency.
    label_ids, inverse = np.unique(arr, return_inverse=True)
    flat_inverse = inverse.ravel()
    counts = np.bincount(flat_inverse)  # per-label pixel counts

    # indexing="ij" (matrix convention) ensures the i-th grid varies along the i-th
    # dimension of the output, correctly aligning with xdata.dims for any number of axes.
    coord_grids = np.meshgrid(*[xdata[ax].to_numpy() for ax in axes], indexing="ij")
    data: dict[str, np.ndarray] = {}
    for ax, grid in zip((str(ax) for ax in axes), coord_grids, strict=True):
        coord_sums = np.bincount(flat_inverse, weights=grid.ravel().astype(float, copy=False))
        data[ax] = coord_sums / counts  # counts > 0 by construction (unique guarantees this)

    return pd.DataFrame(data, index=label_ids), counts


def _points_from_centroids(df: pd.DataFrame, e: SpatialElement, coordinate_system: str) -> DaskDataFrame:
    """Build a Points element from the intrinsic centroids of ``e``, transformed into ``coordinate_system``.

    An ``area`` column, if present, holds intrinsic sizes and is scaled into ``coordinate_system`` as well.
    """
    if "area" in df.columns:
        df = df.assign(area=df["area"] * _area_scale_factor(e, coordinate_system))
    t = get_transformation(e, coordinate_system)
    assert isinstance(t, BaseTransformation)
    points = PointsModel.parse(df, transformations={coordinate_system: t})
    transformed: DaskDataFrame = transform(points, to_coordinate_system=coordinate_system)
    return transformed


@get_centroids.register(DataArray)
@get_centroids.register(DataTree)
def _(
    e: DataArray | DataTree,
    coordinate_system: str = "global",
    return_background: bool = False,
    return_area: bool = False,
) -> DaskDataFrame:
    """Get the centroids of a Labels element (2D or 3D)."""
    model = get_model(e)
    if model not in [Labels2DModel, Labels3DModel]:
        raise ValueError(
            f"Centroids are not supported for elements modeled by {model.__name__}; "
            f"expected a Labels, Shapes or Points element."
        )
    _validate_coordinate_system(e, coordinate_system)

    if isinstance(e, DataTree):
        scale0 = e["scale0"]
        assert isinstance(scale0, DataTree)
        assert len(scale0) == 1
        variable = next(iter(scale0.values()))
        assert isinstance(variable, DataArray)
        e = variable

    df, counts = _get_centroids_for_labels(e)
    if return_area:
        df["area"] = counts.astype(float)
    if not return_background and 0 in df.index:
        df = df.drop(index=0)  # drop the background label
    return _points_from_centroids(df, e, coordinate_system)


@get_centroids.register(GeoDataFrame)
def _(
    e: GeoDataFrame,
    coordinate_system: str = "global",
    return_background: bool = False,
    return_area: bool = False,
) -> DaskDataFrame:
    """Get the centroids of a Shapes element (circles or polygons/multipolygons)."""
    _validate_coordinate_system(e, coordinate_system)
    # separate points from (multi-)polygons
    first_geometry = e["geometry"].iloc[0]
    if isinstance(first_geometry, Point):
        xy = e.geometry.get_coordinates().values
    else:
        assert isinstance(first_geometry, Polygon | MultiPolygon), (
            f"Expected a GeoDataFrame either composed entirely of circles (Points with the `radius` column) or"
            f" Polygons/MultiPolygons. Found {type(first_geometry)} instead."
        )
        xy = e.centroid.get_coordinates().to_numpy()
    xy_df = pd.DataFrame(xy, columns=["x", "y"], index=e.index.copy())
    if return_area:
        # circles are Point geometries (shapely area 0); their size is carried by the radius column
        is_circle = isinstance(first_geometry, Point)
        area = np.pi * e["radius"].to_numpy(dtype=float) ** 2 if is_circle else e.geometry.area.to_numpy()
        xy_df["area"] = area
    return _points_from_centroids(xy_df, e, coordinate_system)


@get_centroids.register(DaskDataFrame)
def _(
    e: DaskDataFrame,
    coordinate_system: str = "global",
    return_background: bool = False,
    return_area: bool = False,
) -> DaskDataFrame:
    """Get the centroids of a Points element."""
    if return_area:
        raise ValueError("`return_area` is not supported for points elements (points have no area).")
    _validate_coordinate_system(e, coordinate_system)
    axes = get_axes_names(e)
    if axes not in [("x", "y"), ("x", "y", "z")]:
        raise ValueError(f"Expected points axes to be ('x', 'y') or ('x', 'y', 'z'), got {axes}.")
    coords = e[list(axes)].compute()
    return _points_from_centroids(coords, e, coordinate_system)


##
