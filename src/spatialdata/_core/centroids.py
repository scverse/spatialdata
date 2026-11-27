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


def _validate_coordinate_system(e: SpatialElement, coordinate_system: str) -> dict[str, BaseTransformation]:
    """Return the element's ``{coordinate_system: transformation}`` map, raising if ``coordinate_system`` is absent."""
    d = get_transformation(e, get_all=True)
    assert isinstance(d, dict)
    if coordinate_system not in d:
        raise ValueError(
            f"No transformation to coordinate system {coordinate_system!r} is available for the given element. "
            f"Available coordinate systems: {list(d.keys())}"
        )
    return d


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


def _get_centroids_for_labels(xdata: xr.DataArray, return_area: bool = False) -> pd.DataFrame:
    """
    Compute centroids for all labels in a DataArray in a single O(n_voxels) pass.

    Works for any number of spatial dimensions (2D and 3D labels). When ``return_area`` is True, an
    ``area`` column (the per-label pixel/voxel count) is added; it is already computed for the
    centroids, so this is free.
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

    df = pd.DataFrame(data, index=label_ids)
    if return_area:
        df["area"] = counts.astype(float)
    return df


def _get_centroids_for_shapes(e: GeoDataFrame, return_area: bool) -> tuple[pd.DataFrame, np.ndarray | None]:
    """Intrinsic per-shape centroids (``x, y`` columns indexed by the element's index) and optional area."""
    first_geometry = e["geometry"].iloc[0]
    if isinstance(first_geometry, Point):
        xy = e.geometry.get_coordinates().values
        # shapely .area is 0 for circles (Point geometry); the radius column carries the size.
        area: np.ndarray | None = np.pi * np.asarray(e["radius"], dtype=float) ** 2 if return_area else None
    else:
        assert isinstance(first_geometry, Polygon | MultiPolygon), (
            f"Expected a GeoDataFrame either composed entirely of circles (Points with the `radius` column) or"
            f" Polygons/MultiPolygons. Found {type(first_geometry)} instead."
        )
        xy = e.centroid.get_coordinates().values
        area = e.geometry.area.to_numpy() if return_area else None
    xy_df = pd.DataFrame(xy, columns=["x", "y"], index=e.index.copy())
    return xy_df, area


def _intrinsic_centroid_frame(
    element: SpatialElement, return_background: bool, return_area: bool
) -> tuple[pd.DataFrame, np.ndarray | None, SpatialElement]:
    """Per-instance intrinsic centroids (coordinate columns, indexed by instance id), optional area.

    Also returns the element the centroids live on (for labels, the ``scale0`` level of a multiscale
    raster), which carries the transformation to apply downstream.
    """
    if isinstance(element, GeoDataFrame):
        xy_df, area = _get_centroids_for_shapes(element, return_area)
        return xy_df, area, element
    if isinstance(element, DaskDataFrame):
        if return_area:
            raise ValueError("`return_area` is not supported for points elements (points have no area).")
        axes = get_axes_names(element)
        if axes not in [("x", "y"), ("x", "y", "z")]:
            raise ValueError(f"Expected points axes to be ('x', 'y') or ('x', 'y', 'z'), got {axes}.")
        coords: pd.DataFrame = element[list(axes)].compute()
        return coords, None, element
    model = get_model(element)
    if model not in (Labels2DModel, Labels3DModel):
        raise ValueError("Expected a `Labels` element. Found an `Image` instead.")
    # ``children`` is typed ``Mapping[str, DataTree]`` and ``data_vars`` yields ``DataArray``s.
    raster = element if isinstance(element, DataArray) else next(iter(element.children["scale0"].data_vars.values()))
    df = _get_centroids_for_labels(raster, return_area=return_area)
    if not return_background and 0 in df.index:
        df = df.drop(index=0)  # drop the background label (its area, if any, goes with it)
    area = df.pop("area").to_numpy() if return_area else None
    return df, area, raster


def _points_from_centroids(
    df: pd.DataFrame, area: np.ndarray | None, e: SpatialElement, coordinate_system: str
) -> DaskDataFrame:
    """Build a Points element from intrinsic centroids, transformed into ``coordinate_system``.

    The intrinsic ``area``, if given, is scaled into ``coordinate_system`` as well.
    """
    out = (
        df.assign(area=np.asarray(area, dtype=float) * _area_scale_factor(e, coordinate_system))
        if area is not None
        else df
    )
    t = get_transformation(e, coordinate_system)
    assert isinstance(t, BaseTransformation)
    points = PointsModel.parse(out, transformations={coordinate_system: t})
    transformed: DaskDataFrame = transform(points, to_coordinate_system=coordinate_system)
    return transformed


@get_centroids.register(DataArray)
@get_centroids.register(DataTree)
@get_centroids.register(GeoDataFrame)
@get_centroids.register(DaskDataFrame)
def _(
    e: DataArray | DataTree | GeoDataFrame | DaskDataFrame,
    coordinate_system: str = "global",
    return_background: bool = False,
    return_area: bool = False,
) -> DaskDataFrame:
    """Get the centroids of a Labels, Shapes or Points element."""
    _validate_coordinate_system(e, coordinate_system)
    df, area, raster = _intrinsic_centroid_frame(e, return_background, return_area)
    return _points_from_centroids(df, area, raster, coordinate_system)


##
