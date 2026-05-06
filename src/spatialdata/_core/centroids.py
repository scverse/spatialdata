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
    assert coordinate_system in d, (
        f"No transformation to coordinate system {coordinate_system} is available for the given element.\n"
        f"Available coordinate systems: {list(d.keys())}"
    )


@singledispatch
def get_centroids(
    e: SpatialElement,
    coordinate_system: str = "global",
    return_background: bool = False,
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
        If True, the centroid of the background label (0) is included in the output.

    Notes
    -----
    For :class:`~shapely.Multipolygon`s, the centroids are the average of the centroids of the polygons that constitute
    each :class:`~shapely.Multipolygon`.
    """
    raise ValueError(f"The object type {type(e)} is not supported.")


def _get_centroids_for_labels(xdata: xr.DataArray) -> pd.DataFrame:
    """
    Compute centroids for all labels in a DataArray in a single O(n_voxels) pass.

    Works for any number of spatial dimensions (2D and 3D labels).
    """
    arr = xdata.data.compute()
    axes = list(xdata.dims)

    # Map label values to a contiguous range for bincount efficiency.
    label_ids, inverse = np.unique(arr, return_inverse=True)
    flat_inverse = inverse.ravel()

    # indexing="ij" (matrix convention) ensures the i-th grid varies along the i-th
    # dimension of the output, correctly aligning with xdata.dims for any number of axes.
    coord_grids = np.meshgrid(*[xdata[ax].values for ax in axes], indexing="ij")
    data: dict[str, np.ndarray] = {}
    counts = np.bincount(flat_inverse)  # per-label pixel counts
    for ax, grid in zip(axes, coord_grids, strict=True):
        coord_sums = np.bincount(flat_inverse, weights=grid.ravel().astype(float))
        data[ax] = coord_sums / counts  # counts > 0 by construction (unique guarantees this)

    return pd.DataFrame(data, index=label_ids)


@get_centroids.register(DataArray)
@get_centroids.register(DataTree)
def _(
    e: DataArray | DataTree,
    coordinate_system: str = "global",
    return_background: bool = False,
) -> DaskDataFrame:
    """Get the centroids of a Labels element (2D or 3D)."""
    model = get_model(e)
    if model not in [Labels2DModel, Labels3DModel]:
        raise ValueError("Expected a `Labels` element. Found an `Image` instead.")
    _validate_coordinate_system(e, coordinate_system)

    if isinstance(e, DataTree):
        assert len(e["scale0"]) == 1
        e = next(iter(e["scale0"].values()))

    df = _get_centroids_for_labels(e)
    if not return_background and 0 in df.index:
        df = df.drop(index=0)  # drop the background label
    t = get_transformation(e, coordinate_system)
    centroids = PointsModel.parse(df, transformations={coordinate_system: t})
    return transform(centroids, to_coordinate_system=coordinate_system)


@get_centroids.register(GeoDataFrame)
def _(e: GeoDataFrame, coordinate_system: str = "global") -> DaskDataFrame:
    """Get the centroids of a Shapes element (circles or polygons/multipolygons)."""
    _validate_coordinate_system(e, coordinate_system)
    t = get_transformation(e, coordinate_system)
    assert isinstance(t, BaseTransformation)
    # separate points from (multi-)polygons
    first_geometry = e["geometry"].iloc[0]
    if isinstance(first_geometry, Point):
        xy = e.geometry.get_coordinates().values
    else:
        assert isinstance(first_geometry, Polygon | MultiPolygon), (
            f"Expected a GeoDataFrame either composed entirely of circles (Points with the `radius` column) or"
            f" Polygons/MultiPolygons. Found {type(first_geometry)} instead."
        )
        xy = e.centroid.get_coordinates().values
    xy_df = pd.DataFrame(xy, columns=["x", "y"], index=e.index.copy())
    points = PointsModel.parse(xy_df, transformations={coordinate_system: t})
    return transform(points, to_coordinate_system=coordinate_system)


@get_centroids.register(DaskDataFrame)
def _(e: DaskDataFrame, coordinate_system: str = "global") -> DaskDataFrame:
    """Get the centroids of a Points element."""
    _validate_coordinate_system(e, coordinate_system)
    axes = get_axes_names(e)
    assert axes in [("x", "y"), ("x", "y", "z")]
    coords = e[list(axes)].compute()
    t = get_transformation(e, coordinate_system)
    assert isinstance(t, BaseTransformation)
    centroids = PointsModel.parse(coords, transformations={coordinate_system: t})
    return transform(centroids, to_coordinate_system=coordinate_system)


##
