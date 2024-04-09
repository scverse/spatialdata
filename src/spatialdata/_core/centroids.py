from __future__ import annotations

from collections import defaultdict
from functools import singledispatch

import dask.array as da
import pandas as pd
import xarray as xr
from dask.dataframe.core import DataFrame as DaskDataFrame
from geopandas import GeoDataFrame
from multiscale_spatial_image import MultiscaleSpatialImage
from shapely import MultiPolygon, Point, Polygon
from spatial_image import SpatialImage

from spatialdata._core.operations.transform import transform
from spatialdata.models import get_axes_names
from spatialdata.models._utils import SpatialElement
from spatialdata.models.models import Image2DModel, Image3DModel, Labels2DModel, Labels3DModel, PointsModel, get_model
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
) -> DaskDataFrame:
    """
    Get the centroids of the geometries contained in a SpatialElement, as a new Points element.

    Parameters
    ----------
    e
        The SpatialElement. Only points, shapes (circles, polygons and multipolygons) and labels are supported.
    coordinate_system
        The coordinate system in which the centroids are computed.

    Notes
    -----
    For :class:`~shapely.Multipolygon`s, the centroids are the average of the centroids of the polygons that constitute
    each :class:`~shapely.Multipolygon`.
    """
    raise ValueError(f"The object type {type(e)} is not supported.")


def _get_centroids_for_axis(xdata: xr.DataArray, axis: str) -> pd.DataFrame:
    """
    Compute the component "axis" of the centroid of each label as a weighted average of the xarray coordinates.

    Parameters
    ----------
    xdata
        The xarray DataArray containing the labels.
    axis
        The axis for which the centroids are computed.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing one column, named after "axis", with the centroids of the labels along that axis.
        The index of the DataFrame is the collection of label values, sorted ascendingly.
    """
    centroids: dict[int, float] = defaultdict(float)
    for i in xdata[axis]:
        portion = xdata.sel(**{axis: i}).data
        u = da.unique(portion, return_counts=True)
        labels_values = u[0].compute()
        counts = u[1].compute()
        for j in range(len(labels_values)):
            label_value = labels_values[j]
            count = counts[j]
            centroids[label_value] += count * i.values.item()

    all_labels_values, all_labels_counts = da.unique(xdata.data, return_counts=True)
    all_labels = dict(zip(all_labels_values.compute(), all_labels_counts.compute()))
    for label_value in centroids:
        centroids[label_value] /= all_labels[label_value]
    centroids = dict(sorted(centroids.items(), key=lambda x: x[0]))
    return pd.DataFrame({axis: centroids.values()}, index=list(centroids.keys()))


@get_centroids.register(SpatialImage)
@get_centroids.register(MultiscaleSpatialImage)
def _(
    e: SpatialImage | MultiscaleSpatialImage,
    coordinate_system: str = "global",
) -> DaskDataFrame:
    """Get the centroids of a Labels element (2D or 3D)."""
    model = get_model(e)
    if model in [Image2DModel, Image3DModel]:
        raise ValueError("Cannot compute centroids for images.")
    assert model in [Labels2DModel, Labels3DModel]
    _validate_coordinate_system(e, coordinate_system)

    if isinstance(e, MultiscaleSpatialImage):
        assert len(e["scale0"]) == 1
        e = SpatialImage(next(iter(e["scale0"].values())))

    dfs = []
    for axis in get_axes_names(e):
        dfs.append(_get_centroids_for_axis(e, axis))
    df = pd.concat(dfs, axis=1)
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
        assert isinstance(first_geometry, (Polygon, MultiPolygon)), (
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
