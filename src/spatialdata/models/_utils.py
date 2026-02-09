from __future__ import annotations

import warnings
from functools import singledispatch
from typing import TYPE_CHECKING, Any, TypeAlias

import dask.dataframe as dd
import geopandas
import numpy as np
import pandas as pd
from anndata import AnnData
from dask.dataframe import DataFrame as DaskDataFrame
from geopandas import GeoDataFrame
from shapely.geometry import MultiPolygon, Point, Polygon
from xarray import DataArray, DataTree

from spatialdata._logging import logger
from spatialdata._utils import _check_match_length_channels_c_dim
from spatialdata.transformations.transformations import BaseTransformation

SpatialElement: TypeAlias = DataArray | DataTree | GeoDataFrame | DaskDataFrame
TRANSFORM_KEY = "transform"
DEFAULT_COORDINATE_SYSTEM = "global"
ValidAxis_t = str
MappingToCoordinateSystem_t = dict[str, BaseTransformation]
C = "c"
Z = "z"
Y = "y"
X = "x"

if TYPE_CHECKING:
    from spatialdata.models.models import RasterSchema


# mypy says that we can't do isinstance(something, SpatialElement),
# even if the code works fine in my machine. Since the solution described here don't work:
# https://stackoverflow.com/questions/45957615/check-a-variable-against-union-type-at-runtime-in-python-3-6,
# I am just using the function below
def has_type_spatial_element(e: Any) -> bool:
    """
    Check if the object has the type of a SpatialElement.

    Parameters
    ----------
    e
        The input object

    Returns
    -------
    Whether the object is a SpatialElement
    (i.e in DataArray | DataTree | GeoDataFrame | DaskDataFrame)
    """
    return isinstance(e, DataArray | DataTree | GeoDataFrame | DaskDataFrame)


# added this code as part of a refactoring to catch errors earlier
def _validate_mapping_to_coordinate_system_type(transformations: MappingToCoordinateSystem_t | None) -> None:
    if not (
        transformations is None
        or isinstance(transformations, dict)
        and all(isinstance(k, str) and isinstance(v, BaseTransformation) for k, v in transformations.items())
    ):
        raise TypeError(
            f"Transform must be of type {MappingToCoordinateSystem_t} or None, but is of type {type(transformations)}."
        )


def validate_axis_name(axis: ValidAxis_t) -> None:
    """
    Check if the axis name is valid.

    Parameters
    ----------
    axis
        The axis name

    Raises
    ------
    TypeError
        If the axis name not in ["c", "x", "y", "z"]
    """
    if axis not in ["c", "x", "y", "z"]:
        raise TypeError(f"Invalid axis: {axis}")


def validate_axes(axes: tuple[ValidAxis_t, ...]) -> None:
    """
    Check if the names of the axes are valid.

    Parameters
    ----------
    axis
        The names of the axes

    Raises
    ------
    TypeError
        If not in ["c", "x", "y", "z"]
    """
    for ax in axes:
        validate_axis_name(ax)
    if len(axes) != len(set(axes)):
        raise ValueError("Axes must be unique.")


def get_spatial_axes(axes: tuple[ValidAxis_t, ...]) -> tuple[ValidAxis_t, ...]:
    """
    Get the spatial axes of interest.

    Parameters
    ----------
    axes
        Should be a subset of ['x', 'y', 'z', 'c']

    Returns
    -------
    The spatial axes, i.e. the input axes but without 'c'
    """
    validate_axes(axes)
    return tuple(ax for ax in axes if ax in [X, Y, Z])


@singledispatch
def get_axes_names(e: SpatialElement) -> tuple[str, ...]:
    """
    Get the dimensions of a SpatialElement.

    Parameters
    ----------
    e
        SpatialElement

    Returns
    -------
    Dimensions of the SpatialElement (e.g. ("z", "y", "x"))
    """
    raise TypeError(f"Unsupported type: {type(e)}")


@get_axes_names.register(DataArray)
def _(e: DataArray) -> tuple[str, ...]:
    dims = e.dims
    _validate_dims(dims)
    return dims  # type: ignore[no-any-return]


@get_axes_names.register(DataTree)
def _(e: DataTree) -> tuple[str, ...]:
    if "scale0" in e:
        # dims_coordinates = tuple(i for i in e["scale0"].dims.keys())

        assert len(e["scale0"].values()) == 1
        xdata = e["scale0"].values().__iter__().__next__()
        dims_data = xdata.dims
        assert isinstance(dims_data, tuple)

        _validate_dims(dims_data)
        return dims_data
    raise ValueError("Spatialdata DataTree does not contain the scale0 key")


@get_axes_names.register(GeoDataFrame)
def _(e: GeoDataFrame) -> tuple[str, ...]:
    all_dims = (X, Y, Z)
    n = e.geometry.iloc[0]._ndim
    dims = all_dims[:n]
    if Z not in dims and Z in e.columns:
        dims += (Z,)
    _validate_dims(dims)
    return dims


@get_axes_names.register(DaskDataFrame)
def _(e: DaskDataFrame) -> tuple[str, ...]:
    valid_dims = (X, Y, Z)
    dims = tuple([c for c in valid_dims if c in e.columns])
    _validate_dims(dims)
    return dims


def _validate_dims(dims: tuple[str, ...]) -> None:
    for c in dims:
        if c not in (X, Y, Z, C):
            raise ValueError(f"Invalid dimension: {c}")
    if dims not in [(X,), (Y,), (Z,), (C,), (X, Y), (X, Y, Z), (Y, X), (Z, Y, X), (C, Y, X), (C, Z, Y, X)]:
        raise ValueError(f"Invalid dimensions: {dims}")


def points_dask_dataframe_to_geopandas(points: DaskDataFrame, suppress_z_warning: bool = False) -> GeoDataFrame:
    """
    Convert a Dask DataFrame to a GeoDataFrame.

    Parameters
    ----------
    points
        Dask DataFrame with columns "x" and "y". Eventually, it can contain a column "z" that will be not included in
        the geometry column.

    Returns
    -------
    The GeoDataFrame with the geometry column constructed from the "x" and "y" columns and, if present, the rest of the
    columns.

    Notes
    -----
    The "z" column is not included in the geometry column because it is not supported by GeoPandas.
    The resulting GeoDataFrame does not currenlty passes the validation of the SpatialData models. In fact currently
    points need to be saved as a Dask DataFrame. We will be restructuring the models to allow for GeoDataFrames soon.

    """
    from spatialdata.transformations import get_transformation, set_transformation

    if "z" in points.columns and not suppress_z_warning:
        logger.warning("Constructing the GeoDataFrame without considering the z coordinate in the geometry.")

    transformations = get_transformation(points, get_all=True)
    assert isinstance(transformations, dict)
    assert len(transformations) > 0
    points = points.compute()
    points_gdf = GeoDataFrame(points, geometry=geopandas.points_from_xy(points["x"], points["y"]))
    points_gdf.reset_index(drop=True, inplace=True)
    # keep the x and y either in the geometry either as columns: we don't duplicate because having this redundancy could
    # lead to subtle bugs when coverting back to dask dataframes
    points_gdf.drop(columns=["x", "y"], inplace=True)
    set_transformation(points_gdf, transformations, set_all=True)
    return points_gdf


def points_geopandas_to_dask_dataframe(gdf: GeoDataFrame, suppress_z_warning: bool = False) -> DaskDataFrame:
    """
    Convert a GeoDataFrame which represents 2D or 3D points to a Dask DataFrame that passes the schema validation.

    Parameters
    ----------
    gdf
        GeoDataFrame with a geometry column that contains 2D or 3D points.

    Returns
    -------
    The Dask DataFrame converted from the GeoDataFrame. The Dask DataFrame passes the schema validation.

    Notes
    -----
    The returned Dask DataFrame gets the 'x' and 'y' columns from the geometry column, and eventually the 'z' column
    (and the rest of the columns), from the remaining columns of the GeoDataFrame.
    """
    from spatialdata.models import PointsModel

    # transformations are transferred automatically
    ddf = dd.from_pandas(gdf[gdf.columns.drop("geometry")], npartitions=1)
    # we don't want redundancy in the columns since this could lead to subtle bugs when converting back to geopandas
    assert "x" not in ddf.columns
    assert "y" not in ddf.columns
    ddf["x"] = gdf.geometry.x
    ddf["y"] = gdf.geometry.y

    # reorder columns
    axes = ["x", "y", "z"] if "z" in ddf.columns else ["x", "y"]
    non_axes = [c for c in ddf.columns if c not in axes]
    ddf = ddf[axes + non_axes]

    # parse
    if "z" in ddf.columns:
        if not suppress_z_warning:
            logger.warning(
                "Constructing the Dask DataFrame using the x and y coordinates from the geometry and the z from an "
                "additional column."
            )
        ddf = PointsModel.parse(ddf, coordinates={"x": "x", "y": "y", "z": "z"})
    else:
        ddf = PointsModel.parse(ddf, coordinates={"x": "x", "y": "y"})
    return ddf


@singledispatch
def get_channel_names(data: Any) -> list[Any]:
    """Get channels from data for an image element (both single and multiscale).

    Parameters
    ----------
    data
        data to get channels from

    Returns
    -------
    List of channels

    Notes
    -----
    For multiscale images, the channels are validated to be consistent across scales.
    """
    raise ValueError(f"Cannot get channels from {type(data)}")


@get_channel_names.register
def _(data: DataArray) -> list[Any]:
    return data.coords["c"].values.tolist()  # type: ignore[no-any-return]


@get_channel_names.register
def _(data: DataTree) -> list[Any]:
    name = list({list(data[i].data_vars.keys())[0] for i in data})[0]
    channels = {tuple(data[i][name].coords["c"].values.tolist()) for i in data}
    if len(channels) > 1:
        raise ValueError(f"Channels are not consistent across scales: {channels}")
    return list(next(iter(channels)))


def force_2d(gdf: GeoDataFrame) -> None:
    """
    Force the geometries of a shapes object GeoDataFrame to be 2D by modifying the geometries in place.

    Geopandas introduced a method called `force_2d()` to drop the z dimension.
    Unfortunately, this feature, as of geopandas == 0.14.3, is still not released.
    Similarly, the recently released shapely >= 2.0.3 implemented `force_2d()`, but currently there are installation
    errors.

    A similar function has been developed in When `.force_2d()`

    Parameters
    ----------
    gdf
        GeoDataFrame with 2D or 3D geometries

    """
    new_shapes = []
    any_3d = False
    for shape in gdf.geometry:
        if shape.has_z:
            any_3d = True
            if isinstance(shape, Point):
                new_shape = Point(shape.x, shape.y)
            elif isinstance(shape, Polygon):
                new_shape = Polygon(np.array(shape.exterior.coords.xy).T)
            elif isinstance(shape, MultiPolygon):
                new_shape = MultiPolygon([Polygon(np.array(p.exterior.coords.xy).T) for p in shape.geoms])
            else:
                raise ValueError(f"Unsupported geometry type: {type(shape)}")
            new_shapes.append(new_shape)
        else:
            new_shapes.append(shape)
    if any_3d:
        gdf.geometry = new_shapes


def get_raster_model_from_data_dims(dims: tuple[str, ...]) -> type[RasterSchema]:
    """
    Get the raster model from the dimensions of the data.

    Parameters
    ----------
    dims
        The dimensions of the data

    Returns
    -------
    The raster model corresponding to the dimensions of the data.
    """
    from spatialdata.models.models import Image2DModel, Image3DModel, Labels2DModel, Labels3DModel

    if not set(dims).issubset({C, Z, Y, X}):
        raise ValueError(f"Invalid dimensions: {dims}")

    if C in dims:
        return Image3DModel if Z in dims else Image2DModel
    return Labels3DModel if Z in dims else Labels2DModel


def convert_region_column_to_categorical(table: AnnData) -> None:
    from spatialdata.models.models import TableModel

    if TableModel.ATTRS_KEY in table.uns:
        region_key = table.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY_KEY]
        if not isinstance(table.obs[region_key].dtype, pd.CategoricalDtype):
            warnings.warn(
                f"Converting `{TableModel.REGION_KEY_KEY}: {region_key}` to categorical dtype.",
                UserWarning,
                stacklevel=2,
            )
            table.obs[region_key] = pd.Categorical(table.obs[region_key])


def set_channel_names(element: DataArray | DataTree, channel_names: str | list[str]) -> DataArray | DataTree:
    """Set the channel names for a image `SpatialElement` in the `SpatialData` object.

    Parameters
    ----------
    element
        The image `SpatialElement` or parsed `ImageModel`.
    channel_names
        The channel names to be assigned to the c dimension of the image `SpatialElement`.

    Returns
    -------
    The image `SpatialElement` or parsed `ImageModel` with the channel names set to the `c` dimension.
    """
    from spatialdata.models import Image2DModel, Image3DModel, get_model

    channel_names = channel_names if isinstance(channel_names, list) else [channel_names]
    model = get_model(element)

    # get_model cannot be used due to circular import so get_axes_names is used instead
    if model in [Image2DModel, Image3DModel]:
        channel_names = _check_match_length_channels_c_dim(element, channel_names, model.dims)  # type: ignore[union-attr]
        if isinstance(element, DataArray):
            element = element.assign_coords(c=channel_names)
        else:
            element = element.msi.assign_coords({"c": channel_names})
    else:
        raise TypeError("Element model does not support setting channel names, no `c` dimension found.")

    return element


def _get_uint_dtype(value: int) -> str:
    max_uint64 = np.iinfo(np.uint64).max
    max_uint32 = np.iinfo(np.uint32).max
    max_uint16 = np.iinfo(np.uint16).max

    if max_uint16 >= value:
        dtype = "uint16"
    elif max_uint32 >= value:
        dtype = "uint32"
    elif max_uint64 >= value:
        dtype = "uint64"
    else:
        raise ValueError(f"Maximum cell number is {value}. Values higher than {max_uint64} are not supported.")
    return dtype
