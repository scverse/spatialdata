from __future__ import annotations

from functools import singledispatch
from typing import Any, Optional, Union

import dask.dataframe as dd
import geopandas
from anndata import AnnData
from dask.dataframe import DataFrame as DaskDataFrame
from geopandas import GeoDataFrame
from multiscale_spatial_image import MultiscaleSpatialImage
from spatial_image import SpatialImage

from spatialdata._logging import logger
from spatialdata.transformations.transformations import BaseTransformation

SpatialElement = Union[SpatialImage, MultiscaleSpatialImage, GeoDataFrame, DaskDataFrame]
TRANSFORM_KEY = "transform"
DEFAULT_COORDINATE_SYSTEM = "global"
# ValidAxis_t = Literal["c", "x", "y", "z"]
ValidAxis_t = str
MappingToCoordinateSystem_t = dict[str, BaseTransformation]
C = "c"
Z = "z"
Y = "y"
X = "x"


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
    (i.e in Union[SpatialImage, MultiscaleSpatialImage, GeoDataFrame, DaskDataFrame])
    """
    return isinstance(e, (SpatialImage, MultiscaleSpatialImage, GeoDataFrame, DaskDataFrame))


# added this code as part of a refactoring to catch errors earlier
def _validate_mapping_to_coordinate_system_type(transformations: Optional[MappingToCoordinateSystem_t]) -> None:
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
    Get the dimensions of a spatial element.

    Parameters
    ----------
    e
        Spatial element

    Returns
    -------
    Dimensions of the spatial element (e.g. ("z", "y", "x"))
    """
    raise TypeError(f"Unsupported type: {type(e)}")


@get_axes_names.register(SpatialImage)
def _(e: SpatialImage) -> tuple[str, ...]:
    dims = e.dims
    # dims_sizes = tuple(list(e.sizes.keys()))
    # # we check that the following values are the same otherwise we could incur in subtle bugs downstreams
    # if dims != dims_sizes:
    #     raise ValueError(f"SpatialImage has inconsistent dimensions: {dims}, {dims_sizes}")
    _validate_dims(dims)
    return dims  # type: ignore[no-any-return]


@get_axes_names.register(MultiscaleSpatialImage)
def _(e: MultiscaleSpatialImage) -> tuple[str, ...]:
    if "scale0" in e:
        # dims_coordinates = tuple(i for i in e["scale0"].dims.keys())

        assert len(e["scale0"].values()) == 1
        xdata = e["scale0"].values().__iter__().__next__()
        dims_data = xdata.dims
        assert isinstance(dims_data, tuple)

        # dims_sizes = tuple(list(xdata.sizes.keys()))

        # # we check that all the following values are the same otherwise we could incur in subtle bugs downstreams
        # if dims_coordinates != dims_data or dims_coordinates != dims_sizes:
        #     raise ValueError(
        #         f"MultiscaleSpatialImage has inconsistent dimensions: {dims_coordinates}, {dims_data}, {dims_sizes}"
        #     )
        _validate_dims(dims_data)
        return dims_data
    raise ValueError("MultiscaleSpatialImage does not contain the scale0 key")
    # return tuple(i for i in e.dims.keys())


@get_axes_names.register(GeoDataFrame)
def _(e: GeoDataFrame) -> tuple[str, ...]:
    all_dims = (X, Y, Z)
    n = e.geometry.iloc[0]._ndim
    dims = all_dims[:n]
    _validate_dims(dims)
    return dims


@get_axes_names.register(DaskDataFrame)
def _(e: AnnData) -> tuple[str, ...]:
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
