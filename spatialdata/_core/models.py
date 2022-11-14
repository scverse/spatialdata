"""This file contains models and schema for SpatialData"""

from functools import singledispatch
from typing import Any, Dict, Literal, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from anndata import AnnData
from dask.array.core import Array as DaskArray
from dask.array.core import from_array
from geopandas import GeoDataFrame
from geopandas.array import GeometryDtype
from multiscale_spatial_image import to_multiscale
from multiscale_spatial_image.multiscale_spatial_image import MultiscaleSpatialImage
from multiscale_spatial_image.to_multiscale.to_multiscale import Methods
from numpy.typing import ArrayLike
from pandas.api.types import is_categorical_dtype
from pandera import Column, DataFrameSchema
from shapely import GeometryType
from shapely.io import from_geojson, from_ragged_array
from spatial_image import SpatialImage, to_spatial_image
from xarray_schema.components import ArrayTypeSchema, AttrSchema, DimsSchema
from xarray_schema.dataarray import DataArraySchema

from spatialdata._constants._constants import RasterType
from spatialdata._core.transformations import BaseTransformation, Identity
from spatialdata._logging import logger

# Types
Chunks_t = Union[
    int,
    Tuple[int, ...],
    Tuple[Tuple[int, ...], ...],
    Mapping[Any, Union[None, int, Tuple[int, ...]]],
]
ScaleFactors_t = Sequence[Union[Dict[str, int], int]]

Z, C, Y, X = "z", "c", "y", "x"

Transform_s = AttrSchema(BaseTransformation, None)

Labels2D_s = DataArraySchema(
    dims=DimsSchema((Y, X)),
    array_type=ArrayTypeSchema(DaskArray),
    attrs={"transform": Transform_s},
)
Image2D_s = DataArraySchema(
    dims=DimsSchema((C, Y, X)),
    array_type=ArrayTypeSchema(DaskArray),
    attrs={"transform": Transform_s},
)
Labels3D_s = DataArraySchema(
    dims=DimsSchema((Z, Y, X)),
    array_type=ArrayTypeSchema(DaskArray),
    attrs={"transform": Transform_s},
)
Image3D_s = DataArraySchema(
    dims=DimsSchema((Z, C, Y, X)),
    array_type=ArrayTypeSchema(DaskArray),
    attrs={"transform": Transform_s},
)


def _get_raster_schema(data: ArrayLike, kind: Literal["Image", "Label"]) -> DataArraySchema:
    # get type
    if isinstance(data, SpatialImage):
        shapes: Tuple[Any, ...] = tuple(data.sizes.values())
    elif isinstance(data, DaskArray):
        shapes = data.shape
    elif isinstance(data, MultiscaleSpatialImage):
        k = tuple(data.keys())[0]
        shapes = tuple(data[k].sizes.values())
    else:
        raise TypeError(f"Unsupported type: {type(data)}")

    # get schema
    if len(shapes) == 2:
        return Labels2D_s
    elif len(shapes) == 3:
        if RasterType.IMAGE == RasterType(kind):
            return Image2D_s
        elif RasterType.LABEL == RasterType(kind):
            return Labels3D_s
        else:
            raise ValueError(f"Wrong kind: {kind}")
    elif len(shapes) == 3:
        return Image3D_s
    else:
        raise ValueError(f"Wrong dimensions: {data.dims}D array.")


def _to_spatial_image(
    data: Union[ArrayLike, DaskArray],
    schema: DataArraySchema,
    transform: Optional[Any] = None,
    scale_factors: Optional[ScaleFactors_t] = None,
    method: Optional[Methods] = None,
    chunks: Optional[Chunks_t] = None,
    **kwargs: Any,
) -> Union[SpatialImage, MultiscaleSpatialImage]:
    data = to_spatial_image(array_like=data, dims=schema.dims.dims, **kwargs)
    if transform is None:
        transform = Identity()
    data.attrs = {"transform": transform}

    # TODO(giovp): don't drop coordinates.
    data = data.drop(data.coords.keys())
    if scale_factors is not None:
        data = to_multiscale(
            data,
            scale_factors=scale_factors,
            method=method,
            chunks=chunks,
        )
        # TODO: add transform to multiscale
    return data


@singledispatch
def validate_raster(data: Any, *args: Any, **kwargs: Any) -> Union[SpatialImage, MultiscaleSpatialImage]:
    """
    Validate (or parse) raster data.

    Parameters
    ----------
    data
        Data to validate.
    kind
        Kind of data to validate. Can be "Image" or "Label".
    transform
        Transformation to apply to the data.
    scale_factors
        Scale factors to apply for multiscale.
    method
        Method to use for multiscale.
    chunks
        Chunks to use for dask array.

    Returns
    -------
    :class:`spatial_image.SpatialImage` or
    :class:`multiscale_spatial_image.multiscale_spatial_image.MultiscaleSpatialImage`.
    """
    raise ValueError(f"Unsupported type: {type(data)}")


@validate_raster.register
def _(
    data: np.ndarray,  # type: ignore[type-arg]
    kind: Literal["Image", "Label"],
    *args: Any,
    **kwargs: Any,
) -> Union[SpatialImage, MultiscaleSpatialImage]:

    data = from_array(data)
    schema = _get_raster_schema(data, kind)
    data = _to_spatial_image(data, schema, *args, **kwargs)
    return data


@validate_raster.register
def _(
    data: DaskArray,
    kind: Literal["Image", "Label"],
    *args: Any,
    **kwargs: Any,
) -> Union[SpatialImage, MultiscaleSpatialImage]:

    schema = _get_raster_schema(data, kind)
    data = _to_spatial_image(data, schema, *args, **kwargs)
    return data


@validate_raster.register
def _(
    data: SpatialImage,
    kind: Literal["Image", "Label"],
    **kwargs: Any,
) -> Union[SpatialImage, MultiscaleSpatialImage]:

    schema = _get_raster_schema(data, kind)
    schema.validate(data)
    return data


@validate_raster.register
def _(
    data: MultiscaleSpatialImage,
    kind: Literal["Image", "Label"],
    **kwargs: Any,
) -> Union[SpatialImage, MultiscaleSpatialImage]:

    schema = _get_raster_schema(data, kind)
    # TODO(giovp): get name from multiscale, consider fix upstream
    name = {list(data[i].data_vars.keys())[0] for i in data.keys()}
    if len(name) > 1:
        raise ValueError(f"Wrong name for datatree: {name}.")
    name = list(name)[0]
    for k in data.keys():
        schema.validate(data[k][name])
    return data


# TODO: should check for column be strict?
# TODO: validate attrs for transform.
Polygons_s = DataFrameSchema(
    {
        "geometry": Column(GeometryDtype),
    }
)


@singledispatch
def validate_polygons(data: Any, *args: Any, **kwargs: Any) -> GeoDataFrame:
    """
    Validate (or parse) polygons data.

    Parameters
    ----------
    data
        Data to validate.
    offsets
        TODO.
    geometry
        If 3 is :class:`shapely.geometry.Polygon`, if 6 is :class:`shapely.geometry.MultiPolygon`.
    transform
        Transformation to apply to the data.

    Returns
    -------
    :class:`geopandas.geodataframe.GeoDataFrame`.
    """
    raise ValueError(f"Unsupported type: {type(data)}")


@validate_polygons.register
def _(
    data: np.ndarray,  # type: ignore[type-arg]
    offsets: Tuple[np.ndarray, ...],  # type: ignore[type-arg]
    geometry: Literal[3, 6],
    transform: Optional[Any] = None,
    **kwargs: Any,
) -> GeoDataFrame:

    geometry = GeometryType(geometry)

    data = from_ragged_array(geometry, data, offsets)
    geo_df = GeoDataFrame({"geometry": data})
    if transform is None:
        transform = Identity()
    geo_df.attrs = {"transform": transform}
    Polygons_s.validate(geo_df)
    return geo_df


@validate_polygons.register
def _(
    data: str,
    transform: Optional[Any] = None,
    **kwargs: Any,
) -> GeoDataFrame:

    data = from_geojson(data)
    geo_df = GeoDataFrame({"geometry", data})
    if transform is None:
        transform = Identity()
    geo_df.attrs = {"transform": transform}
    Polygons_s.validate(geo_df)
    return geo_df


@validate_polygons.register
def _(
    data: GeoDataFrame,
    transform: Optional[Any] = None,
    *args: Any,
    **kwargs: Any,
) -> GeoDataFrame:

    Polygons_s.validate(data)
    if transform is None:
        transform = Identity()
    data.attrs = {"transform": transform}
    return data


# TODO: add schema for validation.
@singledispatch
def validate_shapes(data: Any, *args: Any, **kwargs: Any) -> AnnData:
    """
    Validate (or parse) shapes data.

    Parameters
    ----------
    data
        Data to validate.
    shape_type
        Type of shape to validate. Can be "Circle" or "Square".
    shape_size
        Size of shape to validate.

    Returns
    -------
    :class:`anndata.AnnData`.
    """
    raise ValueError(f"Unsupported type: {type(data)}")


@validate_shapes.register
def _(
    data: np.ndarray,  # type: ignore[type-arg]
    shape_type: Literal["Circle", "Square"],
    shape_size: np.float_,
    transform: Optional[Any] = None,
    **kwargs: Any,
) -> AnnData:

    # TODO: AnnData(obsm={"spatial": data}) doesn't work, shall we change?
    adata = AnnData(np.empty(shape=data.shape), obsm={"spatial": data})
    if transform is None:
        transform = Identity()
    adata.uns["transform"] = transform
    adata.uns["spatialdata_attrs"] = {"type": shape_type, "size": shape_size}

    return adata


@validate_shapes.register
def _(
    data: AnnData,
    **kwargs: Any,
) -> AnnData:

    if "spatial" not in data.obsm:
        raise ValueError("AnnData does not contain shapes coordinates in `adata.obsm['spatial']`.")
    if "transform" not in data.uns:
        raise ValueError("AnnData does not contain `transform`.")
    if "spatialdata_attrs" not in data.uns:
        raise ValueError("AnnData does not contain `spatialdata_attrs`.")
    if "type" not in data.uns["spatialdata_attrs"]:
        raise ValueError("AnnData does not contain `spatialdata_attrs['type']`.")
    if "size" not in data.uns["spatialdata_attrs"]:
        raise ValueError("AnnData does not contain `spatialdata_attrs['size']`.")

    return data


# TODO: add schema for validation.
@singledispatch
def validate_table(data: Any, *args: Any, **kwargs: Any) -> AnnData:
    """
    Validate table data.

    Parameters
    ----------
    data
        Data to validate.

    Returns
    -------
    :class:`anndata.AnnData`.
    """
    raise ValueError(f"Unsupported type: {type(data)}")


@validate_table.register
def _(
    data: AnnData,
    region: Union[str, Sequence[str]],
    region_key: Optional[str] = None,
    instance_key: Optional[str] = None,
    **kwargs: Any,
) -> AnnData:

    if region_key not in data.obs:
        if not isinstance(region, str):
            raise ValueError(f"Region key {region_key} not found in `adata.obs`.")

    if isinstance(region, Sequence):
        if region_key not in data.obs:
            raise ValueError(f"Region key {region_key} not found in `adata.obs`.")
        if instance_key not in data.obs:
            raise ValueError(f"Instance key {instance_key} not found in `adata.obs`.")
        if not data.obs["region_key"].isin(region).all():
            raise ValueError(f"`Region key: {region_key}` values do not match with `region` values.")
        if not is_categorical_dtype(data.obs["region_key"]):
            logger.warning(f"Converting `region_key: {region_key}` to categorical dtype.")
            data.obs["region_key"] = pd.Categorical(data.obs["region_key"])

    # TODO: is validation enough?

    attr = {"region": region, "region_key": region_key, "instance_key": instance_key}
    data.uns["spatialdata_attr"] = attr
    return data


# Points (AnnData)?
