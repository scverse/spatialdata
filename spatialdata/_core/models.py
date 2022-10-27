"""This file contains models and schema for SpatialData"""

from functools import singledispatch
from typing import Any, Dict, Literal, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from dask.array.core import Array as DaskArray
from dask.array.core import from_array
from multiscale_spatial_image import to_multiscale
from multiscale_spatial_image.multiscale_spatial_image import MultiscaleSpatialImage
from multiscale_spatial_image.to_multiscale.to_multiscale import Methods
from numpy.typing import ArrayLike
from pandera import Field, SchemaModel
from pandera.typing import Series
from pandera.typing.geopandas import GeoSeries
from spatial_image import SpatialImage, to_spatial_image
from xarray_schema.components import ArrayTypeSchema, AttrSchema, DimsSchema
from xarray_schema.dataarray import DataArraySchema

from spatialdata._core.transform import BaseTransformation, Identity

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
    if isinstance(data, SpatialImage):
        if len(data.dims) == 2:
            return Labels2D_s
        elif len(data.dims) == 3:
            if kind == "Image":
                return Image2D_s
            elif kind == "Label":
                return Labels3D_s
            else:
                raise ValueError(f"Wrong kind: {kind}")
        elif len(data.dims) == 3:
            return Image3D_s
        else:
            raise ValueError(f"Wrong dimensions: {data.dims}D array.")
    elif isinstance(data, MultiscaleSpatialImage):
        for i in data:
            return _get_raster_schema(data[i], kind)


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
    SpatialImage or MultiscaleSpatialImage.
    """
    raise ValueError(f"Unsupported type: {type(data)}")


@validate_raster.register
def _(
    data: Union[np.ndarray, DaskArray],  # type: ignore[type-arg]
    kind: Literal["Image", "Label"],
    transform: Optional[Any] = None,
    scale_factors: Optional[ScaleFactors_t] = None,
    method: Optional[Methods] = None,
    chunks: Optional[Chunks_t] = None,
    **kwargs: Any,
) -> Union[SpatialImage, MultiscaleSpatialImage]:

    data = from_array(data) if isinstance(data, np.ndarray) else data
    schema = _get_raster_schema(data, kind)

    data = to_spatial_image(array_like=data, dims=schema.dims.dims, **kwargs)
    if transform is None:
        transform = Identity()
        data.attrs = {"transform": transform}
    if scale_factors is not None:
        data = to_multiscale(
            data,
            scale_factors=scale_factors,
            method=method,
            chunks=chunks,
        )
        # TODO: add transform to multiscale
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


class CirclesSchema(SchemaModel):
    geometry: GeoSeries = Field(coerce=True)
    radius: Series[int] = Field(coerce=True)


class SquareSchema(SchemaModel):
    geometry: GeoSeries = Field(coerce=True)
    sides: Optional[Series[int]] = Field(coerce=True)


class PolygonSchema(SchemaModel):
    geometry: GeoSeries = Field(coerce=True)


# Points (AnnData)?
# Tables (AnnData)?
