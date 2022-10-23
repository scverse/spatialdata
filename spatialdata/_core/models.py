"""This file contains models and schema for SpatialData"""

from typing import Any, Dict, Literal, Mapping, Optional, Sequence, Tuple, Union

from dask.array import Array, from_array
from multiscale_spatial_image import to_multiscale
from multiscale_spatial_image.multiscale_spatial_image import MultiscaleSpatialImage
from multiscale_spatial_image.to_multiscale.to_multiscale import Methods
from numpy.typing import ArrayLike
from pandera import Field, SchemaModel
from pandera.typing import Series
from pandera.typing.geopandas import GeoSeries
from spatial_image import SpatialImage, to_spatial_image
from xarray_schema.dataarray import DataArraySchema

from spatialdata._core.transform import Identity

# Types
Chunks_t = Union[
    int,
    Tuple[int, ...],
    Tuple[Tuple[int, ...], ...],
    Mapping[Any, Union[None, int, Tuple[int, ...]]],
]
ScaleFactors_t = Sequence[Union[Dict[str, int], int]]

X, X_t = "x", Literal["x"]
Y, Y_t = "y", Literal["y"]
C, C_t = "c", Literal["c"]
Z, Z_t = "z", Literal["z"]

Labels2D_s = DataArraySchema(dims=(Y, X))
Image2D_s = DataArraySchema(dims=(C, Y, X))

Labels3D_s = DataArraySchema(dims=(Z, Y, X))
Image3D_s = DataArraySchema(dims=(C, Z, Y, X))


def _validate_image(
    data: ArrayLike,
) -> None:
    if isinstance(data, SpatialImage):
        if data.ndim == 2:
            Image2D_s.validate(data)
        elif data.ndim == 3:
            Image3D_s.validate(data)
        else:
            raise ValueError(f"Wrong dimensions: {data.dims}D array.")
    elif isinstance(data, MultiscaleSpatialImage):
        for i in data:
            if len(data[i].dims) == 2:
                Image2D_s.dims.validate(data[i].dims)
            elif len(data[i].dims) == 2:
                Image3D_s.dims.validate(data[i].dims)
            else:
                raise ValueError(f"Wrong dimensions: {data[i].dims}D array.")


def _validate_labels(
    data: ArrayLike,
) -> None:
    if isinstance(data, SpatialImage):
        if data.ndim == 2:
            Labels2D_s.validate(data)
        elif data.ndim == 3:
            Labels3D_s.validate(data)
        else:
            raise ValueError(f"Wrong dimensions: {data.dims}D array.")
    elif isinstance(data, MultiscaleSpatialImage):
        for i in data:
            if len(data[i].dims) == 2:
                Labels2D_s.dims.validate(data[i].dims)
            elif len(data[i].dims) == 2:
                Labels3D_s.dims.validate(data[i].dims)
            else:
                raise ValueError(f"Wrong dimensions: {data[i].dims}D array.")


def _parse_image(
    data: ArrayLike,
    dims: Tuple[int, ...],
    transform: Optional[Any] = None,
    scale_factors: Optional[ScaleFactors_t] = None,
    method: Optional[Methods] = None,
    chunks: Optional[Chunks_t] = None,
    *args: Any,
    **kwargs: Any,
) -> Union[SpatialImage, MultiscaleSpatialImage]:
    if not isinstance(data, Array):
        data = from_array(data)
    image = to_spatial_image(*args, array_like=data, dims=dims, **kwargs)

    if transform is None:
        transform = Identity()
    image.attrs = {"transform": transform}

    if scale_factors is not None:
        # TODO: validate multiscale
        return to_multiscale(image, scale_factors=scale_factors, method=method, chunks=chunks)
    else:
        return image


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
