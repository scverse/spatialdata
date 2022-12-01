from functools import singledispatch
from typing import Union

from anndata import AnnData
from geopandas import GeoDataFrame
from multiscale_spatial_image import MultiscaleSpatialImage
from spatial_image import SpatialImage

from spatialdata._core.transformations import BaseTransformation

SpatialElement = Union[SpatialImage, MultiscaleSpatialImage, GeoDataFrame, AnnData]

__all__ = ["SpatialElement", "get_transform"]


@singledispatch
def get_transform(e: SpatialElement) -> BaseTransformation:
    raise TypeError(f"Unsupported type: {type(e)}")


@get_transform.register(SpatialImage)
def _(e: SpatialImage) -> BaseTransformation:
    t = e.attrs.get("transform")
    assert isinstance(t, BaseTransformation)
    return t


@get_transform.register(MultiscaleSpatialImage)
def _(e: MultiscaleSpatialImage) -> BaseTransformation:
    t = e.attrs.get("transform")
    assert isinstance(t, BaseTransformation)
    return t


@get_transform.register(GeoDataFrame)
def _(e: GeoDataFrame) -> BaseTransformation:
    t = e.attrs.get("transform")
    assert isinstance(t, BaseTransformation)
    return t


@get_transform.register(AnnData)
def _(e: AnnData) -> BaseTransformation:
    t = e.uns["transform"]
    assert isinstance(t, BaseTransformation)
    return t


@singledispatch
def set_transform(e: SpatialElement, t: BaseTransformation) -> None:
    raise TypeError(f"Unsupported type: {type(e)}")


@set_transform.register(SpatialImage)
def _(e: SpatialImage, t: BaseTransformation) -> None:
    e.attrs["transform"] = t


@set_transform.register(MultiscaleSpatialImage)
def _(e: MultiscaleSpatialImage, t: BaseTransformation) -> None:
    e.attrs["transform"] = t


@set_transform.register(GeoDataFrame)
def _(e: GeoDataFrame, t: BaseTransformation) -> None:
    e.attrs["transform"] = t


@set_transform.register(AnnData)
def _(e: AnnData, t: BaseTransformation) -> None:
    e.uns["transform"] = t
