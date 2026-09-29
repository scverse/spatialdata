from __future__ import annotations

from typing import TYPE_CHECKING

from spatialdata.transformations import BaseTransformation

if TYPE_CHECKING:
    from dask.dataframe import DataFrame as DaskDataFrame
    from geopandas import GeoDataFrame
    from xarray import DataArray, DataTree


type SpatialElement = DataArray | DataTree | GeoDataFrame | DaskDataFrame
MappingToCoordinateSystem_t = dict[str, BaseTransformation]
