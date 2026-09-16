from __future__ import annotations

from dask.dataframe import DataFrame as DaskDataFrame
from geopandas import GeoDataFrame
from xarray import DataArray, DataTree

from spatialdata.transformations import BaseTransformation

type SpatialElement = DataArray | DataTree | GeoDataFrame | DaskDataFrame
ValidAxis_t = str
MappingToCoordinateSystem_t = dict[str, BaseTransformation]
