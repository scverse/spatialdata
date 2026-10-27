from __future__ import annotations

from dask.dataframe import DataFrame as DaskDataFrame
from geopandas import GeoDataFrame
from xarray import DataArray, DataTree

from spatialdata.transformations import BaseTransformation

SpatialElement = DataArray | DataTree | GeoDataFrame | DaskDataFrame
MappingToCoordinateSystem_t = dict[str, BaseTransformation]
