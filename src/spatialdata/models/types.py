from __future__ import annotations

from dask.dataframe import DataFrame as DaskDataFrame
from geopandas import GeoDataFrame
from xarray import DataArray, DataTree

SpatialElement = DataArray | DataTree | GeoDataFrame | DaskDataFrame
