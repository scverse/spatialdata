from importlib.metadata import version

__version__ = version("spatialdata")

# Forcing usage of shapely 2.0 by geopandas
# https://geopandas.org/en/stable/getting_started/install.html#using-the-optional-pygeos-dependency
from ._compat import _check_geopandas_using_shapely

_check_geopandas_using_shapely()


__all__ = [
    "models",
    "transformations",
    "dataloader",
    "concatenate",
    "rasterize",
    "transform",
    "aggregate",
    "bounding_box_query",
    "polygon_query",
    "get_values",
    "match_table_to_element",
    "SpatialData",
    "read_zarr",
    "unpad_raster",
    "save_transformations",
]

from spatialdata import dataloader, models, transformations
from spatialdata._core.concatenate import concatenate
from spatialdata._core.operations.aggregate import aggregate
from spatialdata._core.operations.rasterize import rasterize
from spatialdata._core.operations.transform import transform
from spatialdata._core.query.relational_query import get_values, match_table_to_element
from spatialdata._core.query.spatial_query import bounding_box_query, polygon_query
from spatialdata._core.spatialdata import SpatialData
from spatialdata._io._utils import save_transformations
from spatialdata._io.io_zarr import read_zarr
from spatialdata._utils import unpad_raster
