from importlib.metadata import version
from typing import Union

__version__ = version("spatialdata")

# Forcing usage of shapely 2.0 by geopandas
# https://geopandas.org/en/stable/getting_started/install.html#using-the-optional-pygeos-dependency
from ._compat import _check_geopandas_using_shapely

_check_geopandas_using_shapely()


__all__ = [
    "concatenate",
    "rasterize",
    "bounding_box_query",
    "transform",
    "SpatialData",
    "element_utils",
    "models",
    "transformations",
    "read_zarr",
]

from spatialdata import element_utils, models, transformations
from spatialdata._core.concatenate import concatenate
from spatialdata._core.operations.rasterize import rasterize
from spatialdata._core.operations.transform import transform
from spatialdata._core.query.spatial_query import bounding_box_query
from spatialdata._core.spatialdata import SpatialData

from ._io.io_zarr import read_zarr

try:
    from spatialdata._dataloader.datasets import ImageTilesDataset
except ImportError as e:
    _error: Union[str, None] = str(e)
else:
    _error = None
