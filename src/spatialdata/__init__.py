from importlib.metadata import version
from typing import Union

__version__ = version("spatialdata")

# Forcing usage of shapely 2.0 by geopandas
# https://geopandas.org/en/stable/getting_started/install.html#using-the-optional-pygeos-dependency
from ._compat import _check_geopandas_using_shapely

_check_geopandas_using_shapely()


__all__ = [
    "SpatialData",
    "read_zarr",
    # --- from spatialdata._core.core_utils ---
    "concatenate",
    # --- from spatialdata._io ---
    "element_utils",
]
from spatialdata import element_utils, models
from spatialdata._core.spatialdata import SpatialData
from spatialdata._core.spatialdata_operations import concatenate

from ._io.io_zarr import read_zarr

try:
    from spatialdata._dataloader.datasets import ImageTilesDataset
except ImportError as e:
    _error: Union[str, None] = str(e)
else:
    _error = None
