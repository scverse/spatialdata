from __future__ import annotations

import dask

dask.config.set({"dataframe.query-planning": False})
from dask.dataframe import DASK_EXPR_ENABLED

# Setting `dataframe.query-planning` to False is effective only if run before `dask.dataframe` is initialized. In
# the case in which the user had initilized `dask.dataframe` before, we would have DASK_EXPER_ENABLED set to `True`.
# Here we check that this does not happen.
if DASK_EXPR_ENABLED:
    raise RuntimeError(
        "Unsupported backend: dask-expr has been detected as the backend of dask.dataframe. Please "
        "use:\nimport dask\ndask.config.set({'dataframe.query-planning': False})\nbefore importing "
        "dask.dataframe to disable dask-expr. The support is being worked on, for more information please see"
        "https://github.com/scverse/spatialdata/pull/570"
    )
from importlib.metadata import version

__version__ = version("spatialdata")

__all__ = [
    "models",
    "transformations",
    "datasets",
    "dataloader",
    "concatenate",
    "rasterize",
    "rasterize_bins",
    "to_circles",
    "to_polygons",
    "transform",
    "aggregate",
    "bounding_box_query",
    "polygon_query",
    "get_element_annotators",
    "get_element_instances",
    "get_values",
    "join_spatialelement_table",
    "match_element_to_table",
    "match_table_to_element",
    "SpatialData",
    "get_extent",
    "get_centroids",
    "read_zarr",
    "unpad_raster",
    "save_transformations",
    "get_dask_backing_files",
    "are_extents_equal",
    "map_raster",
    "deepcopy",
]

from spatialdata import dataloader, datasets, models, transformations
from spatialdata._core._deepcopy import deepcopy
from spatialdata._core.centroids import get_centroids
from spatialdata._core.concatenate import concatenate
from spatialdata._core.data_extent import are_extents_equal, get_extent
from spatialdata._core.operations.aggregate import aggregate
from spatialdata._core.operations.map import map_raster
from spatialdata._core.operations.rasterize import rasterize
from spatialdata._core.operations.rasterize_bins import rasterize_bins
from spatialdata._core.operations.transform import transform
from spatialdata._core.operations.vectorize import to_circles, to_polygons
from spatialdata._core.query._utils import get_bounding_box_corners
from spatialdata._core.query.relational_query import (
    get_element_annotators,
    get_element_instances,
    get_values,
    join_spatialelement_table,
    match_element_to_table,
    match_table_to_element,
)
from spatialdata._core.query.spatial_query import bounding_box_query, polygon_query
from spatialdata._core.spatialdata import SpatialData
from spatialdata._io._utils import get_dask_backing_files, save_transformations
from spatialdata._io.io_zarr import read_zarr
from spatialdata._utils import unpad_raster
