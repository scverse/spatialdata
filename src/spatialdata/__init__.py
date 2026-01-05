from importlib.metadata import version

import spatialdata.models._accessor  # noqa: F401

__version__ = version("spatialdata")

__all__ = [
    "models",
    "transformations",
    "datasets",
    "dataloader",
    "concatenate",
    "rasterize",
    "rasterize_bins",
    "rasterize_bins_link_table_to_labels",
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
    "match_sdata_to_table",
    "filter_by_table_query",
    "SpatialData",
    "get_extent",
    "get_centroids",
    "SpatialDataFormatType",
    "read_zarr",
    "unpad_raster",
    "get_pyramid_levels",
    "get_dask_backing_files",
    "are_extents_equal",
    "relabel_sequential",
    "map_raster",
    "deepcopy",
    "sanitize_table",
    "sanitize_name",
    "settings",
]

from spatialdata import dataloader, datasets, models, transformations
from spatialdata._core._deepcopy import deepcopy
from spatialdata._core._utils import sanitize_name, sanitize_table
from spatialdata._core.centroids import get_centroids
from spatialdata._core.concatenate import concatenate
from spatialdata._core.data_extent import are_extents_equal, get_extent
from spatialdata._core.operations.aggregate import aggregate
from spatialdata._core.operations.map import map_raster, relabel_sequential
from spatialdata._core.operations.rasterize import rasterize
from spatialdata._core.operations.rasterize_bins import rasterize_bins, rasterize_bins_link_table_to_labels
from spatialdata._core.operations.transform import transform
from spatialdata._core.operations.vectorize import to_circles, to_polygons
from spatialdata._core.query._utils import get_bounding_box_corners
from spatialdata._core.query.relational_query import (
    filter_by_table_query,
    get_element_annotators,
    get_element_instances,
    get_values,
    join_spatialelement_table,
    match_element_to_table,
    match_sdata_to_table,
    match_table_to_element,
)
from spatialdata._core.query.spatial_query import bounding_box_query, polygon_query
from spatialdata._core.spatialdata import SpatialData
from spatialdata._io._utils import get_dask_backing_files
from spatialdata._io.format import SpatialDataFormatType
from spatialdata._io.io_zarr import read_zarr
from spatialdata._utils import get_pyramid_levels, unpad_raster
from spatialdata.config import settings
