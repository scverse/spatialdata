from __future__ import annotations

import importlib
from importlib.metadata import version
from typing import TYPE_CHECKING, Any

__version__ = version("spatialdata")

_submodules = {
    "dataloader",
    "datasets",
    "models",
    "transformations",
}

_LAZY_IMPORTS: dict[str, str] = {
    # _core._deepcopy
    "deepcopy": "spatialdata._core._deepcopy",
    # _core._utils
    "sanitize_name": "spatialdata._core._utils",
    "sanitize_table": "spatialdata._core._utils",
    # _core.centroids
    "get_centroids": "spatialdata._core.centroids",
    # _core.concatenate
    "concatenate": "spatialdata._core.concatenate",
    # _core.data_extent
    "are_extents_equal": "spatialdata._core.data_extent",
    "get_extent": "spatialdata._core.data_extent",
    # _core.operations.aggregate
    "aggregate": "spatialdata._core.operations.aggregate",
    # _core.operations.map
    "map_raster": "spatialdata._core.operations.map",
    "relabel_sequential": "spatialdata._core.operations.map",
    # _core.operations.rasterize
    "rasterize": "spatialdata._core.operations.rasterize",
    # _core.operations.rasterize_bins
    "rasterize_bins": "spatialdata._core.operations.rasterize_bins",
    "rasterize_bins_link_table_to_labels": "spatialdata._core.operations.rasterize_bins",
    # _core.operations.transform
    "transform": "spatialdata._core.operations.transform",
    # _core.operations.vectorize
    "to_circles": "spatialdata._core.operations.vectorize",
    "to_polygons": "spatialdata._core.operations.vectorize",
    # _core.query._utils
    "get_bounding_box_corners": "spatialdata._core.query._utils",
    # _core.query.relational_query
    "filter_by_table_query": "spatialdata._core.query.relational_query",
    "get_element_annotators": "spatialdata._core.query.relational_query",
    "get_element_instances": "spatialdata._core.query.relational_query",
    "get_values": "spatialdata._core.query.relational_query",
    "join_spatialelement_table": "spatialdata._core.query.relational_query",
    "match_element_to_table": "spatialdata._core.query.relational_query",
    "match_sdata_to_table": "spatialdata._core.query.relational_query",
    "match_table_to_element": "spatialdata._core.query.relational_query",
    # _core.query.spatial_query
    "bounding_box_query": "spatialdata._core.query.spatial_query",
    "polygon_query": "spatialdata._core.query.spatial_query",
    # _core.spatialdata
    "SpatialData": "spatialdata._core.spatialdata",
    # _io._utils
    "get_dask_backing_files": "spatialdata._io._utils",
    # _io.format
    "SpatialDataFormatType": "spatialdata._io.format",
    # _io.io_zarr
    "read_zarr": "spatialdata._io.io_zarr",
    # _utils
    "disable_dask_tune_optimization": "spatialdata._utils",
    "get_pyramid_levels": "spatialdata._utils",
    "unpad_raster": "spatialdata._utils",
    # config
    "settings": "spatialdata.config",
}

__all__ = [
    # _core._deepcopy
    "deepcopy",
    # _core._utils
    "sanitize_name",
    "sanitize_table",
    # _core.centroids
    "get_centroids",
    # _core.concatenate
    "concatenate",
    # _core.data_extent
    "are_extents_equal",
    "get_extent",
    # _core.operations.aggregate
    "aggregate",
    # _core.operations.map
    "map_raster",
    "relabel_sequential",
    # _core.operations.rasterize
    "rasterize",
    # _core.operations.rasterize_bins
    "rasterize_bins",
    "rasterize_bins_link_table_to_labels",
    # _core.operations.transform
    "transform",
    # _core.operations.vectorize
    "to_circles",
    "to_polygons",
    # _core.query._utils
    "get_bounding_box_corners",
    # _core.query.relational_query
    "filter_by_table_query",
    "get_element_annotators",
    "get_element_instances",
    "get_values",
    "join_spatialelement_table",
    "match_element_to_table",
    "match_sdata_to_table",
    "match_table_to_element",
    # _core.query.spatial_query
    "bounding_box_query",
    "polygon_query",
    # _core.spatialdata
    "SpatialData",
    # _io._utils
    "get_dask_backing_files",
    # _io.format
    "SpatialDataFormatType",
    # _io.io_zarr
    "read_zarr",
    # _utils
    "disable_dask_tune_optimization",
    "get_pyramid_levels",
    "unpad_raster",
    # config
    "settings",
]

_accessor_loaded = False


def __getattr__(name: str) -> Any:
    global _accessor_loaded
    if not _accessor_loaded:
        _accessor_loaded = True
        import spatialdata.models._accessor  # noqa: F401

    if name in _submodules:
        return importlib.import_module(f"spatialdata.{name}")
    if name in _LAZY_IMPORTS:
        mod = importlib.import_module(_LAZY_IMPORTS[name])
        attr = getattr(mod, name)
        globals()[name] = attr
        return attr
    try:
        return globals()[name]
    except KeyError as e:
        raise AttributeError(f"module 'spatialdata' has no attribute {name!r}") from e


def __dir__() -> list[str]:
    return __all__ + ["__version__"]


if TYPE_CHECKING:
    # submodules
    from spatialdata import dataloader, datasets, models, transformations

    # _core._deepcopy
    from spatialdata._core._deepcopy import deepcopy

    # _core._utils
    from spatialdata._core._utils import sanitize_name, sanitize_table

    # _core.centroids
    from spatialdata._core.centroids import get_centroids

    # _core.concatenate
    from spatialdata._core.concatenate import concatenate

    # _core.data_extent
    from spatialdata._core.data_extent import are_extents_equal, get_extent

    # _core.operations.aggregate
    from spatialdata._core.operations.aggregate import aggregate

    # _core.operations.map
    from spatialdata._core.operations.map import map_raster, relabel_sequential

    # _core.operations.rasterize
    from spatialdata._core.operations.rasterize import rasterize

    # _core.operations.rasterize_bins
    from spatialdata._core.operations.rasterize_bins import rasterize_bins, rasterize_bins_link_table_to_labels

    # _core.operations.transform
    from spatialdata._core.operations.transform import transform

    # _core.operations.vectorize
    from spatialdata._core.operations.vectorize import to_circles, to_polygons

    # _core.query._utils
    from spatialdata._core.query._utils import get_bounding_box_corners

    # _core.query.relational_query
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

    # _core.query.spatial_query
    from spatialdata._core.query.spatial_query import bounding_box_query, polygon_query

    # _core.spatialdata
    from spatialdata._core.spatialdata import SpatialData

    # _io._utils
    from spatialdata._io._utils import get_dask_backing_files

    # _io.format
    from spatialdata._io.format import SpatialDataFormatType

    # _io.io_zarr
    from spatialdata._io.io_zarr import read_zarr

    # _utils
    from spatialdata._utils import disable_dask_tune_optimization, get_pyramid_levels, unpad_raster

    # config
    from spatialdata.config import settings
