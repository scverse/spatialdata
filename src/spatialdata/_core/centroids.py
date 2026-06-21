from __future__ import annotations

from functools import singledispatch
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr
from anndata import AnnData
from dask.dataframe import DataFrame as DaskDataFrame
from geopandas import GeoDataFrame
from shapely import MultiPolygon, Point, Polygon
from xarray import DataArray, DataTree

from spatialdata._core.operations.transform import transform
from spatialdata._core.spatialdata import SpatialData
from spatialdata.models import get_axes_names, get_table_keys
from spatialdata.models._utils import SpatialElement
from spatialdata.models.models import Labels2DModel, Labels3DModel, PointsModel, ShapesModel, get_model
from spatialdata.transformations.operations import get_transformation
from spatialdata.transformations.transformations import BaseTransformation, Identity

BoundingBoxDescription = dict[str, tuple[float, float]]

PersistAs = Literal["Points", "adata"]


def _validate_coordinate_system(e: SpatialElement, coordinate_system: str) -> None:
    d = get_transformation(e, get_all=True)
    assert isinstance(d, dict)
    assert coordinate_system in d, (
        f"No transformation to coordinate system {coordinate_system} is available for the given element.\n"
        f"Available coordinate systems: {list(d.keys())}"
    )


def _validate_persist_args(persist_as: str, coordinate_system: str | None, *, allow_adata: bool) -> None:
    if persist_as not in ("Points", "adata"):
        raise ValueError(f"`persist_as` must be 'Points' or 'adata', got {persist_as!r}.")
    if persist_as == "adata" and not allow_adata:
        raise ValueError(
            "persist_as='adata' writes centroids into the element's annotating table, which needs the "
            "`SpatialData` object: call `get_centroids(sdata, element_name, ..., persist_as='adata')`. "
            "To get the centroids as a standalone element instead, use persist_as='Points'."
        )
    # ``coordinate_system=None`` means "intrinsic coordinates, do not transform". An intrinsic Points
    # element is ill-defined (Points always carry a coordinate system), so intrinsic coords are only
    # meaningful when writing into a table (persist_as='adata').
    if coordinate_system is None and persist_as != "adata":
        raise ValueError("`coordinate_system=None` (intrinsic coordinates) is only supported with persist_as='adata'.")


def _transform_centroid_coords(
    xy: np.ndarray, axes: list[str], e: SpatialElement, coordinate_system: str | None
) -> np.ndarray:
    """Apply the element's affine to centroid coords in-memory; ``None``/``Identity`` pass through.

    ``axes`` is the column order of ``xy`` (e.g. ``["x", "y"]``).
    """
    if coordinate_system is None:
        return xy
    t = get_transformation(e, coordinate_system)
    assert isinstance(t, BaseTransformation)
    if isinstance(t, Identity):
        return xy
    matrix = t.to_affine_matrix(input_axes=tuple(axes), output_axes=tuple(axes))
    n = len(axes)
    return xy @ matrix[:n, :n].T + matrix[:n, n]


@singledispatch
def get_centroids(
    e: SpatialElement | SpatialData,
    coordinate_system: str | None = "global",
    return_background: bool = False,
    return_area: bool = False,
    persist_as: PersistAs = "Points",
) -> DaskDataFrame | AnnData | None:
    """
    Get the centroids of the geometries contained in a SpatialElement.

    Parameters
    ----------
    e
        The SpatialElement (points, shapes — circles, polygons and multipolygons — or labels), or a
        :class:`~spatialdata.SpatialData` object. When a ``SpatialData`` is passed, the second
        positional argument is the name of the element to measure (see the ``SpatialData`` overload).
    coordinate_system
        The coordinate system in which the centroids are computed. ``None`` returns the intrinsic
        coordinates without applying any transformation (only supported with ``persist_as="adata"``).
    return_background
        If True, the centroid of the background label (0) is included in the output (labels only).
    return_area
        If True, also return the per-instance area: the pixel/voxel count for labels and the geometric
        area for shapes (``pi * r**2`` for circles). Not supported for points (raises). With
        ``persist_as="Points"`` the area is added as a feature column of the returned Points element.
    persist_as
        ``"Points"`` (default) returns the centroids as a new Points element, transformed into
        ``coordinate_system``. ``"adata"`` writes the centroids (and area) into the element's
        annotating table and is only available through the :class:`~spatialdata.SpatialData` overload,
        which can resolve that table.

    Returns
    -------
    A Points element (``persist_as="Points"``). With ``persist_as="adata"`` (``SpatialData`` overload),
    ``None`` when written in place, or the new ``AnnData`` table when ``inplace=False``.

    Notes
    -----
    For :class:`~shapely.Multipolygon`s, the centroids are the average of the centroids of the polygons that constitute
    each :class:`~shapely.Multipolygon`. For multiscale labels the centroids are computed on the full-resolution
    ``scale0`` level.
    """
    raise ValueError(f"The object type {type(e)} is not supported.")


def _get_centroids_for_labels(xdata: xr.DataArray, return_area: bool = False) -> pd.DataFrame:
    """
    Compute centroids for all labels in a DataArray in a single O(n_voxels) pass.

    Works for any number of spatial dimensions (2D and 3D labels). When ``return_area`` is True, an
    ``area`` column (the per-label pixel/voxel count) is added; it is already computed for the
    centroids, so this is free.
    """
    arr = xdata.data.compute()
    axes = list(xdata.dims)

    # Map label values to a contiguous range for bincount efficiency.
    label_ids, inverse = np.unique(arr, return_inverse=True)
    flat_inverse = inverse.ravel()
    counts = np.bincount(flat_inverse)  # per-label pixel counts

    # indexing="ij" (matrix convention) ensures the i-th grid varies along the i-th
    # dimension of the output, correctly aligning with xdata.dims for any number of axes.
    coord_grids = np.meshgrid(*[xdata[ax].values for ax in axes], indexing="ij")
    data: dict[str, np.ndarray] = {}
    for ax, grid in zip(axes, coord_grids, strict=True):
        coord_sums = np.bincount(flat_inverse, weights=grid.ravel().astype(float))
        data[ax] = coord_sums / counts  # counts > 0 by construction (unique guarantees this)

    df = pd.DataFrame(data, index=label_ids)
    if return_area:
        df["area"] = counts.astype(float)
    return df


def _get_centroids_for_shapes(e: GeoDataFrame, return_area: bool) -> tuple[pd.DataFrame, np.ndarray | None]:
    """Intrinsic per-shape centroids (``x, y`` columns indexed by the element's index) and optional area."""
    first_geometry = e["geometry"].iloc[0]
    if isinstance(first_geometry, Point):
        xy = e.geometry.get_coordinates().values
        # shapely .area is 0 for circles (Point geometry); the radius column carries the size.
        area = np.pi * np.asarray(e["radius"], dtype=float) ** 2 if return_area else None
    else:
        assert isinstance(first_geometry, Polygon | MultiPolygon), (
            f"Expected a GeoDataFrame either composed entirely of circles (Points with the `radius` column) or"
            f" Polygons/MultiPolygons. Found {type(first_geometry)} instead."
        )
        xy = e.centroid.get_coordinates().values
        area = e.geometry.area.to_numpy() if return_area else None
    xy_df = pd.DataFrame(xy, columns=["x", "y"], index=e.index.copy())
    return xy_df, area


def _intrinsic_centroid_frame(
    element: SpatialElement, return_background: bool, return_area: bool
) -> tuple[pd.DataFrame, np.ndarray | None, SpatialElement]:
    """Per-instance intrinsic centroids (coordinate columns, indexed by instance id), optional area.

    Also returns the element the centroids live on (for labels, the ``scale0`` level of a multiscale
    raster), which carries the transformation to apply downstream.
    """
    model = get_model(element)
    if model in (Labels2DModel, Labels3DModel):
        raster = next(iter(element["scale0"].values())) if isinstance(element, DataTree) else element
        df = _get_centroids_for_labels(raster, return_area=return_area)
        if not return_background and 0 in df.index:
            df = df.drop(index=0)  # drop the background label (its area, if any, goes with it)
        area = df.pop("area").to_numpy() if return_area else None
        return df, area, raster
    if model is ShapesModel:
        xy_df, area = _get_centroids_for_shapes(element, return_area)
        return xy_df, area, element
    if model is PointsModel:
        if return_area:
            raise ValueError("`return_area` is not supported for points elements (points have no area).")
        axes = get_axes_names(element)
        assert axes in [("x", "y"), ("x", "y", "z")]
        return element[list(axes)].compute(), None, element
    raise ValueError(f"Centroids are not supported for {model.__name__}; expected a Labels, Shapes or Points element.")


def _points_from_centroids(
    df: pd.DataFrame, area: np.ndarray | None, e: SpatialElement, coordinate_system: str
) -> DaskDataFrame:
    """Build a Points element from intrinsic centroids, transformed into ``coordinate_system``."""
    out = df.copy()
    if area is not None:
        out["area"] = np.asarray(area, dtype=float)
    t = get_transformation(e, coordinate_system)
    assert isinstance(t, BaseTransformation)
    points = PointsModel.parse(out, transformations={coordinate_system: t})
    return transform(points, to_coordinate_system=coordinate_system)


@get_centroids.register(DataArray)
@get_centroids.register(DataTree)
@get_centroids.register(GeoDataFrame)
@get_centroids.register(DaskDataFrame)
def _(
    e: SpatialElement,
    coordinate_system: str | None = "global",
    return_background: bool = False,
    return_area: bool = False,
    persist_as: PersistAs = "Points",
) -> DaskDataFrame:
    """Get the centroids of a Labels, Shapes or Points element."""
    _validate_persist_args(persist_as, coordinate_system, allow_adata=False)
    assert coordinate_system is not None  # guaranteed by _validate_persist_args (allow_adata=False)
    _validate_coordinate_system(e, coordinate_system)
    df, area, raster = _intrinsic_centroid_frame(e, return_background, return_area)
    return _points_from_centroids(df, area, raster, coordinate_system)


def _resolve_annotating_table(sdata: SpatialData, element_name: str, table_name: str | None) -> str:
    """Resolve the single table that annotates ``element_name`` (where centroids are written)."""
    from spatialdata._core.query.relational_query import get_element_annotators

    if table_name is not None:
        if table_name not in sdata.tables:
            raise KeyError(f"Table {table_name!r} not found in `sdata.tables`.")
        return table_name
    annotators = sorted(get_element_annotators(sdata, element_name))
    if not annotators:
        raise ValueError(
            f"Element {element_name!r} has no annotating table to write centroids into. Use "
            f"persist_as='Points' to get the centroids as a Points element instead, or annotate the "
            f"element with a table first."
        )
    if len(annotators) > 1:
        raise ValueError(
            f"Element {element_name!r} is annotated by multiple tables ({', '.join(annotators)}); "
            f"pass `table_name=` to choose one."
        )
    return str(annotators[0])


def _write_centroids_into_table(
    table: AnnData,
    element_name: str,
    centroids: pd.DataFrame,
    area: np.ndarray | None,
) -> None:
    """Write centroids into ``obsm["spatial"]`` and area into ``obs["area"]`` at the element's rows.

    Only the table rows annotating ``element_name`` are touched (a table may annotate several
    elements); instances annotated but absent from the element are written as NaN.
    """
    _, region_key, instance_key = get_table_keys(table)
    mask = (table.obs[region_key].astype(str) == str(element_name)).to_numpy()
    if not mask.any():
        raise ValueError(f"The resolved table does not annotate element {element_name!r} (no matching rows).")

    # Map each annotated instance id to its centroid row (-1 where absent, e.g. background or filtered
    # instances -> NaN). A *total* miss means the instance_key and the element index never align (e.g.
    # string vs integer ids); fail loudly rather than silently writing an all-NaN obsm["spatial"].
    keys = table.obs[instance_key].to_numpy()[mask]
    idx = centroids.index.get_indexer(keys)
    if (idx == -1).all():
        raise ValueError(
            f"No instance id annotating {element_name!r} matches a centroid; check that the table's "
            f"`{instance_key}` dtype matches the element's instance ids."
        )
    hit = idx != -1
    coord_cols = list(centroids.columns)
    ndim = len(coord_cols)

    existing = np.asarray(table.obsm["spatial"]) if "spatial" in table.obsm else None
    if existing is not None and existing.shape[0] == table.n_obs and existing.shape[1] != ndim:
        raise ValueError(
            f"Existing obsm['spatial'] has {existing.shape[1]} columns but {element_name!r} centroids have {ndim}; "
            f"refusing to overwrite the coordinates of other regions. Persist these centroids with persist_as='Points'."
        )
    if existing is not None and existing.shape == (table.n_obs, ndim):
        spatial = existing.astype(float, copy=True)
    else:
        spatial = np.full((table.n_obs, ndim), np.nan)
    written = np.full((len(idx), ndim), np.nan)
    written[hit] = centroids[coord_cols].to_numpy(dtype=float)[idx[hit]]
    spatial[mask] = written
    table.obsm["spatial"] = spatial

    if area is not None:
        col = table.obs["area"].to_numpy(dtype=float).copy() if "area" in table.obs else np.full(table.n_obs, np.nan)
        written_area = np.full(len(idx), np.nan)
        written_area[hit] = np.asarray(area, dtype=float)[idx[hit]]
        col[mask] = written_area
        table.obs["area"] = col


@get_centroids.register(SpatialData)
def _get_centroids_sdata(
    e: SpatialData,
    element_name: str,
    coordinate_system: str | None = "global",
    return_background: bool = False,
    return_area: bool = False,
    persist_as: PersistAs = "Points",
    table_name: str | None = None,
    inplace: bool = True,
) -> DaskDataFrame | AnnData | None:
    """Get the centroids of ``element_name``, or (``persist_as="adata"``) write them into its annotating table.

    With ``persist_as="adata"`` the centroids go into ``obsm["spatial"]`` (and area into ``obs["area"]``) of the
    resolved annotating table (``table_name=`` disambiguates). ``inplace=True`` (default) mutates that table and
    returns ``None``; ``inplace=False`` writes into a copy of *only that table* and returns the new ``AnnData``,
    leaving ``e`` untouched. ``persist_as="Points"`` behaves like calling :func:`get_centroids` on the element.
    """
    _validate_persist_args(persist_as, coordinate_system, allow_adata=True)
    element = e[element_name]

    if persist_as == "Points":
        return get_centroids(
            element,
            coordinate_system=coordinate_system,
            return_background=return_background,
            return_area=return_area,
        )

    # persist_as == "adata": resolve the annotating table and write the centroids into it.
    if coordinate_system is not None:
        _validate_coordinate_system(element, coordinate_system)
    table_name = _resolve_annotating_table(e, element_name, table_name)
    df, area, raster = _intrinsic_centroid_frame(element, return_background, return_area)
    coord_cols = sorted(df.columns)  # canonical x, y[, z] (squidpy obsm["spatial"] order)
    coords = _transform_centroid_coords(df[coord_cols].to_numpy(), coord_cols, raster, coordinate_system)
    centroids = pd.DataFrame(coords, columns=coord_cols, index=df.index)

    table = e.tables[table_name] if inplace else e.tables[table_name].copy()
    _write_centroids_into_table(table, element_name, centroids, area)
    return None if inplace else table


##
