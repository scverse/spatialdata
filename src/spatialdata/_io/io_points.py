from __future__ import annotations

from pathlib import Path

import zarr
from dask.dataframe import DataFrame as DaskDataFrame
from dask.dataframe import read_parquet
from ome_zarr.format import Format
from upath import UPath

from spatialdata._io._utils import (
    _get_transformations_from_ngff_dict,
    _resolve_zarr_store,
    _write_metadata,
    overwrite_coordinate_transformations_non_raster,
)
from spatialdata._io.format import CurrentPointsFormat, PointsFormats, _parse_version
from spatialdata._store import ZarrStore, make_zarr_store, make_zarr_store_from_group
from spatialdata.models import get_axes_names
from spatialdata.transformations._utils import (
    _get_transformations,
    _set_transformations,
)


def _read_points(
    store: str | Path | UPath | ZarrStore,
) -> DaskDataFrame:
    """Read points from a zarr store (path, hierarchical URI string, or remote ``UPath``)."""
    zarr_store = store if isinstance(store, ZarrStore) else make_zarr_store(store)
    resolved_store = _resolve_zarr_store(zarr_store.path)
    f = zarr.open(resolved_store, mode="r")

    version = _parse_version(f, expect_attrs_key=True)
    assert version is not None
    points_format = PointsFormats[version]

    parquet_store = zarr_store.child("points.parquet")
    # Passing filesystem= to read_parquet makes pyarrow convert dictionary columns into pandas
    # categoricals eagerly per partition and marks them known=True with an empty category list.
    # This happens for ANY pyarrow filesystem (both LocalFileSystem and PyFileSystem(FSSpecHandler(.))
    # return the same broken categorical), so it is a property of the filesystem= handoff itself,
    # not of local-vs-remote. Left as is, it would make write_points' cat.as_known() a no-op and
    # the next to_parquet(filesystem=.) would fail with a per-partition schema mismatch
    # (dictionary<values=null> vs dictionary<values=string>). We demote the categoricals back to
    # "unknown" right here so that write_points recomputes categories consistently across partitions.
    # TODO: allow reading in the metadata without materializing the data.
    points = read_parquet(
        parquet_store.arrow_path(),
        filesystem=parquet_store.arrow_filesystem(),
    )
    assert isinstance(points, DaskDataFrame)
    for column_name in points.columns:
        c = points[column_name]
        if c.dtype == "category" and c.cat.known:
            points[column_name] = c.cat.as_unknown()
    if points.index.name == "__null_dask_index__":
        points = points.rename_axis(None)

    transformations = _get_transformations_from_ngff_dict(f.attrs.asdict()["coordinateTransformations"])
    _set_transformations(points, transformations)

    attrs = points_format.attrs_from_dict(f.attrs.asdict())
    if len(attrs):
        points.attrs["spatialdata_attrs"] = attrs
    return points


def write_points(
    points: DaskDataFrame,
    group: zarr.Group,
    group_type: str = "ngff:points",
    element_format: Format = CurrentPointsFormat(),
) -> None:
    """Write a points element to a zarr store.

    Parameters
    ----------
    points
        The dataframe of the points element.
    group
        The zarr group in the 'points' zarr group to write the points element to.
    group_type
        The type of the element.
    element_format
        The format of the points element used to store it.
    """
    axes = get_axes_names(points)
    transformations = _get_transformations(points)

    parquet_store = make_zarr_store_from_group(group).child("points.parquet")

    # The following code iterates through all columns in the 'points' DataFrame. If the column's datatype is
    # 'category', it checks whether the categories of this column are known. If not, it explicitly converts the
    # categories to known categories using 'c.cat.as_known()' and assigns the transformed Series back to the original
    # DataFrame. This step is crucial when the number of categories exceeds 127, as pyarrow defaults to int8 for
    # unknown categories which can only hold values from -128 to 127.
    for column_name in points.columns:
        c = points[column_name]
        if c.dtype == "category" and not c.cat.known:
            c = c.cat.as_known()
            points[column_name] = c

    points_without_transform = points.copy()
    del points_without_transform.attrs["transform"]
    points_without_transform.to_parquet(
        parquet_store.arrow_path(),
        filesystem=parquet_store.arrow_filesystem(),
    )

    attrs = element_format.attrs_to_dict(points.attrs)
    attrs["version"] = element_format.spatialdata_format_version

    _write_metadata(
        group,
        group_type=group_type,
        axes=list(axes),
        attrs=attrs,
    )
    if transformations is None:
        raise ValueError(f"No transformations specified for element '{group.basename}'. Cannot write.")
    overwrite_coordinate_transformations_non_raster(group=group, axes=axes, transformations=transformations)
