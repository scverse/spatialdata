from collections.abc import MutableMapping
from pathlib import Path
from typing import Union

import zarr
from dask.dataframe import DataFrame as DaskDataFrame  # type: ignore[attr-defined]
from dask.dataframe import read_parquet
from ome_zarr.format import Format

from spatialdata._io import SpatialDataFormatV01
from spatialdata._io._utils import (
    _get_transformations_from_ngff_dict,
    _write_metadata,
    overwrite_coordinate_transformations_non_raster,
)
from spatialdata._io.format import CurrentPointsFormat
from spatialdata.models import get_axes_names
from spatialdata.transformations._utils import (
    _get_transformations,
    _set_transformations,
)


def _read_points(
    store: Union[str, Path, MutableMapping, zarr.Group],  # type: ignore[type-arg]
    fmt: SpatialDataFormatV01 = CurrentPointsFormat(),
) -> DaskDataFrame:
    """Read points from a zarr store."""
    assert isinstance(store, (str, Path))
    f = zarr.open(store, mode="r")

    path = Path(f._store.path) / f.path / "points.parquet"
    table = read_parquet(path)
    assert isinstance(table, DaskDataFrame)

    transformations = _get_transformations_from_ngff_dict(f.attrs.asdict()["coordinateTransformations"])
    _set_transformations(table, transformations)

    attrs = fmt.attrs_from_dict(f.attrs.asdict())
    if len(attrs):
        table.attrs["spatialdata_attrs"] = attrs
    return table


def write_points(
    points: DaskDataFrame,
    group: zarr.Group,
    name: str,
    group_type: str = "ngff:points",
    fmt: Format = CurrentPointsFormat(),
) -> None:
    axes = get_axes_names(points)
    t = _get_transformations(points)

    points_groups = group.require_group(name)
    path = Path(points_groups._store.path) / points_groups.path / "points.parquet"
    points.to_parquet(path)

    attrs = fmt.attrs_to_dict(points.attrs)
    attrs["version"] = fmt.version

    _write_metadata(
        points_groups,
        group_type=group_type,
        axes=list(axes),
        attrs=attrs,
        fmt=fmt,
    )
    assert t is not None
    overwrite_coordinate_transformations_non_raster(group=points_groups, axes=axes, transformations=t)
