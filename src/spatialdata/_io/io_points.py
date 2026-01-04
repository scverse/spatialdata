from pathlib import Path

import zarr
from dask.dataframe import DataFrame as DaskDataFrame
from dask.dataframe import read_parquet
from ome_zarr.format import Format

from spatialdata._io._utils import (
    _get_transformations_from_ngff_dict,
    _write_metadata,
    overwrite_coordinate_transformations_non_raster,
)
from spatialdata._io.format import CurrentPointsFormat, PointsFormats, _parse_version
from spatialdata.models import get_axes_names
from spatialdata.transformations._utils import (
    _get_transformations,
    _set_transformations,
)


def _read_points(
    store: str | Path,
) -> DaskDataFrame:
    """Read points from a zarr store."""
    f = zarr.open(store, mode="r")

    version = _parse_version(f, expect_attrs_key=True)
    assert version is not None
    points_format = PointsFormats[version]

    store_root = f.store_path.store.root
    path = store_root / f.path / "points.parquet"
    # cache on remote file needed for parquet reader to work
    # TODO: allow reading in the metadata without caching all the data
    points = read_parquet("simplecache::" + str(path) if str(path).startswith("http") else path)
    assert isinstance(points, DaskDataFrame)

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

    store_root = group.store_path.store.root
    path = store_root / group.path / "points.parquet"

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
    points_without_transform.to_parquet(path)

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
