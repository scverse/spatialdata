from collections.abc import MutableMapping
from pathlib import Path
from typing import Union

import numpy as np
import zarr
from geopandas import GeoDataFrame
from ome_zarr.format import Format
from shapely import from_ragged_array, to_ragged_array

from spatialdata._io import SpatialDataFormatV01
from spatialdata._io._utils import (
    _get_transformations_from_ngff_dict,
    _write_metadata,
    overwrite_coordinate_transformations_non_raster,
)
from spatialdata._io.format import CurrentShapesFormat
from spatialdata.models import ShapesModel, get_axes_names
from spatialdata.transformations._utils import (
    _get_transformations,
    _set_transformations,
)


def _read_shapes(
    store: Union[str, Path, MutableMapping, zarr.Group],  # type: ignore[type-arg]
    fmt: SpatialDataFormatV01 = CurrentShapesFormat(),
) -> GeoDataFrame:
    """Read shapes from a zarr store."""
    assert isinstance(store, (str, Path))
    f = zarr.open(store, mode="r")

    coords = np.array(f["coords"])
    index = np.array(f["Index"])
    typ = fmt.attrs_from_dict(f.attrs.asdict())
    if typ.name == "POINT":
        radius = np.array(f["radius"])
        geometry = from_ragged_array(typ, coords)
        geo_df = GeoDataFrame({"geometry": geometry, "radius": radius}, index=index)
    else:
        offsets_keys = [k for k in f if k.startswith("offset")]
        offsets = tuple(np.array(f[k]).flatten() for k in offsets_keys)
        geometry = from_ragged_array(typ, coords, offsets)
        geo_df = GeoDataFrame({"geometry": geometry}, index=index)

    transformations = _get_transformations_from_ngff_dict(f.attrs.asdict()["coordinateTransformations"])
    _set_transformations(geo_df, transformations)
    return geo_df


def write_shapes(
    shapes: GeoDataFrame,
    group: zarr.Group,
    name: str,
    group_type: str = "ngff:shapes",
    fmt: Format = CurrentShapesFormat(),
) -> None:
    import numcodecs

    axes = get_axes_names(shapes)
    t = _get_transformations(shapes)

    shapes_group = group.require_group(name)
    geometry, coords, offsets = to_ragged_array(shapes.geometry)
    shapes_group.create_dataset(name="coords", data=coords)
    for i, o in enumerate(offsets):
        shapes_group.create_dataset(name=f"offset{i}", data=o)
    if shapes.index.dtype.kind == "U" or shapes.index.dtype.kind == "O":
        shapes_group.create_dataset(
            name="Index", data=shapes.index.values, dtype=object, object_codec=numcodecs.VLenUTF8()
        )
    else:
        shapes_group.create_dataset(name="Index", data=shapes.index.values)
    if geometry.name == "POINT":
        shapes_group.create_dataset(name=ShapesModel.RADIUS_KEY, data=shapes[ShapesModel.RADIUS_KEY].values)

    attrs = fmt.attrs_to_dict(geometry)
    attrs["version"] = fmt.version

    _write_metadata(
        shapes_group,
        group_type=group_type,
        axes=list(axes),
        attrs=attrs,
        fmt=fmt,
    )
    assert t is not None
    overwrite_coordinate_transformations_non_raster(group=shapes_group, axes=axes, transformations=t)
