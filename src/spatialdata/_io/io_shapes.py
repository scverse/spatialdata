from collections.abc import MutableMapping
from pathlib import Path

import numpy as np
import zarr
from geopandas import GeoDataFrame, read_parquet
from ome_zarr.format import Format
from shapely import from_ragged_array, to_ragged_array

from spatialdata._io._utils import (
    _get_transformations_from_ngff_dict,
    _write_metadata,
    overwrite_coordinate_transformations_non_raster,
)
from spatialdata._io.format import (
    CurrentShapesFormat,
    ShapesFormats,
    ShapesFormatV01,
    ShapesFormatV02,
    _parse_version,
)
from spatialdata.models import ShapesModel, get_axes_names
from spatialdata.transformations._utils import (
    _get_transformations,
    _set_transformations,
)


def _read_shapes(
    store: str | Path | MutableMapping | zarr.Group,  # type: ignore[type-arg]
) -> GeoDataFrame:
    """Read shapes from a zarr store."""
    assert isinstance(store, str | Path)
    f = zarr.open(store, mode="r")
    version = _parse_version(f, expect_attrs_key=True)
    assert version is not None
    format = ShapesFormats[version]

    if isinstance(format, ShapesFormatV01):
        coords = np.array(f["coords"])
        index = np.array(f["Index"])
        typ = format.attrs_from_dict(f.attrs.asdict())
        if typ.name == "POINT":
            radius = np.array(f["radius"])
            geometry = from_ragged_array(typ, coords)
            geo_df = GeoDataFrame({"geometry": geometry, "radius": radius}, index=index)
        else:
            offsets_keys = [k for k in f if k.startswith("offset")]
            offsets = tuple(np.array(f[k]).flatten() for k in offsets_keys)
            geometry = from_ragged_array(typ, coords, offsets)
            geo_df = GeoDataFrame({"geometry": geometry}, index=index)
    elif isinstance(format, ShapesFormatV02):
        path = Path(f._store.path) / f.path / "shapes.parquet"
        geo_df = read_parquet(path)
    else:
        raise ValueError(
            f"Unsupported shapes format {format} from version {version}. Please update the spatialdata library."
        )

    transformations = _get_transformations_from_ngff_dict(f.attrs.asdict()["coordinateTransformations"])
    _set_transformations(geo_df, transformations)
    return geo_df


def write_shapes(
    shapes: GeoDataFrame,
    group: zarr.Group,
    name: str,
    group_type: str = "ngff:shapes",
    format: Format = CurrentShapesFormat(),
) -> None:
    import numcodecs

    axes = get_axes_names(shapes)
    t = _get_transformations(shapes)

    shapes_group = group.require_group(name)

    if isinstance(format, ShapesFormatV01):
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

        attrs = format.attrs_to_dict(geometry)
        attrs["version"] = format.spatialdata_format_version
    elif isinstance(format, ShapesFormatV02):
        path = Path(shapes_group._store.path) / shapes_group.path / "shapes.parquet"
        shapes.to_parquet(path)

        attrs = format.attrs_to_dict(shapes.attrs)
        attrs["version"] = format.spatialdata_format_version
    else:
        raise ValueError(f"Unsupported format version {format.version}. Please update the spatialdata library.")

    _write_metadata(
        shapes_group,
        group_type=group_type,
        axes=list(axes),
        attrs=attrs,
    )
    assert t is not None
    overwrite_coordinate_transformations_non_raster(group=shapes_group, axes=axes, transformations=t)
