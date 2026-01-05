from pathlib import Path
from typing import Any, Literal

import numpy as np
import zarr
from geopandas import GeoDataFrame, read_parquet
from natsort import natsorted
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
    ShapesFormatV03,
    _parse_version,
)
from spatialdata.models import ShapesModel, get_axes_names
from spatialdata.transformations._utils import (
    _get_transformations,
    _set_transformations,
)


def _read_shapes(
    store: str | Path,
) -> GeoDataFrame:
    """Read shapes from a zarr store."""
    f = zarr.open(store, mode="r")
    version = _parse_version(f, expect_attrs_key=True)
    assert version is not None
    shape_format = ShapesFormats[version]

    if isinstance(shape_format, ShapesFormatV01):
        coords = np.array(f["coords"])
        index = np.array(f["Index"])
        typ = shape_format.attrs_from_dict(f.attrs.asdict())
        if typ.name == "POINT":
            radius = np.array(f["radius"])
            geometry = from_ragged_array(typ, coords)
            geo_df = GeoDataFrame({"geometry": geometry, "radius": radius}, index=index)
        else:
            offsets_keys = [k for k in f if k.startswith("offset")]
            offsets_keys = natsorted(offsets_keys)
            offsets = tuple(np.array(f[k]).flatten() for k in offsets_keys)
            geometry = from_ragged_array(typ, coords, offsets)
            geo_df = GeoDataFrame({"geometry": geometry}, index=index)
    elif isinstance(shape_format, ShapesFormatV02 | ShapesFormatV03):
        store_root = f.store_path.store.root
        path = Path(store_root) / f.path / "shapes.parquet"
        geo_df = read_parquet(path)
    else:
        raise ValueError(
            f"Unsupported shapes format {shape_format} from version {version}. Please update the spatialdata library."
        )

    transformations = _get_transformations_from_ngff_dict(f.attrs.asdict()["coordinateTransformations"])
    _set_transformations(geo_df, transformations)
    return geo_df


def write_shapes(
    shapes: GeoDataFrame,
    group: zarr.Group,
    group_type: str = "ngff:shapes",
    element_format: Format = CurrentShapesFormat(),
    geometry_encoding: Literal["WKB", "geoarrow"] | None = None,
) -> None:
    """Write shapes to spatialdata zarr store.

    Note that the parquet file is not recognized as part of the zarr hierarchy as it is not a valid component of a
    zarr store, e.g. group, array or metadata file.

    Parameters
    ----------
    shapes
        The shapes dataframe
    group
        The zarr group in the 'shapes' zarr group to write the shapes element to.
    group_type
        The type of the element.
    element_format
        The format of the shapes element used to store it.
    geometry_encoding
        Whether to use the WKB or geoarrow encoding for GeoParquet. See :meth:`geopandas.GeoDataFrame.to_parquet` for
        details. If None, uses the value from :attr:`spatialdata.settings.shapes_geometry_encoding`.
    """
    from spatialdata.config import settings

    if geometry_encoding is None:
        geometry_encoding = settings.shapes_geometry_encoding

    axes = get_axes_names(shapes)
    transformations = _get_transformations(shapes)
    if transformations is None:
        raise ValueError(f"{group.basename} does not have any transformations and can therefore not be written.")
    if isinstance(element_format, ShapesFormatV01):
        attrs = _write_shapes_v01(shapes, group, element_format)
    elif isinstance(element_format, ShapesFormatV02 | ShapesFormatV03):
        attrs = _write_shapes_v02_v03(shapes, group, element_format, geometry_encoding=geometry_encoding)
    else:
        raise ValueError(f"Unsupported format version {element_format.version}. Please update the spatialdata library.")

    _write_metadata(
        group,
        group_type=group_type,
        axes=list(axes),
        attrs=attrs,
    )
    overwrite_coordinate_transformations_non_raster(group=group, axes=axes, transformations=transformations)


def _write_shapes_v01(shapes: GeoDataFrame, group: zarr.Group, element_format: Format) -> Any:
    """Write shapes to spatialdata zarr store using format ShapesFormatV01.

    Parameters
    ----------
    shapes
        The shapes dataframe
    group
        The zarr group in the 'shapes' zarr group to write the shapes element to.
    element_format
        The format of the shapes element used to store it.
    """
    import numcodecs

    # np.array() creates a writable copy, needed for pandas 3.0 CoW compatibility
    # https://github.com/geopandas/geopandas/issues/3697
    geometry, coords, offsets = to_ragged_array(np.array(shapes.geometry))
    group.create_array(name="coords", data=coords)
    for i, o in enumerate(offsets):
        group.create_array(name=f"offset{i}", data=o)
    if shapes.index.dtype.kind == "U" or shapes.index.dtype.kind == "O":
        group.create_array(name="Index", data=shapes.index.values, dtype=object, object_codec=numcodecs.VLenUTF8())
    else:
        group.create_array(name="Index", data=shapes.index.values)
    if geometry.name == "POINT":
        group.create_array(name=ShapesModel.RADIUS_KEY, data=shapes[ShapesModel.RADIUS_KEY].values)

    attrs = element_format.attrs_to_dict(geometry)
    attrs["version"] = element_format.spatialdata_format_version
    return attrs


def _write_shapes_v02_v03(
    shapes: GeoDataFrame, group: zarr.Group, element_format: Format, geometry_encoding: Literal["WKB", "geoarrow"]
) -> Any:
    """Write shapes to spatialdata zarr store using format ShapesFormatV02 or ShapesFormatV03.

    Parameters
    ----------
    shapes
        The shapes dataframe
    group
        The zarr group in the 'shapes' zarr group to write the shapes element to.
    element_format
        The format of the shapes element used to store it.
    geometry_encoding
        Whether to use the WKB or geoarrow encoding for GeoParquet. See :meth:`geopandas.GeoDataFrame.to_parquet` for
        details.
    """
    from spatialdata.models._utils import TRANSFORM_KEY

    store_root = group.store_path.store.root
    path = store_root / group.path / "shapes.parquet"

    # Temporarily remove transformations from attrs to avoid serialization issues
    transforms = shapes.attrs[TRANSFORM_KEY]
    del shapes.attrs[TRANSFORM_KEY]
    shapes.to_parquet(path, geometry_encoding=geometry_encoding)
    shapes.attrs[TRANSFORM_KEY] = transforms

    attrs = element_format.attrs_to_dict(shapes.attrs)
    attrs["version"] = element_format.spatialdata_format_version
    return attrs
