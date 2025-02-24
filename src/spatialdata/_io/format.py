from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import ome_zarr.format
import zarr
from anndata import AnnData
from ome_zarr.format import CurrentFormat, Format, FormatV01, FormatV02, FormatV03, FormatV04
from pandas.api.types import CategoricalDtype
from shapely import GeometryType

from spatialdata.models.models import ATTRS_KEY, PointsModel, ShapesModel

CoordinateTransform_t = list[dict[str, Any]]

Shapes_s = ShapesModel()
Points_s = PointsModel()


def _parse_version(group: zarr.Group, expect_attrs_key: bool) -> str | None:
    """
    Parse the version of the spatialdata encoding for the element.

    Parameters
    ----------
    group
        The Zarr group where the element is stored.
    expect_attrs_key
        Boolean value specifying if the version is expected to be found as a key-value store in the .attrs[ATTRS_KEY],
         where .attrs is the attrs of the Zarr group (expect_attrs_key == True), or if the version is expected to be
         found as key-value store directly in .attrs

    Returns
    -------
    The string specifying the encoding version of the element, if found. `None` otherwise.
    """
    if expect_attrs_key and ATTRS_KEY not in group.attrs:
        return None
    attrs_key_group = group.attrs[ATTRS_KEY] if expect_attrs_key else group.attrs
    version_found = "version" in attrs_key_group
    if not version_found:
        return None
    version = attrs_key_group["version"]
    assert isinstance(version, str)
    return version


class SpatialDataFormat(CurrentFormat):
    pass


class SpatialDataContainerFormatV01(SpatialDataFormat):
    @property
    def spatialdata_format_version(self) -> str:
        return "0.1"

    def attrs_from_dict(self, metadata: dict[str, Any]) -> dict[str, Any]:
        return {}

    def attrs_to_dict(self) -> dict[str, str | dict[str, Any]]:
        from spatialdata import __version__

        return {"spatialdata_software_version": __version__}


class RasterFormatV01(SpatialDataFormat):
    """Formatter for raster data."""

    def generate_coordinate_transformations(self, shapes: list[tuple[Any]]) -> None | list[list[dict[str, Any]]]:
        data_shape = shapes[0]
        coordinate_transformations: list[list[dict[str, Any]]] = []
        # calculate minimal 'scale' transform based on pyramid dims
        for shape in shapes:
            assert len(shape) == len(data_shape)
            scale = [full / level for full, level in zip(data_shape, shape, strict=True)]
            from spatialdata.transformations.ngff.ngff_transformations import NgffScale

            coordinate_transformations.append([NgffScale(scale=scale).to_dict()])
        return coordinate_transformations

    def validate_coordinate_transformations(
        self,
        ndim: int,
        nlevels: int,
        coordinate_transformations: None | list[list[dict[str, Any]]] = None,
    ) -> None:
        """
        Validate that a list of dicts contains a 'scale' transformation.

        Raises ValueError if no 'scale' found or doesn't match ndim
        :param ndim:Number of image dimensions.
        """
        if coordinate_transformations is None:
            raise ValueError("coordinate_transformations must be provided")
        ct_count = len(coordinate_transformations)
        if ct_count != nlevels:
            raise ValueError(f"coordinate_transformations count: {ct_count} must match datasets {nlevels}")
        for transformations in coordinate_transformations:
            assert isinstance(transformations, list)
            types = [t.get("type", None) for t in transformations]
            if any(t is None for t in types):
                raise ValueError(f"Missing type in: {transformations}")

            # new validation
            import json

            json0 = [json.dumps(t) for t in transformations]
            from spatialdata.transformations.ngff.ngff_transformations import (
                NgffBaseTransformation,
            )

            parsed = [NgffBaseTransformation.from_dict(t) for t in transformations]
            json1 = [json.dumps(p.to_dict()) for p in parsed]
            import numpy as np

            assert np.all([j0 == j1 for j0, j1 in zip(json0, json1, strict=True)])

    # eventually we are fully compliant with NGFF and we can drop SPATIALDATA_FORMAT_VERSION and simply rely on
    # "version"; still, until the coordinate transformations make it into NGFF, we need to have our extension
    @property
    def spatialdata_format_version(self) -> str:
        return "0.1"

    @property
    def version(self) -> str:
        return "0.4"


class RasterFormatV02(RasterFormatV01):
    @property
    def spatialdata_format_version(self) -> str:
        return "0.2"

    @property
    def version(self) -> str:
        # 0.1 -> 0.2 changed the version string for the NGFF format, from 0.4 to 0.6-dev-spatialdata as discussed here
        # https://github.com/scverse/spatialdata/pull/849
        return "0.4-dev-spatialdata"


class ShapesFormatV01(SpatialDataFormat):
    """Formatter for shapes."""

    @property
    def spatialdata_format_version(self) -> str:
        return "0.1"

    def attrs_from_dict(self, metadata: dict[str, Any]) -> GeometryType:
        if Shapes_s.ATTRS_KEY not in metadata:
            raise KeyError(f"Missing key {Shapes_s.ATTRS_KEY} in shapes metadata.")
        metadata_ = metadata[Shapes_s.ATTRS_KEY]
        if Shapes_s.GEOS_KEY not in metadata_:
            raise KeyError(f"Missing key {Shapes_s.GEOS_KEY} in shapes metadata.")
        for k in [Shapes_s.TYPE_KEY, Shapes_s.NAME_KEY]:
            if k not in metadata_[Shapes_s.GEOS_KEY]:
                raise KeyError(f"Missing key {k} in shapes metadata.")

        typ = GeometryType(metadata_[Shapes_s.GEOS_KEY][Shapes_s.TYPE_KEY])
        assert typ.name == metadata_[Shapes_s.GEOS_KEY][Shapes_s.NAME_KEY]
        assert self.spatialdata_format_version == metadata_["version"]
        return typ

    def attrs_to_dict(self, geometry: GeometryType) -> dict[str, str | dict[str, Any]]:
        return {Shapes_s.GEOS_KEY: {Shapes_s.NAME_KEY: geometry.name, Shapes_s.TYPE_KEY: geometry.value}}


class ShapesFormatV02(SpatialDataFormat):
    """Formatter for shapes."""

    @property
    def spatialdata_format_version(self) -> str:
        return "0.2"

    # no need for attrs_from_dict as we are not saving metadata except for the coordinate transformations
    def attrs_to_dict(self, data: dict[str, Any]) -> dict[str, str | dict[str, Any]]:
        return {}


class PointsFormatV01(SpatialDataFormat):
    """Formatter for points."""

    @property
    def spatialdata_format_version(self) -> str:
        return "0.1"

    def attrs_from_dict(self, metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
        if Points_s.ATTRS_KEY not in metadata:
            raise KeyError(f"Missing key {Points_s.ATTRS_KEY} in points metadata.")
        metadata_ = metadata[Points_s.ATTRS_KEY]
        assert self.spatialdata_format_version == metadata_["version"]
        d = {}
        if Points_s.FEATURE_KEY in metadata_:
            d[Points_s.FEATURE_KEY] = metadata_[Points_s.FEATURE_KEY]
        if Points_s.INSTANCE_KEY in metadata_:
            d[Points_s.INSTANCE_KEY] = metadata_[Points_s.INSTANCE_KEY]
        return d

    def attrs_to_dict(self, data: dict[str, Any]) -> dict[str, dict[str, Any]]:
        d = {}
        if Points_s.ATTRS_KEY in data:
            if Points_s.INSTANCE_KEY in data[Points_s.ATTRS_KEY]:
                d[Points_s.INSTANCE_KEY] = data[Points_s.ATTRS_KEY][Points_s.INSTANCE_KEY]
            if Points_s.FEATURE_KEY in data[Points_s.ATTRS_KEY]:
                d[Points_s.FEATURE_KEY] = data[Points_s.ATTRS_KEY][Points_s.FEATURE_KEY]
        return d


class TablesFormatV01(SpatialDataFormat):
    """Formatter for the table."""

    @property
    def spatialdata_format_version(self) -> str:
        return "0.1"

    def validate_table(
        self,
        table: AnnData,
        region_key: None | str = None,
        instance_key: None | str = None,
    ) -> None:
        if not isinstance(table, AnnData):
            raise TypeError(f"`table` must be `anndata.AnnData`, was {type(table)}.")
        if region_key is not None and not isinstance(table.obs[region_key].dtype, CategoricalDtype):
            raise ValueError(
                f"`table.obs[region_key]` must be of type `categorical`, not `{type(table.obs[region_key])}`."
            )
        if instance_key is not None and table.obs[instance_key].isnull().values.any():
            raise ValueError("`table.obs[instance_key]` must not contain null values, but it does.")


CurrentRasterFormat = RasterFormatV02
CurrentShapesFormat = ShapesFormatV02
CurrentPointsFormat = PointsFormatV01
CurrentTablesFormat = TablesFormatV01
CurrentSpatialDataContainerFormats = SpatialDataContainerFormatV01

ShapesFormats = {
    "0.1": ShapesFormatV01(),
    "0.2": ShapesFormatV02(),
}
PointsFormats = {
    "0.1": PointsFormatV01(),
}
TablesFormats = {
    "0.1": TablesFormatV01(),
}
RasterFormats = {
    "0.1": RasterFormatV01(),
    "0.2": RasterFormatV02(),
}
SpatialDataContainerFormats = {
    "0.1": SpatialDataContainerFormatV01(),
}


def format_implementations() -> Iterator[Format]:
    """Return an instance of each format implementation, newest to oldest."""
    yield RasterFormatV02()
    # yield RasterFormatV01()  # same format string as FormatV04
    yield FormatV04()
    yield FormatV03()
    yield FormatV02()
    yield FormatV01()


# monkeypatch the ome_zarr.format module to include the SpatialDataFormat (we want to use the APIs from ome_zarr to
# read, but signal that the format we are using is a dev version of NGFF, since it builds on some open PR that are
# not released yet)
ome_zarr.format.format_implementations = format_implementations


def _parse_formats(formats: SpatialDataFormat | list[SpatialDataFormat] | None) -> dict[str, SpatialDataFormat]:
    parsed = {
        "raster": CurrentRasterFormat(),
        "shapes": CurrentShapesFormat(),
        "points": CurrentPointsFormat(),
        "tables": CurrentTablesFormat(),
        "SpatialData": CurrentSpatialDataContainerFormats(),
    }
    if formats is None:
        return parsed
    if not isinstance(formats, list):
        formats = [formats]

    # this is to ensure that the variable `formats`, which is passed by the user, does not contain multiple versions
    # of the same format
    modified = {
        "raster": False,
        "shapes": False,
        "points": False,
        "tables": False,
        "SpatialData": False,
    }

    def _check_modified(element_type: str) -> None:
        if modified[element_type]:
            raise ValueError(f"Duplicate format {element_type} in input argument.")
        modified[element_type] = True

    for fmt in formats:
        if any(isinstance(fmt, type(v)) for v in ShapesFormats.values()):
            _check_modified("shapes")
            parsed["shapes"] = fmt
        elif any(isinstance(fmt, type(v)) for v in PointsFormats.values()):
            _check_modified("points")
            parsed["points"] = fmt
        elif any(isinstance(fmt, type(v)) for v in TablesFormats.values()):
            _check_modified("tables")
            parsed["tables"] = fmt
        elif any(isinstance(fmt, type(v)) for v in RasterFormats.values()):
            _check_modified("raster")
            parsed["raster"] = fmt
        elif any(isinstance(fmt, type(v)) for v in SpatialDataContainerFormats.values()):
            _check_modified("SpatialData")
            parsed["SpatialData"] = fmt
        else:
            raise ValueError(f"Unsupported format {fmt}")
    return parsed
