from typing import Any, Optional, Union

from anndata import AnnData
from multiscale_spatial_image.multiscale_spatial_image import MultiscaleSpatialImage
from ome_zarr.format import CurrentFormat
from pandas.api.types import is_categorical_dtype
from shapely import GeometryType
from spatial_image import SpatialImage

from spatialdata._io._utils import get_channels
from spatialdata.models import PointsModel, ShapesModel

CoordinateTransform_t = list[dict[str, Any]]

Shapes_s = ShapesModel()
Points_s = PointsModel()


class SpatialDataFormatV01(CurrentFormat):
    """SpatialDataFormat defines the format of the spatialdata package."""


class RasterFormatV01(SpatialDataFormatV01):
    """Formatter for raster data."""

    def generate_coordinate_transformations(self, shapes: list[tuple[Any]]) -> Optional[list[list[dict[str, Any]]]]:
        data_shape = shapes[0]
        coordinate_transformations: list[list[dict[str, Any]]] = []
        # calculate minimal 'scale' transform based on pyramid dims
        for shape in shapes:
            assert len(shape) == len(data_shape)
            scale = [full / level for full, level in zip(data_shape, shape)]
            from spatialdata.transformations.ngff.ngff_transformations import NgffScale

            coordinate_transformations.append([NgffScale(scale=scale).to_dict()])
        return coordinate_transformations

    def validate_coordinate_transformations(
        self,
        ndim: int,
        nlevels: int,
        coordinate_transformations: Optional[list[list[dict[str, Any]]]] = None,
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
                raise ValueError("Missing type in: %s" % transformations)

            # new validation
            import json

            json0 = [json.dumps(t) for t in transformations]
            from spatialdata.transformations.ngff.ngff_transformations import (
                NgffBaseTransformation,
            )

            parsed = [NgffBaseTransformation.from_dict(t) for t in transformations]
            json1 = [json.dumps(p.to_dict()) for p in parsed]
            import numpy as np

            assert np.all([j0 == j1 for j0, j1 in zip(json0, json1)])

    def channels_to_metadata(
        self,
        data: Union[SpatialImage, MultiscaleSpatialImage],
        channels_metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Union[int, str]]:
        """Convert channels to omero metadata."""
        channels = get_channels(data)
        metadata: dict[str, Any] = {"channels": []}
        if channels_metadata is not None:
            if set(channels_metadata.keys()).symmetric_difference(set(channels)):
                for c in channels:
                    metadata["channels"].append({"label": c} | channels_metadata[c])
            else:
                raise ValueError("Channels metadata must contain all channels.")
        else:
            for c in channels:
                metadata["channels"].append({"label": c})
        return metadata

    def channels_from_metadata(self, omero_metadata: dict[str, Any]) -> list[Any]:
        """Convert channels from omero metadata."""
        return [d["label"] for d in omero_metadata["channels"]]


class ShapesFormatV01(SpatialDataFormatV01):
    """Formatter for shapes."""

    @property
    def version(self) -> str:
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
        assert self.version == metadata_["version"]
        return typ

    def attrs_to_dict(self, geometry: GeometryType) -> dict[str, Union[str, dict[str, Any]]]:
        return {Shapes_s.GEOS_KEY: {Shapes_s.NAME_KEY: geometry.name, Shapes_s.TYPE_KEY: geometry.value}}


class PointsFormatV01(SpatialDataFormatV01):
    """Formatter for points."""

    @property
    def version(self) -> str:
        return "0.1"

    def attrs_from_dict(self, metadata: dict[str, Any]) -> dict[str, dict[str, Any]]:
        if Points_s.ATTRS_KEY not in metadata:
            raise KeyError(f"Missing key {Points_s.ATTRS_KEY} in points metadata.")
        metadata_ = metadata[Points_s.ATTRS_KEY]
        assert self.version == metadata_["version"]
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


class TablesFormatV01(SpatialDataFormatV01):
    """Formatter for the table."""

    @property
    def version(self) -> str:
        return "0.1"

    def validate_table(
        self,
        table: AnnData,
        region_key: Optional[str] = None,
        instance_key: Optional[str] = None,
    ) -> None:
        if not isinstance(table, AnnData):
            raise TypeError(f"`table` must be `anndata.AnnData`, was {type(table)}.")
        if region_key is not None and not is_categorical_dtype(table.obs[region_key]):
            raise ValueError(
                f"`table.obs[region_key]` must be of type `categorical`, not `{type(table.obs[region_key])}`."
            )
        if instance_key is not None and table.obs[instance_key].isnull().values.any():
            raise ValueError("`table.obs[instance_key]` must not contain null values, but it does.")


CurrentRasterFormat = RasterFormatV01
CurrentShapesFormat = ShapesFormatV01
CurrentPointsFormat = PointsFormatV01
CurrentTablesFormat = TablesFormatV01
