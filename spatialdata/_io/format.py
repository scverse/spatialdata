from typing import Any, Dict, List, Optional, Tuple, Union

from anndata import AnnData
from ome_zarr.format import CurrentFormat
from pandas.api.types import is_categorical_dtype
from shapely import GeometryType

from spatialdata._core.models import PointsModel, PolygonsModel, ShapesModel

CoordinateTransform_t = List[Dict[str, Any]]

Polygon_s = PolygonsModel()
Shapes_s = ShapesModel()
Points_s = PointsModel()


class SpatialDataFormatV01(CurrentFormat):
    """
    SpatialDataFormat defines the format of the spatialdata
    package.
    """

    @property
    def spatialdata_version(self) -> str:
        return "0.1"

    def validate_table(
        self,
        table: AnnData,
        region_key: Optional[str] = None,
        instance_key: Optional[str] = None,
    ) -> None:
        if not isinstance(table, AnnData):
            raise TypeError(f"`tables` must be `anndata.AnnData`, was {type(table)}.")
        if region_key is not None:
            if not is_categorical_dtype(table.obs[region_key]):
                raise ValueError(
                    f"`tables.obs[region_key]` must be of type `categorical`, not `{type(table.obs[region_key])}`."
                )
        if instance_key is not None:
            if table.obs[instance_key].isnull().values.any():
                raise ValueError("`tables.obs[instance_key]` must not contain null values, but it does.")

    def generate_coordinate_transformations(self, shapes: List[Tuple[Any]]) -> Optional[List[List[Dict[str, Any]]]]:

        data_shape = shapes[0]
        coordinate_transformations: List[List[Dict[str, Any]]] = []
        # calculate minimal 'scale' transform based on pyramid dims
        for shape in shapes:
            assert len(shape) == len(data_shape)
            scale = [full / level for full, level in zip(data_shape, shape)]
            from spatialdata._core.transformations import Scale

            coordinate_transformations.append([Scale(scale=scale).to_dict()])
        return coordinate_transformations

    def validate_coordinate_transformations(
        self,
        ndim: int,
        nlevels: int,
        coordinate_transformations: Optional[List[List[Dict[str, Any]]]] = None,
    ) -> None:
        """
        Validates that a list of dicts contains a 'scale' transformation
        Raises ValueError if no 'scale' found or doesn't match ndim
        :param ndim:       Number of image dimensions
        """

        if coordinate_transformations is None:
            raise ValueError("coordinate_transformations must be provided")
        ct_count = len(coordinate_transformations)
        if ct_count != nlevels:
            raise ValueError(f"coordinate_transformations count: {ct_count} must match datasets {nlevels}")
        for transformations in coordinate_transformations:
            assert isinstance(transformations, list)
            types = [t.get("type", None) for t in transformations]
            if any([t is None for t in types]):
                raise ValueError("Missing type in: %s" % transformations)

            # new validation
            import json

            json0 = [json.dumps(t) for t in transformations]
            from spatialdata._core.transformations import BaseTransformation

            parsed = [BaseTransformation.from_dict(t) for t in transformations]
            json1 = [json.dumps(p.to_dict()) for p in parsed]
            import numpy as np

            assert np.all([j0 == j1 for j0, j1 in zip(json0, json1)])


class PolygonsFormat(SpatialDataFormatV01):
    """Formatter for polygons."""

    def attrs_from_dict(self, metadata: Dict[str, Any]) -> GeometryType:
        if Polygon_s.ATTRS_KEY not in metadata:
            raise KeyError(f"Missing key {Polygon_s.ATTRS_KEY} in polygons metadata.")
        metadata_ = metadata[Polygon_s.ATTRS_KEY]
        if Polygon_s.GEOS_KEY not in metadata_:
            raise KeyError(f"Missing key {Polygon_s.GEOS_KEY} in polygons metadata.")
        for k in [Polygon_s.TYPE_KEY, Polygon_s.NAME_KEY]:
            if k not in metadata_[Polygon_s.GEOS_KEY]:
                raise KeyError(f"Missing key {k} in polygons metadata.")

        typ = GeometryType(metadata_[Polygon_s.GEOS_KEY][Polygon_s.TYPE_KEY])
        assert typ.name == metadata_[Polygon_s.GEOS_KEY][Polygon_s.NAME_KEY]
        assert self.spatialdata_version == metadata_["version"]
        return typ

    def attrs_to_dict(self, geometry: GeometryType) -> Dict[str, Union[str, Dict[str, Any]]]:
        return {Polygon_s.GEOS_KEY: {Polygon_s.NAME_KEY: geometry.name, Polygon_s.TYPE_KEY: geometry.value}}


class ShapesFormat(SpatialDataFormatV01):
    """Formatter for shapes."""

    def attrs_from_dict(self, metadata: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        if Shapes_s.ATTRS_KEY not in metadata:
            raise KeyError(f"Missing key {Shapes_s.ATTRS_KEY} in shapes metadata.")
        metadata_ = metadata[Shapes_s.ATTRS_KEY]
        if Shapes_s.TYPE_KEY not in metadata_:
            raise KeyError(f"Missing key {Shapes_s.TYPE_KEY} in shapes metadata.")
        return {Shapes_s.TYPE_KEY: metadata_[Shapes_s.TYPE_KEY]}

    def attrs_to_dict(self, data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        return {Shapes_s.TYPE_KEY: data[Shapes_s.ATTRS_KEY][Shapes_s.TYPE_KEY]}


class PointsFormat(SpatialDataFormatV01):
    """Formatter for points."""

    def attrs_from_dict(self, metadata: Dict[str, Any]) -> None:
        raise NotImplementedError

    def attrs_to_dict(self, data: Dict[str, Any]) -> None:
        raise NotImplementedError
