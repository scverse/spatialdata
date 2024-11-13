from typing import Any

import pytest
from shapely import GeometryType

from spatialdata._io.format import CurrentPointsFormat, CurrentShapesFormat, ShapesFormatV01
from spatialdata.models import PointsModel, ShapesModel

Points_f = CurrentPointsFormat()
Shapes_f = CurrentShapesFormat()


class TestFormat:
    """Test format."""

    @pytest.mark.parametrize("attrs_key", [PointsModel.ATTRS_KEY])
    @pytest.mark.parametrize("feature_key", [None, PointsModel.FEATURE_KEY])
    @pytest.mark.parametrize("instance_key", [None, PointsModel.INSTANCE_KEY])
    def test_format_points(
        self,
        attrs_key: str | None,
        feature_key: str | None,
        instance_key: str | None,
    ) -> None:
        metadata: dict[str, Any] = {attrs_key: {"version": Points_f.spatialdata_format_version}}
        format_metadata: dict[str, Any] = {attrs_key: {}}
        if feature_key is not None:
            metadata[attrs_key][feature_key] = "target"
        if instance_key is not None:
            metadata[attrs_key][instance_key] = "cell_id"
        format_metadata[attrs_key] = Points_f.attrs_from_dict(metadata)
        metadata[attrs_key].pop("version")
        assert metadata[attrs_key] == Points_f.attrs_to_dict(format_metadata)
        if feature_key is None and instance_key is None:
            assert len(format_metadata[attrs_key]) == len(metadata[attrs_key]) == 0

    @pytest.mark.parametrize("attrs_key", [ShapesModel.ATTRS_KEY])
    @pytest.mark.parametrize("geos_key", [ShapesModel.GEOS_KEY])
    @pytest.mark.parametrize("type_key", [ShapesModel.TYPE_KEY])
    @pytest.mark.parametrize("name_key", [ShapesModel.NAME_KEY])
    @pytest.mark.parametrize("shapes_type", [0, 3, 6])
    def test_format_shapes_v1(
        self,
        attrs_key: str,
        geos_key: str,
        type_key: str,
        name_key: str,
        shapes_type: int,
    ) -> None:
        shapes_dict = {
            0: "POINT",
            3: "POLYGON",
            6: "MULTIPOLYGON",
        }
        metadata: dict[str, Any] = {attrs_key: {"version": ShapesFormatV01().spatialdata_format_version}}
        format_metadata: dict[str, Any] = {attrs_key: {}}
        metadata[attrs_key][geos_key] = {}
        metadata[attrs_key][geos_key][type_key] = shapes_type
        metadata[attrs_key][geos_key][name_key] = shapes_dict[shapes_type]
        format_metadata[attrs_key] = ShapesFormatV01().attrs_from_dict(metadata)
        metadata[attrs_key].pop("version")
        geometry = GeometryType(metadata[attrs_key][geos_key][type_key])
        assert metadata[attrs_key] == ShapesFormatV01().attrs_to_dict(geometry)

    @pytest.mark.parametrize("attrs_key", [ShapesModel.ATTRS_KEY])
    def test_format_shapes_v2(
        self,
        attrs_key: str,
    ) -> None:
        # not testing anything, maybe remove
        metadata: dict[str, Any] = {attrs_key: {"version": Shapes_f.spatialdata_format_version}}
        metadata[attrs_key].pop("version")
        assert metadata[attrs_key] == Shapes_f.attrs_to_dict({})
