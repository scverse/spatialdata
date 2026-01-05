import json
import tempfile
from pathlib import Path
from typing import Any

import pytest
from shapely import GeometryType

from spatialdata import read_zarr
from spatialdata._io.format import (
    PointsFormatType,
    PointsFormatV01,
    PointsFormatV02,
    RasterFormatV01,
    RasterFormatV02,
    RasterFormatV03,
    ShapesFormatType,
    ShapesFormatV01,
    ShapesFormatV02,
    ShapesFormatV03,
    SpatialDataContainerFormatV01,
    SpatialDataContainerFormatV02,
    SpatialDataFormatType,
    TablesFormatV01,
    TablesFormatV02,
)
from spatialdata.models import PointsModel, ShapesModel
from spatialdata.testing import assert_spatial_data_objects_are_identical


class TestFormat:
    """Test format."""

    @pytest.mark.parametrize("element_format", [PointsFormatV01(), PointsFormatV02()])
    @pytest.mark.parametrize("attrs_key", [PointsModel.ATTRS_KEY])
    @pytest.mark.parametrize("feature_key", [None, PointsModel.FEATURE_KEY])
    @pytest.mark.parametrize("instance_key", [None, PointsModel.INSTANCE_KEY])
    def test_format_points_v1_v2(
        self,
        element_format: PointsFormatType,
        attrs_key: str | None,
        feature_key: str | None,
        instance_key: str | None,
    ) -> None:
        metadata: dict[str, Any] = {attrs_key: {"version": element_format.spatialdata_format_version}}
        format_metadata: dict[str, Any] = {attrs_key: {}}
        if feature_key is not None:
            metadata[attrs_key][feature_key] = "target"
        if instance_key is not None:
            metadata[attrs_key][instance_key] = "cell_id"
        format_metadata[attrs_key] = element_format.attrs_from_dict(metadata)
        metadata[attrs_key].pop("version")
        assert metadata[attrs_key] == element_format.attrs_to_dict(format_metadata)
        if feature_key is None and instance_key is None:
            assert len(format_metadata[attrs_key]) == len(metadata[attrs_key]) == 0

    @pytest.mark.parametrize("attrs_key", [ShapesModel.ATTRS_KEY])
    @pytest.mark.parametrize("geos_key", [ShapesModel.GEOS_KEY])
    @pytest.mark.parametrize("type_key", [ShapesModel.TYPE_KEY])
    @pytest.mark.parametrize("name_key", [ShapesModel.NAME_KEY])
    @pytest.mark.parametrize("shapes_type", [0, 3, 6])
    def test_format_shape_v1(
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

    @pytest.mark.parametrize("element_format", [ShapesFormatV02(), ShapesFormatV03()])
    @pytest.mark.parametrize("attrs_key", [ShapesModel.ATTRS_KEY])
    def test_format_shapes_v2_v3(
        self,
        element_format: ShapesFormatType,
        attrs_key: str,
    ) -> None:
        metadata: dict[str, Any] = {attrs_key: {"version": element_format.spatialdata_format_version}}
        metadata[attrs_key].pop("version")
        assert metadata[attrs_key] == element_format.attrs_to_dict({})

    @pytest.mark.parametrize("rformat", [RasterFormatV01, RasterFormatV02, RasterFormatV03])
    def test_format_raster_v1_v2_v3(self, images, rformat: type[SpatialDataFormatType]) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            sdata_container_format = (
                SpatialDataContainerFormatV01() if rformat != RasterFormatV03 else SpatialDataContainerFormatV02()
            )
            images.write(Path(tmpdir) / "images.zarr", sdata_formats=[sdata_container_format, rformat()])

            metadata_file = ".zattrs" if rformat != RasterFormatV03 else "zarr.json"
            zattrs_file = Path(tmpdir) / "images.zarr/images/image2d/" / metadata_file

            with open(zattrs_file) as infile:
                zattrs = json.load(infile)
                if rformat == RasterFormatV01:
                    ngff_version = zattrs["multiscales"][0]["version"]
                    assert ngff_version == "0.4"
                elif rformat == RasterFormatV02:
                    ngff_version = zattrs["multiscales"][0]["version"]
                    assert ngff_version == "0.4-dev-spatialdata"
                else:
                    ngff_version = zattrs["attributes"]["ome"]["version"]
                    assert rformat == RasterFormatV03
                    assert ngff_version == "0.5-dev-spatialdata"

    # TODO: add tests for TablesFormatV01 and TablesFormatV02


class TestFormatConversions:
    """Test format conversions between older formats and newer."""

    def test_shapes_v1_to_v2_to_v3(self, shapes):
        with tempfile.TemporaryDirectory() as tmpdir:
            f1 = Path(tmpdir) / "data1.zarr"
            f2 = Path(tmpdir) / "data2.zarr"
            f3 = Path(tmpdir) / "data3.zarr"

            shapes.write(f1, sdata_formats=[ShapesFormatV01(), SpatialDataContainerFormatV01()])
            shapes_read_v1 = read_zarr(f1)
            assert_spatial_data_objects_are_identical(shapes, shapes_read_v1)
            assert shapes_read_v1.is_self_contained()

            shapes_read_v1.write(f2, sdata_formats=[ShapesFormatV02(), SpatialDataContainerFormatV01()])
            shapes_read_v2 = read_zarr(f2)
            assert_spatial_data_objects_are_identical(shapes, shapes_read_v2)
            assert shapes_read_v2.is_self_contained()

            shapes_read_v2.write(f3, sdata_formats=[ShapesFormatV03(), SpatialDataContainerFormatV02()])
            shapes_read_v3 = read_zarr(f3)
            assert_spatial_data_objects_are_identical(shapes, shapes_read_v3)
            assert shapes_read_v3.is_self_contained()

    def test_raster_images_v1_to_v2_to_v3(self, images):
        with tempfile.TemporaryDirectory() as tmpdir:
            f1 = Path(tmpdir) / "data1.zarr"
            f2 = Path(tmpdir) / "data2.zarr"
            f3 = Path(tmpdir) / "data3.zarr"

            with pytest.raises(ValueError, match="Unsupported format"):
                images.write(f1, sdata_formats=RasterFormatV01())

            images.write(f1, sdata_formats=[RasterFormatV01(), SpatialDataContainerFormatV01()])
            images_read_v1 = read_zarr(f1)
            assert_spatial_data_objects_are_identical(images, images_read_v1)
            assert images_read_v1.is_self_contained()

            images_read_v1.write(f2, sdata_formats=[RasterFormatV02(), SpatialDataContainerFormatV01()])
            images_read_v2 = read_zarr(f2)
            assert_spatial_data_objects_are_identical(images, images_read_v2)
            assert images_read_v2.is_self_contained()

            images_read_v2.write(f3, sdata_formats=[RasterFormatV03(), SpatialDataContainerFormatV02()])
            images_read_v3 = read_zarr(f3)
            assert_spatial_data_objects_are_identical(images, images_read_v3)
            assert images_read_v3.is_self_contained()

    def test_raster_labels_v1_to_v2_to_v3(self, labels):
        with tempfile.TemporaryDirectory() as tmpdir:
            f1 = Path(tmpdir) / "data1.zarr"
            f2 = Path(tmpdir) / "data2.zarr"
            f3 = Path(tmpdir) / "data3.zarr"

            labels.write(f1, sdata_formats=[RasterFormatV01(), SpatialDataContainerFormatV01()])
            labels_read_v1 = read_zarr(f1)
            assert_spatial_data_objects_are_identical(labels, labels_read_v1)
            assert labels_read_v1.is_self_contained()

            labels_read_v1.write(f2, sdata_formats=[RasterFormatV02(), SpatialDataContainerFormatV01()])
            labels_read_v2 = read_zarr(f2)
            assert_spatial_data_objects_are_identical(labels, labels_read_v2)
            assert labels_read_v2.is_self_contained()

            labels_read_v2.write(f3, sdata_formats=[RasterFormatV03(), SpatialDataContainerFormatV02()])
            labels_read_v3 = read_zarr(f3)
            assert_spatial_data_objects_are_identical(labels, labels_read_v3)
            assert labels_read_v3.is_self_contained()

    def test_points_v1_to_v2(self, points):
        with tempfile.TemporaryDirectory() as tmpdir:
            f1 = Path(tmpdir) / "data1.zarr"
            f2 = Path(tmpdir) / "data2.zarr"

            points.write(f1, sdata_formats=[PointsFormatV01(), SpatialDataContainerFormatV01()])
            points_read_v1 = read_zarr(f1)
            assert_spatial_data_objects_are_identical(points, points_read_v1)
            assert points_read_v1.is_self_contained()

            points_read_v1.write(f2, sdata_formats=[PointsFormatV02(), SpatialDataContainerFormatV02()])
            points_read_v2 = read_zarr(f2)
            assert_spatial_data_objects_are_identical(points, points_read_v2)
            assert points_read_v2.is_self_contained()

    def test_tables_v1_to_v2(self, table_multiple_annotations):
        with tempfile.TemporaryDirectory() as tmpdir:
            f1 = Path(tmpdir) / "data1.zarr"
            f2 = Path(tmpdir) / "data2.zarr"

            table_multiple_annotations.write(f1, sdata_formats=[TablesFormatV01(), SpatialDataContainerFormatV01()])
            table_read_v1 = read_zarr(f1)
            assert_spatial_data_objects_are_identical(table_multiple_annotations, table_read_v1)

            table_read_v1.write(f2, sdata_formats=[TablesFormatV02(), SpatialDataContainerFormatV02()])
            table_read_v2 = read_zarr(f2)
            assert_spatial_data_objects_are_identical(table_multiple_annotations, table_read_v2)

    def test_container_v1_to_v2(self, full_sdata):
        with tempfile.TemporaryDirectory() as tmpdir:
            f1 = Path(tmpdir) / "data1.zarr"
            f2 = Path(tmpdir) / "data2.zarr"

            full_sdata.write(f1, sdata_formats=[SpatialDataContainerFormatV01()])
            sdata_read_v1 = read_zarr(f1)
            assert_spatial_data_objects_are_identical(full_sdata, sdata_read_v1)
            assert sdata_read_v1.is_self_contained()
            assert sdata_read_v1.has_consolidated_metadata()

            sdata_read_v1.write(f2, sdata_formats=[SpatialDataContainerFormatV02()])
            sdata_read_v2 = read_zarr(f2)
            assert_spatial_data_objects_are_identical(full_sdata, sdata_read_v2)
            assert sdata_read_v2.is_self_contained()
            assert sdata_read_v2.has_consolidated_metadata()

    def test_channel_names_raster_images_v1_to_v2_to_v3(self, images):
        with tempfile.TemporaryDirectory() as tmpdir:
            f1 = Path(tmpdir) / "data1.zarr"
            f2 = Path(tmpdir) / "data2.zarr"
            f3 = Path(tmpdir) / "data3.zarr"

            new_channels = ["first", "second", "third"]

            images.write(f1, sdata_formats=[RasterFormatV01(), SpatialDataContainerFormatV01()])
            images_read_v1 = read_zarr(f1)
            images_read_v1.set_channel_names("image2d", new_channels, write=True)
            images_read_v1.set_channel_names("image2d_multiscale", new_channels, write=True)
            assert images_read_v1["image2d"].coords["c"].data.tolist() == new_channels
            assert images_read_v1["image2d_multiscale"]["scale0"]["image"].coords["c"].data.tolist() == new_channels

            images_read_v1.write(f2, sdata_formats=[SpatialDataContainerFormatV01()])
            images_read_v1 = read_zarr(f2)
            assert images_read_v1["image2d"].coords["c"].data.tolist() == new_channels
            assert images_read_v1["image2d_multiscale"]["scale0"]["image"].coords["c"].data.tolist() == new_channels

            images_read_v1.write(f3, sdata_formats=[SpatialDataContainerFormatV02()])
            sdata_read_v2 = read_zarr(f3)
            assert sdata_read_v2["image2d"].coords["c"].data.tolist() == new_channels
            assert sdata_read_v2["image2d_multiscale"]["scale0"]["image"].coords["c"].data.tolist() == new_channels
