from pathlib import Path

import numpy as np
import pandas as pd
from anndata import AnnData
from geopandas import GeoDataFrame
from multiscale_spatial_image.multiscale_spatial_image import MultiscaleSpatialImage
from spatial_image import SpatialImage

from spatialdata import SpatialData
from spatialdata.utils import are_directories_identical


class TestReadWrite:
    def test_images(self, tmp_path: str, images: SpatialData) -> None:
        """Test read/write."""
        tmpdir = Path(tmp_path) / "tmp.zarr"
        images.write(tmpdir)
        sdata = SpatialData.read(tmpdir)
        assert images.images.keys() == sdata.images.keys()
        for k in images.images.keys():
            if isinstance(sdata.images[k], SpatialImage):
                assert images.images[k].equals(sdata.images[k])
            elif isinstance(images.images[k], MultiscaleSpatialImage):
                assert images.images[k].equals(sdata.images[k])

    def test_labels(self, tmp_path: str, labels: SpatialData) -> None:
        """Test read/write."""
        tmpdir = Path(tmp_path) / "tmp.zarr"
        labels.write(tmpdir)
        sdata = SpatialData.read(tmpdir)
        assert labels.labels.keys() == sdata.labels.keys()
        for k in labels.labels.keys():
            if isinstance(sdata.labels[k], SpatialImage):
                assert labels.labels[k].equals(sdata.labels[k])
            elif isinstance(sdata.labels[k], MultiscaleSpatialImage):
                assert labels.labels[k].equals(sdata.labels[k])

    def test_polygons(self, tmp_path: str, polygons: SpatialData) -> None:
        """Test read/write."""
        tmpdir = Path(tmp_path) / "tmp.zarr"
        polygons.write(tmpdir)
        sdata = SpatialData.read(tmpdir)
        assert polygons.polygons.keys() == sdata.polygons.keys()
        for k in polygons.polygons.keys():
            assert isinstance(sdata.polygons[k], GeoDataFrame)
            assert polygons.polygons[k].equals(sdata.polygons[k])

    def test_shapes(self, tmp_path: str, shapes: SpatialData) -> None:
        """Test read/write."""
        tmpdir = Path(tmp_path) / "tmp.zarr"
        shapes.write(tmpdir)
        sdata = SpatialData.read(tmpdir)
        assert shapes.shapes.keys() == sdata.shapes.keys()
        for k in shapes.shapes.keys():
            assert isinstance(sdata.shapes[k], AnnData)
            np.testing.assert_array_equal(shapes.shapes[k].obsm["spatial"], sdata.shapes[k].obsm["spatial"])
            assert shapes.shapes[k].uns == sdata.shapes[k].uns

    def test_points(self, tmp_path: str, points: SpatialData) -> None:
        """Test read/write."""
        tmpdir = Path(tmp_path) / "tmp.zarr"
        points.write(tmpdir)
        sdata = SpatialData.read(tmpdir)
        assert points.points.keys() == sdata.points.keys()
        for k in points.points.keys():
            assert isinstance(sdata.points[k], AnnData)
            np.testing.assert_array_equal(points.points[k].obsm["spatial"], sdata.points[k].obsm["spatial"])
            assert points.points[k].uns == sdata.points[k].uns

    def _test_table(self, tmp_path: str, table: SpatialData) -> None:
        tmpdir = Path(tmp_path) / "tmp.zarr"
        table.write(tmpdir)
        sdata = SpatialData.read(tmpdir)
        pd.testing.assert_frame_equal(table.table.obs, sdata.table.obs)
        try:
            assert table.table.uns == sdata.table.uns
        except ValueError as e:
            raise e

    def test_table_single_annotation(self, tmp_path: str, table_single_annotation: SpatialData) -> None:
        """Test read/write."""
        self._test_table(tmp_path, table_single_annotation)

    def test_table_multiple_annotations(self, tmp_path: str, table_multiple_annotations: SpatialData) -> None:
        """Test read/write."""
        self._test_table(tmp_path, table_multiple_annotations)

    def test_roundtrip(
        self,
        tmp_path: str,
        sdata: SpatialData,
    ) -> None:
        tmpdir = Path(tmp_path) / "tmp.zarr"

        # TODO: not checking for consistency ATM
        sdata.write(tmpdir)
        sdata2 = SpatialData.read(tmpdir)
        tmpdir2 = Path(tmp_path) / "tmp2.zarr"
        sdata2.write(tmpdir2)
        are_directories_identical(tmpdir, tmpdir2, exclude_regexp="[1-9][0-9]*.*")
