from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from geopandas import GeoDataFrame
from multiscale_spatial_image.multiscale_spatial_image import MultiscaleSpatialImage
from spatial_image import SpatialImage

from spatialdata import SpatialData
from spatialdata._core.elements import Image
from spatialdata.utils import are_directories_identical


class TestReadWrite:
    def test_images(self, tmp_path: str, images: SpatialData) -> None:
        """Test read/write."""
        tmpdir = Path(tmp_path) / "tmp.zarr"
        images.write(tmpdir)
        sdata = SpatialData.read(tmpdir)
        assert images.images.keys() == sdata.images.keys()
        for k1, k2 in zip(images.images.keys(), sdata.images.keys()):
            if isinstance(sdata.images[k1], SpatialImage):
                assert images.images[k1].equals(sdata.images[k2])
            elif isinstance(images.images[k1], MultiscaleSpatialImage):
                assert images.images[k1].equals(sdata.images[k2])

    def test_labels(self, tmp_path: str, labels: SpatialData) -> None:
        """Test read/write."""
        tmpdir = Path(tmp_path) / "tmp.zarr"
        labels.write(tmpdir)
        sdata = SpatialData.read(tmpdir)
        assert labels.labels.keys() == sdata.labels.keys()
        for k1, k2 in zip(labels.labels.keys(), sdata.labels.keys()):
            if isinstance(sdata.labels[k1], SpatialImage):
                assert labels.labels[k1].equals(sdata.labels[k2])
            elif isinstance(sdata.labels[k1], MultiscaleSpatialImage):
                assert labels.labels[k1].equals(sdata.labels[k2])

    def test_polygons(self, tmp_path: str, polygons: SpatialData) -> None:
        """Test read/write."""
        tmpdir = Path(tmp_path) / "tmp.zarr"
        polygons.write(tmpdir)
        sdata = SpatialData.read(tmpdir)
        assert polygons.polygons.keys() == sdata.polygons.keys()
        for k1, k2 in zip(polygons.polygons.keys(), sdata.polygons.keys()):
            assert isinstance(sdata.polygons[k1], GeoDataFrame)
            assert polygons.polygons[k1].equals(sdata.polygons[k2])

    def test_shapes(self, tmp_path: str, shapes: SpatialData) -> None:
        """Test read/write."""
        tmpdir = Path(tmp_path) / "tmp.zarr"
        shapes.write(tmpdir)
        sdata = SpatialData.read(tmpdir)
        assert shapes.shapes.keys() == sdata.shapes.keys()
        for k1, k2 in zip(shapes.shapes.keys(), sdata.shapes.keys()):
            assert isinstance(sdata.shapes[k1], AnnData)
            np.testing.assert_array_equal(shapes.shapes[k1].obsm["spatial"], sdata.shapes[k2].obsm["spatial"])
            assert shapes.shapes[k1].uns == sdata.shapes[k2].uns

    def test_points(self, tmp_path: str, points: SpatialData) -> None:
        """Test read/write."""
        tmpdir = Path(tmp_path) / "tmp.zarr"
        points.write(tmpdir)
        sdata = SpatialData.read(tmpdir)
        assert points.points.keys() == sdata.points.keys()
        for k1, k2 in zip(points.points.keys(), sdata.points.keys()):
            assert isinstance(sdata.points[k1], AnnData)
            np.testing.assert_array_equal(points.points[k1].obsm["spatial"], sdata.points[k2].obsm["spatial"])
            assert points.points[k1].uns == sdata.points[k2].uns

    def test_table(self, tmp_path: str, table: SpatialData) -> None:
        """Test read/write."""
        tmpdir = Path(tmp_path) / "tmp.zarr"
        table.write(tmpdir)
        sdata = SpatialData.read(tmpdir)
        pd.testing.assert_frame_equal(table.table.obs, sdata.table.obs)
        assert table.table.uns == sdata.table.uns

    # TODO: refactor, add spatialdata build to conftest
    def test_roundtrip(
        self,
        tmp_path: str,
        images: SpatialData,
        labels: SpatialData,
        polygons: SpatialData,
        shapes: SpatialData,
        points: SpatialData,
        table: SpatialData,
    ) -> None:
        tmpdir = Path(tmp_path) / "tmp.zarr"

        # TODO: not checking for consistency ATM
        sdata = SpatialData(
            images=images.images,
            labels=labels.labels,
            polygons=polygons.polygons,
            shapes=shapes.shapes,
            points=points.points,
            table=table.table,
        )
        sdata.write(tmpdir)
        sdata2 = SpatialData.read(tmpdir)
        tmpdir2 = Path(tmp_path) / "tmp2.zarr"
        sdata2.write(tmpdir2)
        are_directories_identical(tmpdir, tmpdir2, exclude_regexp="[1-9][0-9]*.*")


@pytest.mark.skip("Consider delete.")
def test_readwrite_roundtrip(sdata: SpatialData, tmp_path: str):
    print(sdata)

    tmpdir = Path(tmp_path) / "tmp.zarr"
    sdata.write(tmpdir)
    sdata2 = SpatialData.read(tmpdir)

    assert are_directories_identical(tmpdir, tmpdir)
    if sdata.table is not None or sdata2.table is not None:
        assert sdata.table is None and sdata2.table is None or sdata.table.shape == sdata2.table.shape
    assert sdata.images.keys() == sdata2.images.keys()
    for k in sdata.images.keys():
        assert sdata.images[k].shape == sdata2.images[k].shape
        assert isinstance(sdata.images[k], Image) == isinstance(sdata2.images[k], Image)
    assert list(sdata.labels.keys()) == list(sdata2.labels.keys())

    tmpdir2 = Path(tmp_path) / "tmp2.zarr"
    sdata2.write(tmpdir2)
    # install ome-zarr-py from https://github.com/LucaMarconato/ome-zarr-py since this merges some branches with
    # bugfixes (see https://github.com/ome/ome-zarr-py/issues/219#issuecomment-1237263744)
    # also, we exclude the comparison of images that are not full scale in the pyramid representation, as they are
    # different due to a bug ( see discussion in the link above)
    assert are_directories_identical(tmpdir, tmpdir2, exclude_regexp="[1-9][0-9]*.*")
