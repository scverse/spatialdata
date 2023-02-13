from pathlib import Path

import pandas as pd
import pytest
from dask.dataframe.core import DataFrame as DaskDataFrame
from dask.dataframe.utils import assert_eq
from geopandas import GeoDataFrame
from multiscale_spatial_image.multiscale_spatial_image import MultiscaleSpatialImage
from shapely.geometry import Point
from spatial_image import SpatialImage

from spatialdata import SpatialData
from spatialdata.utils import are_directories_identical
from tests.conftest import _get_images, _get_labels, _get_points, _get_shapes


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

    def test_shapes(self, tmp_path: str, shapes: SpatialData) -> None:
        """Test read/write."""
        tmpdir = Path(tmp_path) / "tmp.zarr"
        shapes.write(tmpdir)
        sdata = SpatialData.read(tmpdir)
        assert shapes.shapes.keys() == sdata.shapes.keys()
        for k in shapes.shapes.keys():
            assert isinstance(sdata.shapes[k], GeoDataFrame)
            assert shapes.shapes[k].equals(sdata.shapes[k])
            if "radius" in shapes.shapes["circles"].columns:
                assert shapes.shapes["circles"]["radius"].equals(sdata.shapes["circles"]["radius"])
                assert isinstance(sdata.shapes["circles"]["geometry"][0], Point)

    def test_points(self, tmp_path: str, points: SpatialData) -> None:
        """Test read/write."""
        tmpdir = Path(tmp_path) / "tmp.zarr"
        points.write(tmpdir)
        sdata = SpatialData.read(tmpdir)
        assert points.points.keys() == sdata.points.keys()
        for k in points.points.keys():
            assert isinstance(sdata.points[k], DaskDataFrame)
            assert assert_eq(points.points[k], sdata.points[k], check_divisions=False)
            assert points.points[k].attrs == points.points[k].attrs

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

    def test_incremental_io(
        self,
        tmp_path: str,
        full_sdata: SpatialData,
    ) -> None:
        tmpdir = Path(tmp_path) / "tmp.zarr"
        sdata = full_sdata

        sdata.add_image(name="sdata_not_saved_yet", image=_get_images().values().__iter__().__next__())
        sdata.write(tmpdir)

        for k, v in _get_images().items():
            if isinstance(v, SpatialImage):
                v.name = f"incremental_{k}"
            elif isinstance(v, MultiscaleSpatialImage):
                for scale in v:
                    names = list(v[scale].keys())
                    assert len(names) == 1
                    name = names[0]
                    v[scale] = v[scale].rename_vars({name: f"incremental_{k}"})
            sdata.add_image(name=f"incremental_{k}", image=v)
            with pytest.raises(ValueError):
                sdata.add_image(name=f"incremental_{k}", image=v)
            sdata.add_image(name=f"incremental_{k}", image=v, overwrite=True)

        for k, v in _get_labels().items():
            if isinstance(v, SpatialImage):
                v.name = f"incremental_{k}"
            elif isinstance(v, MultiscaleSpatialImage):
                for scale in v:
                    names = list(v[scale].keys())
                    assert len(names) == 1
                    name = names[0]
                    v[scale] = v[scale].rename_vars({name: f"incremental_{k}"})
            sdata.add_labels(name=f"incremental_{k}", labels=v)
            with pytest.raises(ValueError):
                sdata.add_labels(name=f"incremental_{k}", labels=v)
            sdata.add_labels(name=f"incremental_{k}", labels=v, overwrite=True)

        for k, v in _get_shapes().items():
            sdata.add_shapes(name=f"incremental_{k}", shapes=v)
            with pytest.raises(ValueError):
                sdata.add_shapes(name=f"incremental_{k}", shapes=v)
            sdata.add_shapes(name=f"incremental_{k}", shapes=v, overwrite=True)
            break

        for k, v in _get_points().items():
            sdata.add_points(name=f"incremental_{k}", points=v)
            with pytest.raises(ValueError):
                sdata.add_points(name=f"incremental_{k}", points=v)
            sdata.add_points(name=f"incremental_{k}", points=v, overwrite=True)
            break
