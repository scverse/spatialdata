import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from dask.dataframe.core import DataFrame as DaskDataFrame
from dask.dataframe.utils import assert_eq
from geopandas import GeoDataFrame
from multiscale_spatial_image.multiscale_spatial_image import MultiscaleSpatialImage
from spatial_image import SpatialImage

from spatialdata import SpatialData
from spatialdata._core._spatialdata_ops import get_transformation, set_transformation
from spatialdata._core.transformations import Identity, Scale
from spatialdata.utils import are_directories_identical
from tests.conftest import _get_images, _get_labels, _get_polygons, _get_shapes


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

    # @pytest.mark.skip("waiting for the new points implementation")
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

        # TODO: remove this line when points are ready
        for k in sdata.points:
            del sdata.points[k]

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

        for k, v in _get_polygons().items():
            sdata.add_polygons(name=f"incremental_{k}", polygons=v)
            with pytest.raises(ValueError):
                sdata.add_polygons(name=f"incremental_{k}", polygons=v)
            sdata.add_polygons(name=f"incremental_{k}", polygons=v, overwrite=True)
            # only one element to save time to do the test. We have this for the other types too, but not for images
            # and labels beacuse we want to test both the Multiscale and the non-Multiscale case
            break

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


def test_io_and_lazy_loading_points(points):
    assert False


def test_io_and_lazy_loading_raster(images, labels):
    # addresses bug https://github.com/scverse/spatialdata/issues/117
    sdatas = {"images": images, "labels": labels}
    for k, sdata in sdatas.items():
        d = sdata.__getattribute__(k)
        elem_name = list(d.keys())[0]
        with tempfile.TemporaryDirectory() as td:
            f = os.path.join(td, "data.zarr")
            dask0 = d[elem_name].data
            sdata.write(f)
            dask1 = d[elem_name].data
            assert all("from-zarr" not in key for key in dask0.dask.layers)
            assert any("from-zarr" in key for key in dask1.dask.layers)


def test_replace_transformation_on_disk_raster(images, labels):
    sdatas = {"images": images, "labels": labels}
    for k, sdata in sdatas.items():
        d = sdata.__getattribute__(k)
        # unlike the non-raster case we are testing all the elements (2d and 3d, multiscale and not)
        # TODO: we can actually later on merge this test and the one below keepin the logic of this function here
        for elem_name in d.keys():
            kwargs = {k: {elem_name: d[elem_name]}}
            single_sdata = SpatialData(**kwargs)
            with tempfile.TemporaryDirectory() as td:
                f = os.path.join(td, "data.zarr")
                single_sdata.write(f)
                t0 = get_transformation(SpatialData.read(f).__getattribute__(k)[elem_name])
                assert type(t0) == Identity
                set_transformation(
                    single_sdata.__getattribute__(k)[elem_name], Scale([2.0], axes=("x",)), write_to_sdata=single_sdata
                )
                t1 = get_transformation(SpatialData.read(f).__getattribute__(k)[elem_name])
                assert type(t1) == Scale


def test_replace_transformation_on_disk_non_raster(polygons, shapes, points):
    sdatas = {"polygons": polygons, "shapes": shapes, "points": points}
    for k, sdata in sdatas.items():
        d = sdata.__getattribute__(k)
        elem_name = list(d.keys())[0]
        with tempfile.TemporaryDirectory() as td:
            f = os.path.join(td, "data.zarr")
            sdata.write(f)
            t0 = get_transformation(SpatialData.read(f).__getattribute__(k)[elem_name])
            assert type(t0) == Identity
            set_transformation(sdata.__getattribute__(k)[elem_name], Scale([2.0], axes=("x",)), write_to_sdata=sdata)
            t1 = get_transformation(SpatialData.read(f).__getattribute__(k)[elem_name])
            assert type(t1) == Scale
