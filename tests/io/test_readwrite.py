import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from anndata import AnnData
from dask.dataframe.core import DataFrame as DaskDataFrame
from dask.dataframe.utils import assert_eq
from geopandas import GeoDataFrame
from multiscale_spatial_image.multiscale_spatial_image import MultiscaleSpatialImage
from numpy.random import default_rng
from shapely.geometry import Point
from spatial_image import SpatialImage
from spatialdata import SpatialData, read_zarr
from spatialdata._io._utils import _are_directories_identical
from spatialdata.models import Image2DModel, TableModel
from spatialdata.transformations.operations import (
    get_transformation,
    set_transformation,
)
from spatialdata.transformations.transformations import Identity, Scale

from tests.conftest import _get_images, _get_labels, _get_points, _get_shapes

RNG = default_rng(0)


class TestReadWrite:
    def test_images(self, tmp_path: str, images: SpatialData) -> None:
        """Test read/write."""
        tmpdir = Path(tmp_path) / "tmp.zarr"
        images.write(tmpdir)
        sdata = SpatialData.read(tmpdir)
        assert images.images.keys() == sdata.images.keys()
        for k in images.images:
            assert images.images[k].equals(sdata.images[k])

    def test_labels(self, tmp_path: str, labels: SpatialData) -> None:
        """Test read/write."""
        tmpdir = Path(tmp_path) / "tmp.zarr"
        labels.write(tmpdir)
        sdata = SpatialData.read(tmpdir)
        assert labels.labels.keys() == sdata.labels.keys()
        for k in labels.labels:
            assert labels.labels[k].equals(sdata.labels[k])

    def test_shapes(self, tmp_path: str, shapes: SpatialData) -> None:
        """Test read/write."""
        tmpdir = Path(tmp_path) / "tmp.zarr"
        shapes.write(tmpdir)
        sdata = SpatialData.read(tmpdir)
        assert shapes.shapes.keys() == sdata.shapes.keys()
        for k in shapes.shapes:
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
        for k in points.points:
            assert isinstance(sdata.points[k], DaskDataFrame)
            assert assert_eq(points.points[k], sdata.points[k], check_divisions=False)
            assert points.points[k].attrs == points.points[k].attrs

    def _test_table(self, tmp_path: str, table: SpatialData) -> None:
        tmpdir = Path(tmp_path) / "tmp.zarr"
        table.write(tmpdir)
        sdata = SpatialData.read(tmpdir)
        pd.testing.assert_frame_equal(table["table"].obs, sdata["table"].obs)
        try:
            assert table["table"].uns == sdata["table"].uns
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
        _are_directories_identical(tmpdir, tmpdir2, exclude_regexp="[1-9][0-9]*.*")

    def test_incremental_io(
        self,
        tmp_path: str,
        full_sdata: SpatialData,
    ) -> None:
        tmpdir = Path(tmp_path) / "tmp.zarr"
        sdata = full_sdata

        sdata.images["sdata_not_saved_yet"] = _get_images().values().__iter__().__next__()
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
            sdata.images[f"incremental_{k}"] = v
            with pytest.warns(UserWarning):
                sdata.images[f"incremental_{k}"] = v
                sdata[f"incremental_{k}"] = v

        for k, v in _get_labels().items():
            if isinstance(v, SpatialImage):
                v.name = f"incremental_{k}"
            elif isinstance(v, MultiscaleSpatialImage):
                for scale in v:
                    names = list(v[scale].keys())
                    assert len(names) == 1
                    name = names[0]
                    v[scale] = v[scale].rename_vars({name: f"incremental_{k}"})
            sdata.labels[f"incremental_{k}"] = v
            with pytest.warns(UserWarning):
                sdata.labels[f"incremental_{k}"] = v
                sdata[f"incremental_{k}"] = v

        for k, v in _get_shapes().items():
            sdata.shapes[f"incremental_{k}"] = v
            with pytest.warns(UserWarning):
                sdata.shapes[f"incremental_{k}"] = v
                sdata[f"incremental_{k}"] = v
            break

        for k, v in _get_points().items():
            sdata.points[f"incremental_{k}"] = v
            with pytest.warns(UserWarning):
                sdata.points[f"incremental_{k}"] = v
                sdata[f"incremental_{k}"] = v
            break

    def test_incremental_io_table(self, table_single_annotation: SpatialData) -> None:
        s = table_single_annotation
        t = s.table[:10, :].copy()
        with pytest.raises(ValueError):
            s.table = t
        del s.table
        s.table = t

        with tempfile.TemporaryDirectory() as td:
            f = os.path.join(td, "data.zarr")
            s.write(f)
            s2 = SpatialData.read(f)
            assert len(s2.table) == len(t)
            del s2.table
            s2.table = s.table
            assert len(s2.table) == len(s.table)
            f2 = os.path.join(td, "data2.zarr")
            s2.write(f2)
            s3 = SpatialData.read(f2)
            assert len(s3.table) == len(s2.table)

    def test_io_and_lazy_loading_points(self, points):
        elem_name = list(points.points.keys())[0]
        with tempfile.TemporaryDirectory() as td:
            f = os.path.join(td, "data.zarr")
            dask0 = points.points[elem_name]
            points.write(f)
            assert all("read-parquet" not in key for key in dask0.dask.layers)
            dask1 = read_zarr(f).points[elem_name]
            assert any("read-parquet" in key for key in dask1.dask.layers)

    def test_io_and_lazy_loading_raster(self, images, labels):
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
                dask1 = read_zarr(f)[elem_name].data
                assert any("from-zarr" in key for key in dask1.dask.layers)

    def test_replace_transformation_on_disk_raster(self, images, labels):
        sdatas = {"images": images, "labels": labels}
        for k, sdata in sdatas.items():
            d = sdata.__getattribute__(k)
            # unlike the non-raster case we are testing all the elements (2d and 3d, multiscale and not)
            # TODO: we can actually later on merge this test and the one below keepin the logic of this function here
            for elem_name in d:
                kwargs = {k: {elem_name: d[elem_name]}}
                single_sdata = SpatialData(**kwargs)
                with tempfile.TemporaryDirectory() as td:
                    f = os.path.join(td, "data.zarr")
                    single_sdata.write(f)
                    t0 = get_transformation(SpatialData.read(f).__getattribute__(k)[elem_name])
                    assert type(t0) == Identity
                    set_transformation(
                        single_sdata.__getattribute__(k)[elem_name],
                        Scale([2.0], axes=("x",)),
                        write_to_sdata=single_sdata,
                    )
                    t1 = get_transformation(SpatialData.read(f).__getattribute__(k)[elem_name])
                    assert type(t1) == Scale

    def test_replace_transformation_on_disk_non_raster(self, shapes, points):
        sdatas = {"shapes": shapes, "points": points}
        for k, sdata in sdatas.items():
            d = sdata.__getattribute__(k)
            elem_name = list(d.keys())[0]
            with tempfile.TemporaryDirectory() as td:
                f = os.path.join(td, "data.zarr")
                sdata.write(f)
                t0 = get_transformation(SpatialData.read(f).__getattribute__(k)[elem_name])
                assert type(t0) == Identity
                set_transformation(
                    sdata.__getattribute__(k)[elem_name], Scale([2.0], axes=("x",)), write_to_sdata=sdata
                )
                t1 = get_transformation(SpatialData.read(f).__getattribute__(k)[elem_name])
                assert type(t1) == Scale

    def test_overwrite_files_without_backed_data(self, full_sdata):
        with tempfile.TemporaryDirectory() as tmpdir:
            f = os.path.join(tmpdir, "data.zarr")
            old_data = SpatialData()
            old_data.write(f)
            # Since not backed, no risk of overwriting backing data.
            # Should not raise "The file path specified is the same as the one used for backing."
            full_sdata.write(f, overwrite=True)

    def test_not_overwrite_files_without_backed_data_but_with_dask_backed_data(self, full_sdata, points):
        with tempfile.TemporaryDirectory() as tmpdir:
            f = os.path.join(tmpdir, "data.zarr")
            points.write(f)
            points2 = SpatialData.read(f)
            p = points2["points_0"]
            full_sdata["points_0"] = p
            with pytest.raises(
                ValueError,
                match="The file path specified is a parent directory of one or more files used for backing for one or ",
            ):
                full_sdata.write(f, overwrite=True)

    def test_overwrite_files_with_backed_data(self, full_sdata):
        # addressing https://github.com/scverse/spatialdata/issues/137
        with tempfile.TemporaryDirectory() as tmpdir:
            f = os.path.join(tmpdir, "data.zarr")
            full_sdata.write(f)
            with pytest.raises(ValueError, match="The file path specified is the same as the one used for backing."):
                full_sdata.write(f, overwrite=True)

        # support for overwriting backed sdata has been temporarily removed
        # with tempfile.TemporaryDirectory() as tmpdir:
        #     f = os.path.join(tmpdir, "data.zarr")
        #     full_sdata.write(f)
        #     full_sdata.write(f, overwrite=True)
        #     print(full_sdata)
        #
        #     sdata2 = SpatialData(
        #         images=full_sdata.images,
        #         labels=full_sdata.labels,
        #         points=full_sdata.points,
        #         shapes=full_sdata.shapes,
        #         table=full_sdata.table,
        #     )
        #     sdata2.write(f, overwrite=True)

    def test_overwrite_onto_non_zarr_file(self, full_sdata):
        with tempfile.TemporaryDirectory() as tmpdir:
            f0 = os.path.join(tmpdir, "test.txt")
            with open(f0, "w"):
                with pytest.raises(ValueError):
                    full_sdata.write(f0)
                with pytest.raises(ValueError):
                    full_sdata.write(f0, overwrite=True)
            f1 = os.path.join(tmpdir, "test.zarr")
            os.mkdir(f1)
            with pytest.raises(ValueError):
                full_sdata.write(f1)


def test_io_table(shapes):
    adata = AnnData(X=RNG.normal(size=(5, 10)))
    adata.obs["region"] = "circles"
    adata.obs["instance"] = shapes.shapes["circles"].index
    adata = TableModel().parse(adata, region="circles", region_key="region", instance_key="instance")
    shapes.table = adata
    del shapes.tables["table"]
    shapes.table = adata
    with tempfile.TemporaryDirectory() as tmpdir:
        f = os.path.join(tmpdir, "data.zarr")
        shapes.write(f)
        shapes2 = SpatialData.read(f)
        assert "table" in shapes2.tables
        assert shapes2["table"].shape == (5, 10)

        del shapes2.tables["table"]
        assert "table" not in shapes2.tables
        shapes2.table = adata
        assert "table" in shapes2.tables
        assert shapes2["table"].shape == (5, 10)


def test_bug_rechunking_after_queried_raster():
    # https://github.com/scverse/spatialdata-io/issues/117
    ##
    single_scale = Image2DModel.parse(RNG.random((100, 10, 10)), chunks=(5, 5, 5))
    multi_scale = Image2DModel.parse(RNG.random((100, 10, 10)), scale_factors=[2, 2], chunks=(5, 5, 5))
    images = {"single_scale": single_scale, "multi_scale": multi_scale}
    sdata = SpatialData(images=images)
    queried = sdata.query.bounding_box(
        axes=("x", "y"), min_coordinate=[2, 5], max_coordinate=[12, 12], target_coordinate_system="global"
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        f = os.path.join(tmpdir, "data.zarr")
        queried.write(f)

    ##

    pass
