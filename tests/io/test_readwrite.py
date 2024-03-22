import os
import tempfile
from pathlib import Path

import dask.dataframe as dd
import numpy as np
import pytest
from anndata import AnnData
from numpy.random import default_rng
from spatialdata import SpatialData
from spatialdata._io._utils import _are_directories_identical, get_channels, get_dask_backing_files
from spatialdata.models import Image2DModel
from spatialdata.testing import assert_spatial_data_objects_are_identical
from spatialdata.transformations.operations import (
    get_transformation,
    set_transformation,
)
from spatialdata.transformations.transformations import Identity, Scale

from tests.conftest import _get_images, _get_labels, _get_points, _get_shapes

RNG = default_rng(0)


class TestReadWrite:
    def test_images(self, tmp_path: str, images: SpatialData) -> None:
        tmpdir = Path(tmp_path) / "tmp.zarr"

        # ensures that we are inplicitly testing the read and write of channel names
        assert get_channels(images["image2d"]) == ["r", "g", "b"]
        assert get_channels(images["image2d_multiscale"]) == ["r", "g", "b"]

        images.write(tmpdir)
        sdata = SpatialData.read(tmpdir)
        assert_spatial_data_objects_are_identical(images, sdata)

    def test_labels(self, tmp_path: str, labels: SpatialData) -> None:
        tmpdir = Path(tmp_path) / "tmp.zarr"
        labels.write(tmpdir)
        sdata = SpatialData.read(tmpdir)
        assert_spatial_data_objects_are_identical(labels, sdata)

    def test_shapes(self, tmp_path: str, shapes: SpatialData) -> None:
        tmpdir = Path(tmp_path) / "tmp.zarr"

        # check the index is correctly written and then read
        shapes["circles"].index = np.arange(1, len(shapes["circles"]) + 1)

        shapes.write(tmpdir)
        sdata = SpatialData.read(tmpdir)
        assert_spatial_data_objects_are_identical(shapes, sdata)

    def test_points(self, tmp_path: str, points: SpatialData) -> None:
        tmpdir = Path(tmp_path) / "tmp.zarr"

        # check the index is correctly written and then read
        new_index = dd.from_array(np.arange(1, len(points["points_0"]) + 1))
        points["points_0"] = points["points_0"].set_index(new_index)

        points.write(tmpdir)
        sdata = SpatialData.read(tmpdir)
        assert_spatial_data_objects_are_identical(points, sdata)

    def _test_table(self, tmp_path: str, table: SpatialData) -> None:
        tmpdir = Path(tmp_path) / "tmp.zarr"
        table.write(tmpdir)
        sdata = SpatialData.read(tmpdir)
        assert_spatial_data_objects_are_identical(table, sdata)

    def test_single_table_single_annotation(self, tmp_path: str, table_single_annotation: SpatialData) -> None:
        self._test_table(tmp_path, table_single_annotation)

    def test_single_table_multiple_annotations(self, tmp_path: str, table_multiple_annotations: SpatialData) -> None:
        self._test_table(tmp_path, table_multiple_annotations)

    def test_multiple_tables(self, tmp_path: str, tables: list[AnnData]) -> None:
        sdata_tables = SpatialData(tables={str(i): tables[i] for i in range(len(tables))})
        self._test_table(tmp_path, sdata_tables)

    def test_roundtrip(
        self,
        tmp_path: str,
        sdata: SpatialData,
    ) -> None:
        tmpdir = Path(tmp_path) / "tmp.zarr"

        sdata.write(tmpdir)
        sdata2 = SpatialData.read(tmpdir)
        tmpdir2 = Path(tmp_path) / "tmp2.zarr"
        sdata2.write(tmpdir2)
        _are_directories_identical(tmpdir, tmpdir2, exclude_regexp="[1-9][0-9]*.*")

    def test_incremental_io_in_memory(
        self,
        full_sdata: SpatialData,
    ) -> None:
        sdata = full_sdata

        for k, v in _get_images().items():
            sdata.images[f"additional_{k}"] = v
            with pytest.warns(UserWarning):
                sdata.images[f"additional_{k}"] = v
                sdata[f"additional_{k}"] = v
            with pytest.raises(KeyError, match="Key `table` already exists."):
                sdata["table"] = v

        for k, v in _get_labels().items():
            sdata.labels[f"additional_{k}"] = v
            with pytest.warns(UserWarning):
                sdata.labels[f"additional_{k}"] = v
                sdata[f"additional_{k}"] = v
            with pytest.raises(KeyError, match="Key `table` already exists."):
                sdata["table"] = v

        for k, v in _get_shapes().items():
            sdata.shapes[f"additional_{k}"] = v
            with pytest.warns(UserWarning):
                sdata.shapes[f"additional_{k}"] = v
                sdata[f"additional_{k}"] = v
            with pytest.raises(KeyError, match="Key `table` already exists."):
                sdata["table"] = v

        for k, v in _get_points().items():
            sdata.points[f"additional_{k}"] = v
            with pytest.warns(UserWarning):
                sdata.points[f"additional_{k}"] = v
                sdata[f"additional_{k}"] = v
            with pytest.raises(KeyError, match="Key `table` already exists."):
                sdata["table"] = v

    def test_incremental_io_on_disk(self, tmp_path: str, full_sdata: SpatialData):
        tmpdir = Path(tmp_path) / "incremental_io.zarr"
        sdata = SpatialData()
        sdata.write(tmpdir)

        for name in [
            "image2d",
            "image3d_multiscale_xarray",
            "labels2d",
            "labels3d_multiscale_xarray",
            "points_0",
            "multipoly",
            "table",
        ]:
            sdata[name] = full_sdata[name]
            sdata.write_element(name)

            with pytest.raises(
                ValueError, match="The Zarr store already exists. Use `overwrite=True` to try overwriting the store."
            ):
                sdata.write_element(name)

            with pytest.raises(ValueError, match="Currently, overwriting existing elements is not supported."):
                sdata.write_element(name, overwrite=True)

            # workaround 1, mostly safe (no guarantee: untested for Windows platform, network drives, multi-threaded
            # setups, ...)
            new_name = f"{name}_new_place"
            sdata[new_name] = sdata[name]
            sdata.write_element(new_name)
            # TODO: del sdata[name] on-disk
            # TODO: sdata.write(name)
            # TODO: del sdata['new_place'] on-disk
            # TODO: del sdata['new_place']

            # workaround 2, unsafe but sometimes acceptable depending on the user's workflow
            # TODO: del[sdata] on-diks
            # TODO: sdata.write(name)

    def test_incremental_io_table_legacy(self, table_single_annotation: SpatialData) -> None:
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
            assert len(get_dask_backing_files(points)) == 0

            sdata2 = SpatialData.read(f)
            dask1 = sdata2[elem_name]
            assert any("read-parquet" in key for key in dask1.dask.layers)
            assert len(get_dask_backing_files(sdata2)) > 0

    def test_io_and_lazy_loading_raster(self, images, labels):
        sdatas = {"images": images, "labels": labels}
        for k, sdata in sdatas.items():
            d = sdata.__getattribute__(k)
            elem_name = list(d.keys())[0]
            with tempfile.TemporaryDirectory() as td:
                f = os.path.join(td, "data.zarr")
                dask0 = sdata[elem_name].data
                sdata.write(f)
                assert all("from-zarr" not in key for key in dask0.dask.layers)
                assert len(get_dask_backing_files(sdata)) == 0

                sdata2 = SpatialData.read(f)
                dask1 = sdata2[elem_name].data
                assert any("from-zarr" in key for key in dask1.dask.layers)
                assert len(get_dask_backing_files(sdata2)) > 0

    def test_replace_transformation_on_disk_raster(self, images, labels):
        sdatas = {"images": images, "labels": labels}
        for k, sdata in sdatas.items():
            d = sdata.__getattribute__(k)
            # unlike the non-raster case, we are testing all the elements (2d and 3d, multiscale and not)
            for elem_name in d:
                kwargs = {k: {elem_name: d[elem_name]}}
                single_sdata = SpatialData(**kwargs)
                with tempfile.TemporaryDirectory() as td:
                    f = os.path.join(td, "data.zarr")
                    single_sdata.write(f)
                    t0 = get_transformation(SpatialData.read(f)[elem_name])
                    assert type(t0) == Identity
                    set_transformation(
                        single_sdata[elem_name],
                        Scale([2.0], axes=("x",)),
                        write_to_sdata=single_sdata,
                    )
                    t1 = get_transformation(SpatialData.read(f)[elem_name])
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
                set_transformation(sdata[elem_name], Scale([2.0], axes=("x",)), write_to_sdata=sdata)
                t1 = get_transformation(SpatialData.read(f)[elem_name])
                assert type(t1) == Scale

    def test_overwrite_works_when_no_zarr_store(self, full_sdata):
        with tempfile.TemporaryDirectory() as tmpdir:
            f = os.path.join(tmpdir, "data.zarr")
            old_data = SpatialData()
            old_data.write(f)
            # Since no, no risk of overwriting backing data.
            # Should not raise "The file path specified is the same as the one used for backing."
            full_sdata.write(f, overwrite=True)

    def test_overwrite_fails_when_no_zarr_store_bug_dask_backed_data(self, full_sdata, points, images, labels):
        sdatas = {"images": images, "labels": labels, "points": points}
        elements = {"images": "image2d", "labels": "labels2d", "points": "points_0"}
        for k, sdata in sdatas.items():
            element = elements[k]
            with tempfile.TemporaryDirectory() as tmpdir:
                f = os.path.join(tmpdir, "data.zarr")
                sdata.write(f)

                # now we have a sdata with dask-backed elements
                sdata2 = SpatialData.read(f)
                p = sdata2[element]
                full_sdata[element] = p
                with pytest.raises(
                    ValueError,
                    match="The Zarr store already exists. Use `overwrite=True` to try overwriting the store.",
                ):
                    full_sdata.write(f)

                with pytest.raises(
                    ValueError,
                    match="The file path specified is a parent directory of one or more files used for backing for one "
                    "or ",
                ):
                    full_sdata.write(f, overwrite=True)

    def test_overwrite_fails_when_zarr_store_present(self, full_sdata):
        # addressing https://github.com/scverse/spatialdata/issues/137
        with tempfile.TemporaryDirectory() as tmpdir:
            f = os.path.join(tmpdir, "data.zarr")
            full_sdata.write(f)

            with pytest.raises(
                ValueError,
                match="The Zarr store already exists. Use `overwrite=True` to try overwriting the store.",
            ):
                full_sdata.write(f)

            with pytest.raises(
                ValueError,
                match="The file path specified either contains either is contained in the one used for backing.",
            ):
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

    def test_overwrite_fails_onto_non_zarr_file(self, full_sdata):
        ERROR_MESSAGE = (
            "The target file path specified already exists, and it has been detected to not be a Zarr store."
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            f0 = os.path.join(tmpdir, "test.txt")
            with open(f0, "w"):
                with pytest.raises(
                    ValueError,
                    match=ERROR_MESSAGE,
                ):
                    full_sdata.write(f0)
                with pytest.raises(
                    ValueError,
                    match=ERROR_MESSAGE,
                ):
                    full_sdata.write(f0, overwrite=True)
            f1 = os.path.join(tmpdir, "test.zarr")
            os.mkdir(f1)
            with pytest.raises(ValueError, match=ERROR_MESSAGE):
                full_sdata.write(f1)
            with pytest.raises(ValueError, match=ERROR_MESSAGE):
                full_sdata.write(f1, overwrite=True)


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
