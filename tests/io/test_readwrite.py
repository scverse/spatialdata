import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import dask.dataframe as dd
import numpy as np
import pytest
from anndata import AnnData
from numpy.random import default_rng

from spatialdata import SpatialData, deepcopy, read_zarr
from spatialdata._io._utils import _are_directories_identical, get_dask_backing_files
from spatialdata.datasets import blobs
from spatialdata.models import Image2DModel
from spatialdata.models._utils import get_channels
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

    def test_incremental_io_list_of_elements(self, shapes: SpatialData) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            f = os.path.join(tmpdir, "data.zarr")
            shapes.write(f)
            new_shapes0 = deepcopy(shapes["circles"])
            new_shapes1 = deepcopy(shapes["poly"])
            shapes["new_shapes0"] = new_shapes0
            shapes["new_shapes1"] = new_shapes1
            assert "shapes/new_shapes0" not in shapes.elements_paths_on_disk()
            assert "shapes/new_shapes1" not in shapes.elements_paths_on_disk()

            shapes.write_element(["new_shapes0", "new_shapes1"])
            assert "shapes/new_shapes0" in shapes.elements_paths_on_disk()
            assert "shapes/new_shapes1" in shapes.elements_paths_on_disk()

            shapes.delete_element_from_disk(["new_shapes0", "new_shapes1"])
            assert "shapes/new_shapes0" not in shapes.elements_paths_on_disk()
            assert "shapes/new_shapes1" not in shapes.elements_paths_on_disk()

    @pytest.mark.parametrize("dask_backed", [True, False])
    @pytest.mark.parametrize("workaround", [1, 2])
    def test_incremental_io_on_disk(
        self, tmp_path: str, full_sdata: SpatialData, dask_backed: bool, workaround: int
    ) -> None:
        """
        This tests shows workaround on how to rewrite existing data on disk.

        The user is recommended to study them, understand the implications and eventually adapt them to their use case.
        We are working on simpler workarounds and on a more robust solution to this problem, but unfortunately it is not
        yet available.

        In particular the complex "dask-backed" case for workaround 1 could be simplified once
        """
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
            if dask_backed:
                # this forces the element to write to be dask-backed from disk. In this case, overwriting the data is
                # more laborious because we are writing the data to the same location that defines the data!
                sdata = read_zarr(sdata.path)

            with pytest.raises(
                ValueError, match="The Zarr store already exists. Use `overwrite=True` to try overwriting the store."
            ):
                sdata.write_element(name)

            with pytest.raises(ValueError, match="Cannot overwrite."):
                sdata.write_element(name, overwrite=True)

            if workaround == 1:
                new_name = f"{name}_new_place"
                # workaround 1, mostly safe (untested for Windows platform, network drives, multi-threaded
                # setups, ...). If the scenario matches your use case, please use with caution.

                if not dask_backed:  # easier case
                    # a. write a backup copy of the data
                    sdata[new_name] = sdata[name]
                    sdata.write_element(new_name)
                    # b. rewrite the original data
                    sdata.delete_element_from_disk(name)
                    sdata.write_element(name)
                    # c. remove the backup copy
                    del sdata[new_name]
                    sdata.delete_element_from_disk(new_name)
                else:  # dask-backed case, more complex
                    # a. write a backup copy of the data
                    sdata[new_name] = sdata[name]
                    sdata.write_element(new_name)
                    # a2. remove the in-memory copy from the SpatialData object (note,
                    # at this point the backup copy still exists on-disk)
                    del sdata[new_name]
                    del sdata[name]
                    # a3 load the backup copy into memory
                    sdata_copy = read_zarr(sdata.path)
                    # b1. rewrite the original data
                    sdata.delete_element_from_disk(name)
                    sdata[name] = sdata_copy[new_name]
                    sdata.write_element(name)
                    # b2. reload the new data into memory (because it has been written but in-memory it still points
                    # from the backup location)
                    sdata = read_zarr(sdata.path)
                    # c. remove the backup copy
                    del sdata[new_name]
                    sdata.delete_element_from_disk(new_name)
            elif workaround == 2:
                # workaround 2, unsafe but sometimes acceptable depending on the user's workflow.

                # this works only if the data is not dask-backed, otherwise an exception will be raised because the code
                # would be trying to delete the data that the Dask object is pointing to!
                if not dask_backed:
                    # a. rewrite the original data (risky!)
                    sdata.delete_element_from_disk(name)
                    sdata.write_element(name)

    def test_incremental_io_table_legacy(self, table_single_annotation: SpatialData) -> None:
        s = table_single_annotation
        t = s["table"][:10, :].copy()
        with pytest.raises(ValueError):
            s.table = t
        del s["table"]
        s.table = t

        with tempfile.TemporaryDirectory() as td:
            f = os.path.join(td, "data.zarr")
            s.write(f)
            s2 = SpatialData.read(f)
            assert len(s2["table"]) == len(t)
            del s2["table"]
            s2.table = s["table"]
            assert len(s2["table"]) == len(s["table"])
            f2 = os.path.join(td, "data2.zarr")
            s2.write(f2)
            s3 = SpatialData.read(f2)
            assert len(s3["table"]) == len(s2["table"])

    def test_io_and_lazy_loading_points(self, points):
        with tempfile.TemporaryDirectory() as td:
            f = os.path.join(td, "data.zarr")
            points.write(f)
            assert len(get_dask_backing_files(points)) == 0

            sdata2 = SpatialData.read(f)
            assert len(get_dask_backing_files(sdata2)) > 0

    def test_io_and_lazy_loading_raster(self, images, labels):
        sdatas = {"images": images, "labels": labels}
        for k, sdata in sdatas.items():
            d = getattr(sdata, k)
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
            d = getattr(sdata, k)
            # unlike the non-raster case, we are testing all the elements (2d and 3d, multiscale and not)
            for elem_name in d:
                kwargs = {k: {elem_name: d[elem_name]}}
                single_sdata = SpatialData(**kwargs)
                with tempfile.TemporaryDirectory() as td:
                    f = os.path.join(td, "data.zarr")
                    single_sdata.write(f)
                    t0 = get_transformation(SpatialData.read(f)[elem_name])
                    assert isinstance(t0, Identity)
                    set_transformation(
                        single_sdata[elem_name],
                        Scale([2.0], axes=("x",)),
                        write_to_sdata=single_sdata,
                    )
                    t1 = get_transformation(SpatialData.read(f)[elem_name])
                    assert isinstance(t1, Scale)

    def test_replace_transformation_on_disk_non_raster(self, shapes, points):
        sdatas = {"shapes": shapes, "points": points}
        for k, sdata in sdatas.items():
            d = sdata.__getattribute__(k)
            elem_name = list(d.keys())[0]
            with tempfile.TemporaryDirectory() as td:
                f = os.path.join(td, "data.zarr")
                sdata.write(f)
                t0 = get_transformation(SpatialData.read(f).__getattribute__(k)[elem_name])
                assert isinstance(t0, Identity)
                set_transformation(sdata[elem_name], Scale([2.0], axes=("x",)), write_to_sdata=sdata)
                t1 = get_transformation(SpatialData.read(f)[elem_name])
                assert isinstance(t1, Scale)

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
                    match="Cannot overwrite.",
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
                match="Cannot overwrite.",
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


def test_self_contained(full_sdata: SpatialData) -> None:
    # data only in-memory, so the SpatialData object and all its elements are self-contained
    assert full_sdata.is_self_contained()
    description = full_sdata.elements_are_self_contained()
    assert all(description.values())

    with tempfile.TemporaryDirectory() as tmpdir:
        # data saved to disk, it's self contained
        f = os.path.join(tmpdir, "data.zarr")
        full_sdata.write(f)
        full_sdata.is_self_contained()

        # we read the data, so it's self-contained
        sdata2 = SpatialData.read(f)
        assert sdata2.is_self_contained()

        # we save the data to a new location, so it's not self-contained anymore
        f2 = os.path.join(tmpdir, "data2.zarr")
        sdata2.write(f2)
        assert not sdata2.is_self_contained()

        # because of the images, labels and points
        description = sdata2.elements_are_self_contained()
        for element_name, self_contained in description.items():
            if any(element_name.startswith(prefix) for prefix in ["image", "labels", "points"]):
                assert not self_contained
            else:
                assert self_contained

        # but if we read it again, it's self-contained
        sdata3 = SpatialData.read(f2)
        assert sdata3.is_self_contained()

        # or if we do some more targeted manipulation
        sdata2.path = Path(f)
        assert sdata2.is_self_contained()

        # finally, an example of a non-self-contained element which is not depending on files external to the Zarr store
        # here we create an element which combines lazily 3 elements of the Zarr store; it will be a nonsense element,
        # but good for testing
        v = sdata2["points_0"]["x"].loc[0].values
        v.compute_chunk_sizes()

        combined = (
            v
            + sdata2["labels2d"].expand_dims("c", 1).transpose("c", "y", "x")
            + sdata2["image2d"].sel(c="r").expand_dims("c", 1)
            + v
        )
        combined = Image2DModel.parse(combined)
        assert len(get_dask_backing_files(combined)) == 3

        sdata2["combined"] = combined

        assert not sdata2.is_self_contained()
        description = sdata2.elements_are_self_contained()
        assert description["combined"] is False
        assert all(description[element_name] for element_name in description if element_name != "combined")


def test_symmetric_different_with_zarr_store(full_sdata: SpatialData) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        f = os.path.join(tmpdir, "data.zarr")
        full_sdata.write(f)

        # the list of element on-disk and in-memory is the same
        only_in_memory, only_on_disk = full_sdata._symmetric_difference_with_zarr_store()
        assert len(only_in_memory) == 0
        assert len(only_on_disk) == 0

        full_sdata["new_image2d"] = full_sdata.images["image2d"]
        full_sdata["new_labels2d"] = full_sdata.labels["labels2d"]
        full_sdata["new_points_0"] = full_sdata.points["points_0"]
        full_sdata["new_circles"] = full_sdata.shapes["circles"]
        full_sdata["new_table"] = full_sdata.tables["table"]
        del full_sdata.images["image2d"]
        del full_sdata.labels["labels2d"]
        del full_sdata.points["points_0"]
        del full_sdata.shapes["circles"]
        del full_sdata.tables["table"]

        # now the list of element on-disk and in-memory is different
        only_in_memory, only_on_disk = full_sdata._symmetric_difference_with_zarr_store()
        assert set(only_in_memory) == {
            "images/new_image2d",
            "labels/new_labels2d",
            "points/new_points_0",
            "shapes/new_circles",
            "tables/new_table",
        }
        assert set(only_on_disk) == {
            "images/image2d",
            "labels/labels2d",
            "points/points_0",
            "shapes/circles",
            "tables/table",
        }


def test_change_path_of_subset(full_sdata: SpatialData) -> None:
    """A subset SpatialData object has not Zarr path associated, show that we can reassign the path"""
    with tempfile.TemporaryDirectory() as tmpdir:
        f = os.path.join(tmpdir, "data.zarr")
        full_sdata.write(f)

        subset = full_sdata.subset(["image2d", "labels2d", "points_0", "circles", "table"])

        assert subset.path is None
        subset.path = Path(f)

        assert subset.is_self_contained()
        only_in_memory, only_on_disk = subset._symmetric_difference_with_zarr_store()
        assert len(only_in_memory) == 0
        assert len(only_on_disk) > 0

        f2 = os.path.join(tmpdir, "data2.zarr")
        subset.write(f2)
        assert subset.is_self_contained()
        only_in_memory, only_on_disk = subset._symmetric_difference_with_zarr_store()
        assert len(only_in_memory) == 0
        assert len(only_on_disk) == 0

        # changes in the subset object will be reflected in the original object (in-memory, on-disk only if we save
        # them with .write_element())
        scale = Scale([2.0], axes=("x",))
        set_transformation(subset["image2d"], scale)
        assert isinstance(get_transformation(full_sdata["image2d"]), Scale)

        # if we don't want this, we can read the data again from the new path
        sdata2 = SpatialData.read(f2)
        set_transformation(sdata2["labels2d"], scale)
        assert not isinstance(get_transformation(full_sdata["labels2d"]), Scale)
        assert not isinstance(get_transformation(subset["labels2d"]), Scale)


def _check_valid_name(f: Callable[[str], Any]) -> None:
    with pytest.raises(TypeError, match="Name must be a string, not "):
        f(2)
    with pytest.raises(ValueError, match="Name cannot be an empty string."):
        f("")
    with pytest.raises(ValueError, match="Name must contain only alphanumeric characters, underscores, and hyphens."):
        f("not valid")
    with pytest.raises(ValueError, match="Name must contain only alphanumeric characters, underscores, and hyphens."):
        f("this/is/not/valid")


def test_incremental_io_valid_name(points: SpatialData) -> None:
    _check_valid_name(points.write_element)
    _check_valid_name(points.write_metadata)
    _check_valid_name(points.write_transformations)
    _check_valid_name(points.delete_element_from_disk)


cached_sdata_blobs = blobs()


@pytest.mark.parametrize("element_name", ["image2d", "labels2d", "points_0", "circles", "table"])
def test_delete_element_from_disk(full_sdata, element_name: str) -> None:
    # can't delete an element for a SpatialData object without associated Zarr store
    with pytest.raises(ValueError, match="The SpatialData object is not backed by a Zarr store."):
        full_sdata.delete_element_from_disk("image2d")

    with tempfile.TemporaryDirectory() as tmpdir:
        f = os.path.join(tmpdir, "data.zarr")
        full_sdata.write(f)

        # cannot delete an element which is in-memory, but not in the Zarr store
        subset = full_sdata.subset(["points_0_1"])
        f2 = os.path.join(tmpdir, "data2.zarr")
        subset.write(f2)
        full_sdata.path = Path(f2)
        with pytest.raises(
            ValueError,
            match=f"Element {element_name} is not found in the Zarr store associated with the SpatialData object.",
        ):
            subset.delete_element_from_disk(element_name)
        full_sdata.path = Path(f)

        # cannot delete an element which is not in the Zarr store (and not even in-memory)
        with pytest.raises(
            ValueError,
            match="Element not_existing is not found in the Zarr store associated with the SpatialData object.",
        ):
            full_sdata.delete_element_from_disk("not_existing")

        # can delete an element present both in-memory and on-disk
        full_sdata.delete_element_from_disk(element_name)
        only_in_memory, only_on_disk = full_sdata._symmetric_difference_with_zarr_store()
        element_type = full_sdata._element_type_from_element_name(element_name)
        element_path = f"{element_type}/{element_name}"
        assert element_path in only_in_memory

        # resave it
        full_sdata.write_element(element_name)

        # now delete it from memory, and then show it can still be deleted on-disk
        del getattr(full_sdata, element_type)[element_name]
        full_sdata.delete_element_from_disk(element_name)
        on_disk = full_sdata.elements_paths_on_disk()
        assert element_path not in on_disk


@pytest.mark.parametrize("element_name", ["image2d", "labels2d", "points_0", "circles", "table"])
def test_element_already_on_disk_different_type(full_sdata, element_name: str) -> None:
    # Constructing a corrupted object (element present both on disk and in-memory but with different type).
    # Attempting to perform and IO operation will trigger an error.
    # The checks assessed in this test will not be needed anymore after
    # https://github.com/scverse/spatialdata/issues/504 is addressed
    with tempfile.TemporaryDirectory() as tmpdir:
        f = os.path.join(tmpdir, "data.zarr")
        full_sdata.write(f)

        element_type = full_sdata._element_type_from_element_name(element_name)
        wrong_group = "images" if element_type == "tables" else "tables"
        del getattr(full_sdata, element_type)[element_name]
        getattr(full_sdata, wrong_group)[element_name] = (
            getattr(cached_sdata_blobs, wrong_group).values().__iter__().__next__()
        )
        ERROR_MSG = "The in-memory object should have a different name."

        with pytest.raises(
            ValueError,
            match=ERROR_MSG,
        ):
            full_sdata.delete_element_from_disk(element_name)

        with pytest.raises(
            ValueError,
            match=ERROR_MSG,
        ):
            full_sdata.write_element(element_name)

        with pytest.raises(
            ValueError,
            match=ERROR_MSG,
        ):
            full_sdata.write_metadata(element_name)

        with pytest.raises(
            ValueError,
            match=ERROR_MSG,
        ):
            full_sdata.write_transformations(element_name)
