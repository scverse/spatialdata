import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest
import zarr
from anndata import AnnData
from numpy.random import default_rng
from shapely import MultiPolygon, Polygon
from upath import UPath
from zarr.errors import GroupNotFoundError

import spatialdata.config
from spatialdata import SpatialData, deepcopy, read_zarr
from spatialdata._core.validation import ValidationError
from spatialdata._io._utils import _are_directories_identical, get_dask_backing_files
from spatialdata._io.format import (
    CurrentSpatialDataContainerFormat,
    SpatialDataContainerFormats,
    SpatialDataContainerFormatType,
    SpatialDataContainerFormatV01,
)
from spatialdata.datasets import blobs
from spatialdata.models import Image2DModel
from spatialdata.models._utils import get_channel_names
from spatialdata.testing import assert_spatial_data_objects_are_identical
from spatialdata.transformations.operations import (
    get_transformation,
    set_transformation,
)
from spatialdata.transformations.transformations import Identity, Scale
from tests.conftest import (
    _get_images,
    _get_labels,
    _get_points,
    _get_shapes,
    _get_table,
    _get_tables,
)

RNG = default_rng(0)
SDATA_FORMATS = list(SpatialDataContainerFormats.values())


@pytest.mark.parametrize("sdata_container_format", SDATA_FORMATS)
class TestReadWrite:
    def test_images(
        self,
        tmp_path: str,
        images: SpatialData,
        sdata_container_format: SpatialDataContainerFormatType,
    ) -> None:
        tmpdir = Path(tmp_path) / "tmp.zarr"

        # ensures that we are inplicitly testing the read and write of channel names
        assert get_channel_names(images["image2d"]) == ["r", "g", "b"]
        assert get_channel_names(images["image2d_multiscale"]) == ["r", "g", "b"]

        images.write(tmpdir, sdata_formats=sdata_container_format)
        sdata = SpatialData.read(tmpdir)
        assert_spatial_data_objects_are_identical(images, sdata)

    def test_labels(
        self,
        tmp_path: str,
        labels: SpatialData,
        sdata_container_format: SpatialDataContainerFormatType,
    ) -> None:
        tmpdir = Path(tmp_path) / "tmp.zarr"
        labels.write(tmpdir, sdata_formats=sdata_container_format)
        sdata = SpatialData.read(tmpdir)
        assert_spatial_data_objects_are_identical(labels, sdata)

    @pytest.mark.parametrize("geometry_encoding", ["WKB", "geoarrow"])
    def test_shapes(
        self,
        tmp_path: str,
        shapes: SpatialData,
        sdata_container_format: SpatialDataContainerFormatType,
        geometry_encoding: Literal["WKB", "geoarrow"],
    ) -> None:
        tmpdir = Path(tmp_path) / "tmp.zarr"

        # check the index is correctly written and then read
        shapes["circles"].index = np.arange(1, len(shapes["circles"]) + 1)

        # add a mixed Polygon + MultiPolygon element
        shapes["mixed"] = pd.concat([shapes["poly"], shapes["multipoly"]])

        shapes.write(tmpdir, sdata_formats=sdata_container_format, shapes_geometry_encoding=geometry_encoding)
        sdata = SpatialData.read(tmpdir)

        if geometry_encoding == "WKB":
            assert_spatial_data_objects_are_identical(shapes, sdata)
        else:
            # convert each Polygon to a MultiPolygon
            mixed_multipolygon = shapes["mixed"].assign(
                geometry=lambda df: df.geometry.apply(lambda g: MultiPolygon([g]) if isinstance(g, Polygon) else g)
            )
            assert sdata["mixed"].equals(mixed_multipolygon)
            assert not sdata["mixed"].equals(shapes["mixed"])

            del shapes["mixed"]
            del sdata["mixed"]
            assert_spatial_data_objects_are_identical(shapes, sdata)

    @pytest.mark.parametrize("geometry_encoding", ["WKB", "geoarrow"])
    def test_shapes_geometry_encoding_write_element(
        self,
        tmp_path: str,
        shapes: SpatialData,
        sdata_container_format: SpatialDataContainerFormatType,
        geometry_encoding: Literal["WKB", "geoarrow"],
    ) -> None:
        """Test shapes geometry encoding with write_element() and global settings."""
        tmpdir = Path(tmp_path) / "tmp.zarr"

        # First write an empty SpatialData to create the zarr store
        empty_sdata = SpatialData()
        empty_sdata.write(tmpdir, sdata_formats=sdata_container_format)

        shapes["mixed"] = pd.concat([shapes["poly"], shapes["multipoly"]])

        # Add shapes to the empty sdata
        for shape_name in shapes.shapes:
            empty_sdata[shape_name] = shapes[shape_name]

        # Store original setting and set global encoding
        original_encoding = spatialdata.config.settings.shapes_geometry_encoding
        try:
            spatialdata.config.settings.shapes_geometry_encoding = geometry_encoding

            # Write each shape element - should use global setting
            for shape_name in shapes.shapes:
                empty_sdata.write_element(shape_name, sdata_formats=sdata_container_format)

                # Verify the encoding metadata in the parquet file
                parquet_file = tmpdir / "shapes" / shape_name / "shapes.parquet"
                with pq.ParquetFile(parquet_file) as pf:
                    md = pf.metadata
                    d = json.loads(md.metadata[b"geo"].decode("utf-8"))
                    found_encoding = d["columns"]["geometry"]["encoding"]
                    if geometry_encoding == "WKB":
                        expected_encoding = "WKB"
                    elif shape_name == "circles":
                        expected_encoding = "point"
                    elif shape_name == "poly":
                        expected_encoding = "polygon"
                    elif shape_name in ["multipoly", "mixed"]:
                        expected_encoding = "multipolygon"
                    else:
                        raise ValueError(
                            f"Uncovered case for shape_name: {shape_name}, found encoding: {found_encoding}."
                        )
                    assert found_encoding == expected_encoding
        finally:
            spatialdata.config.settings.shapes_geometry_encoding = original_encoding

    def test_points(
        self,
        tmp_path: str,
        points: SpatialData,
        sdata_container_format: SpatialDataContainerFormatType,
    ) -> None:
        tmpdir = Path(tmp_path) / "tmp.zarr"

        # check the index is correctly written and then read
        new_index = dd.from_array(np.arange(1, len(points["points_0"]) + 1))
        points["points_0"] = points["points_0"].set_index(new_index)

        points.write(tmpdir, sdata_formats=sdata_container_format)
        sdata = SpatialData.read(tmpdir)
        assert_spatial_data_objects_are_identical(points, sdata)

    def _test_table(
        self,
        tmp_path: str,
        table: SpatialData,
        sdata_container_format: SpatialDataContainerFormatType,
    ) -> None:
        tmpdir = Path(tmp_path) / "tmp.zarr"
        table.write(tmpdir, sdata_formats=sdata_container_format)
        sdata = SpatialData.read(tmpdir)
        assert_spatial_data_objects_are_identical(table, sdata)

    def test_single_table_single_annotation(
        self,
        tmp_path: str,
        table_single_annotation: SpatialData,
        sdata_container_format: SpatialDataContainerFormatType,
    ) -> None:
        self._test_table(
            tmp_path,
            table_single_annotation,
            sdata_container_format=sdata_container_format,
        )

    def test_single_table_multiple_annotations(
        self,
        tmp_path: str,
        table_multiple_annotations: SpatialData,
        sdata_container_format: SpatialDataContainerFormatType,
    ) -> None:
        self._test_table(
            tmp_path,
            table_multiple_annotations,
            sdata_container_format=sdata_container_format,
        )

    def test_multiple_tables(
        self,
        tmp_path: str,
        tables: list[AnnData],
        sdata_container_format: SpatialDataContainerFormatType,
    ) -> None:
        sdata_tables = SpatialData(tables={str(i): tables[i] for i in range(len(tables))})
        self._test_table(tmp_path, sdata_tables, sdata_container_format=sdata_container_format)

    def test_roundtrip(
        self,
        tmp_path: str,
        sdata: SpatialData,
        sdata_container_format: SpatialDataContainerFormatType,
    ) -> None:
        tmpdir = Path(tmp_path) / "tmp.zarr"

        sdata.write(tmpdir, sdata_formats=sdata_container_format)
        sdata2 = SpatialData.read(tmpdir)
        tmpdir2 = Path(tmp_path) / "tmp2.zarr"
        sdata2.write(tmpdir2, sdata_formats=sdata_container_format)
        _are_directories_identical(tmpdir, tmpdir2, exclude_regexp="[1-9][0-9]*.*")

    def test_incremental_io_list_of_elements(
        self,
        shapes: SpatialData,
        sdata_container_format: SpatialDataContainerFormatType,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            f = os.path.join(tmpdir, "data.zarr")
            shapes.write(f, sdata_formats=sdata_container_format)
            new_shapes0 = deepcopy(shapes["circles"])
            new_shapes1 = deepcopy(shapes["poly"])
            shapes["new_shapes0"] = new_shapes0
            shapes["new_shapes1"] = new_shapes1
            assert "shapes/new_shapes0" not in shapes.elements_paths_on_disk()
            assert "shapes/new_shapes1" not in shapes.elements_paths_on_disk()

            shapes.write_element(["new_shapes0", "new_shapes1"], sdata_formats=sdata_container_format)
            assert "shapes/new_shapes0" in shapes.elements_paths_on_disk()
            assert "shapes/new_shapes1" in shapes.elements_paths_on_disk()

            shapes.delete_element_from_disk(["new_shapes0", "new_shapes1"])
            assert "shapes/new_shapes0" not in shapes.elements_paths_on_disk()
            assert "shapes/new_shapes1" not in shapes.elements_paths_on_disk()

    @staticmethod
    def _workaround1_non_dask_backed(
        sdata: SpatialData,
        name: str,
        new_name: str,
        sdata_container_format: SpatialDataContainerFormatType = CurrentSpatialDataContainerFormat(),
    ) -> None:
        # a. write a backup copy of the data
        sdata[new_name] = sdata[name]
        sdata.write_element(new_name, sdata_formats=sdata_container_format)
        # b. rewrite the original data
        sdata.delete_element_from_disk(name)
        sdata.write_element(name, sdata_formats=sdata_container_format)
        # c. remove the backup copy
        del sdata[new_name]
        sdata.delete_element_from_disk(new_name)

    @staticmethod
    def _workaround1_dask_backed(
        sdata: SpatialData,
        name: str,
        new_name: str,
        sdata_container_format: SpatialDataContainerFormatType = CurrentSpatialDataContainerFormat(),
    ) -> None:
        # a. write a backup copy of the data
        sdata[new_name] = sdata[name]
        sdata.write_element(new_name, sdata_formats=sdata_container_format)
        # a2. remove the in-memory copy from the SpatialData object (note,
        # at this point the backup copy still exists on-disk)
        del sdata[new_name]
        del sdata[name]
        # a3 load the backup copy into memory
        sdata_copy = read_zarr(sdata.path)
        # b1. rewrite the original data
        sdata.delete_element_from_disk(name)
        sdata[name] = sdata_copy[new_name]
        sdata.write_element(name, sdata_formats=sdata_container_format)
        # b2. reload the new data into memory (because it has been written but in-memory it still points
        # from the backup location)
        sdata = read_zarr(sdata.path)
        # c. remove the backup copy
        del sdata[new_name]
        sdata.delete_element_from_disk(new_name)

    @pytest.mark.parametrize("dask_backed", [True, False])
    @pytest.mark.parametrize("workaround", [1, 2])
    def test_incremental_io_on_disk(
        self,
        tmp_path: str,
        full_sdata: SpatialData,
        dask_backed: bool,
        workaround: int,
        sdata_container_format: SpatialDataContainerFormatType,
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
        sdata.write(tmpdir, sdata_formats=sdata_container_format)

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
            sdata.write_element(name, sdata_formats=sdata_container_format)
            if dask_backed:
                # this forces the element to write to be dask-backed from disk. In this case, overwriting the data is
                # more laborious because we are writing the data to the same location that defines the data!
                sdata = read_zarr(sdata.path)

            with pytest.raises(
                ValueError,
                match="The Zarr store already exists. Use `overwrite=True` to try overwriting the store.",
            ):
                sdata.write_element(name, sdata_formats=sdata_container_format)

            match = (
                "Details: the target path contains one or more files that Dask use for backing elements in the "
                "SpatialData object"
                if dask_backed
                and name
                in [
                    "image2d",
                    "labels2d",
                    "image3d_multiscale_xarray",
                    "labels3d_multiscale_xarray",
                    "points_0",
                ]
                else "Details: the target path in which to save an element is a subfolder of the current Zarr store."
            )
            with pytest.raises(
                ValueError,
                match=match,
            ):
                sdata.write_element(name, overwrite=True, sdata_formats=sdata_container_format)

            if workaround == 1:
                new_name = f"{name}_new_place"
                # workaround 1, mostly safe (untested for Windows platform, network drives, multi-threaded
                # setups, ...). If the scenario matches your use case, please use with caution.

                if not dask_backed:  # easier case
                    self._workaround1_non_dask_backed(
                        sdata=sdata,
                        name=name,
                        new_name=new_name,
                        sdata_container_format=sdata_container_format,
                    )
                else:  # dask-backed case, more complex
                    self._workaround1_dask_backed(
                        sdata=sdata,
                        name=name,
                        new_name=new_name,
                        sdata_container_format=sdata_container_format,
                    )
            elif workaround == 2:
                # workaround 2, unsafe but sometimes acceptable depending on the user's workflow.

                # this works only if the data is not dask-backed, otherwise an exception will be raised because the code
                # would be trying to delete the data that the Dask object is pointing to!
                if not dask_backed:
                    # a. rewrite the original data (risky!)
                    sdata.delete_element_from_disk(name)
                    sdata.write_element(name, sdata_formats=sdata_container_format)

    def test_io_and_lazy_loading_points(self, points, sdata_container_format: SpatialDataContainerFormatType):
        with tempfile.TemporaryDirectory() as td:
            f = os.path.join(td, "data.zarr")
            points.write(f, sdata_formats=sdata_container_format)
            assert len(get_dask_backing_files(points)) == 0

            sdata2 = SpatialData.read(f)
            assert len(get_dask_backing_files(sdata2)) > 0

    def test_io_and_lazy_loading_raster(self, images, labels, sdata_container_format: SpatialDataContainerFormatType):
        sdatas = {"images": images, "labels": labels}
        for k, sdata in sdatas.items():
            d = getattr(sdata, k)
            elem_name = list(d.keys())[0]
            with tempfile.TemporaryDirectory() as td:
                f = os.path.join(td, "data.zarr")
                dask0 = sdata[elem_name].data
                sdata.write(f, sdata_formats=sdata_container_format)
                assert all("from-zarr" not in key for key in dask0.dask.layers)
                assert len(get_dask_backing_files(sdata)) == 0

                sdata2 = SpatialData.read(f)
                dask1 = sdata2[elem_name].data
                assert any("from-zarr" in key for key in dask1.dask.layers)
                assert len(get_dask_backing_files(sdata2)) > 0

    def test_replace_transformation_on_disk_raster(
        self, images, labels, sdata_container_format: SpatialDataContainerFormatType
    ):
        sdatas = {"images": images, "labels": labels}
        for k, sdata in sdatas.items():
            d = getattr(sdata, k)
            # unlike the non-raster case, we are testing all the elements (2d and 3d, multiscale and not)
            for elem_name in d:
                kwargs = {k: {elem_name: d[elem_name]}}
                single_sdata = SpatialData(**kwargs)
                with tempfile.TemporaryDirectory() as td:
                    f = os.path.join(td, "data.zarr")
                    single_sdata.write(f, sdata_formats=sdata_container_format)
                    t0 = get_transformation(SpatialData.read(f)[elem_name])
                    assert isinstance(t0, Identity)
                    set_transformation(
                        single_sdata[elem_name],
                        Scale([2.0], axes=("x",)),
                        write_to_sdata=single_sdata,
                    )
                    t1 = get_transformation(SpatialData.read(f)[elem_name])
                    assert isinstance(t1, Scale)

    def test_replace_transformation_on_disk_non_raster(
        self, shapes, points, sdata_container_format: SpatialDataContainerFormatType
    ):
        sdatas = {"shapes": shapes, "points": points}
        for k, sdata in sdatas.items():
            d = sdata.__getattribute__(k)
            elem_name = list(d.keys())[0]
            with tempfile.TemporaryDirectory() as td:
                f = os.path.join(td, "data.zarr")
                sdata.write(f, sdata_formats=sdata_container_format)
                t0 = get_transformation(SpatialData.read(f).__getattribute__(k)[elem_name])
                assert isinstance(t0, Identity)
                set_transformation(sdata[elem_name], Scale([2.0], axes=("x",)), write_to_sdata=sdata)
                t1 = get_transformation(SpatialData.read(f)[elem_name])
                assert isinstance(t1, Scale)

    def test_write_overwrite_fails_when_no_zarr_store(
        self, full_sdata, sdata_container_format: SpatialDataContainerFormatType
    ):
        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / "data.zarr"
            f.mkdir()
            old_data = SpatialData()
            with pytest.raises(ValueError, match="The target file path specified already exists"):
                old_data.write(f, sdata_formats=sdata_container_format)
            with pytest.raises(ValueError, match="The target file path specified already exists"):
                full_sdata.write(f, overwrite=True, sdata_formats=sdata_container_format)

    def test_overwrite_fails_when_no_zarr_store_but_dask_backed_data(
        self,
        full_sdata,
        points,
        images,
        labels,
        sdata_container_format: SpatialDataContainerFormatType,
    ):
        sdatas = {"images": images, "labels": labels, "points": points}
        elements = {"images": "image2d", "labels": "labels2d", "points": "points_0"}
        for k, sdata in sdatas.items():
            element = elements[k]
            with tempfile.TemporaryDirectory() as tmpdir:
                f = os.path.join(tmpdir, "data.zarr")
                sdata.write(f, sdata_formats=sdata_container_format)

                # now we have a sdata with dask-backed elements
                sdata2 = SpatialData.read(f)
                p = sdata2[element]
                full_sdata[element] = p
                with pytest.raises(
                    ValueError,
                    match="The Zarr store already exists. Use `overwrite=True` to try overwriting the store.",
                ):
                    full_sdata.write(f, sdata_formats=sdata_container_format)

                with pytest.raises(
                    ValueError,
                    match=r"Details: the target path contains one or more files that Dask use for "
                    "backing elements in the SpatialData object",
                ):
                    full_sdata.write(f, overwrite=True, sdata_formats=sdata_container_format)

    def test_overwrite_fails_when_zarr_store_present(
        self, full_sdata, sdata_container_format: SpatialDataContainerFormatType
    ):
        # addressing https://github.com/scverse/spatialdata/issues/137
        with tempfile.TemporaryDirectory() as tmpdir:
            f = os.path.join(tmpdir, "data.zarr")
            full_sdata.write(f, sdata_formats=sdata_container_format)

            with pytest.raises(
                ValueError,
                match="The Zarr store already exists. Use `overwrite=True` to try overwriting the store.",
            ):
                full_sdata.write(f, sdata_formats=sdata_container_format)

            with pytest.raises(
                ValueError,
                match=r"Details: the target path either contains, coincides or is contained in the current Zarr store",
            ):
                full_sdata.write(f, overwrite=True, sdata_formats=sdata_container_format)

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

    def test_overwrite_fails_onto_non_zarr_file(
        self, full_sdata, sdata_container_format: SpatialDataContainerFormatType
    ):
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
                    full_sdata.write(f0, sdata_formats=sdata_container_format)
                with pytest.raises(
                    ValueError,
                    match=ERROR_MESSAGE,
                ):
                    full_sdata.write(f0, overwrite=True, sdata_formats=sdata_container_format)
            f1 = os.path.join(tmpdir, "test.zarr")
            os.mkdir(f1)
            with pytest.raises(ValueError, match=ERROR_MESSAGE):
                full_sdata.write(f1, sdata_formats=sdata_container_format)
            with pytest.raises(ValueError, match=ERROR_MESSAGE):
                full_sdata.write(f1, overwrite=True, sdata_formats=sdata_container_format)


def test_incremental_io_in_memory(
    full_sdata: SpatialData,
) -> None:
    sdata = full_sdata

    for k, v in _get_images().items():
        sdata.images[f"additional_{k}"] = v
        with pytest.raises(KeyError, match="Key `table` is not unique"):
            sdata["table"] = v

    for k, v in _get_labels().items():
        sdata.labels[f"additional_{k}"] = v
        with pytest.raises(KeyError, match="Key `table` is not unique"):
            sdata["table"] = v

    for k, v in _get_shapes().items():
        sdata.shapes[f"additional_{k}"] = v
        with pytest.raises(KeyError, match="Key `table` is not unique"):
            sdata["table"] = v

    for k, v in _get_points().items():
        sdata.points[f"additional_{k}"] = v
        with pytest.raises(KeyError, match="Key `table` is not unique"):
            sdata["table"] = v

    for k, v in _get_tables(region="labels2d").items():
        sdata.tables[f"additional_{k}"] = v
        with pytest.raises(KeyError, match="Key `poly` is not unique"):
            sdata["poly"] = v


def test_bug_rechunking_after_queried_raster():
    # https://github.com/scverse/spatialdata-io/issues/117
    ##
    single_scale = Image2DModel.parse(RNG.random((100, 10, 10)), chunks=(5, 5, 5))
    multi_scale = Image2DModel.parse(RNG.random((100, 10, 10)), scale_factors=[2, 2], chunks=(5, 5, 5))
    images = {"single_scale": single_scale, "multi_scale": multi_scale}
    sdata = SpatialData(images=images)
    queried = sdata.query.bounding_box(
        axes=("x", "y"),
        min_coordinate=[2, 5],
        max_coordinate=[12, 12],
        target_coordinate_system="global",
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        f = os.path.join(tmpdir, "data.zarr")
        queried.write(f)


@pytest.mark.parametrize("sdata_container_format", SDATA_FORMATS)
def test_self_contained(full_sdata: SpatialData, sdata_container_format: SpatialDataContainerFormatType) -> None:
    # data only in-memory, so the SpatialData object and all its elements are self-contained
    assert full_sdata.is_self_contained()
    description = full_sdata.elements_are_self_contained()
    assert all(description.values())

    with tempfile.TemporaryDirectory() as tmpdir:
        # data saved to disk, it's self contained
        f = os.path.join(tmpdir, "data.zarr")
        full_sdata.write(f, sdata_formats=sdata_container_format)
        full_sdata.is_self_contained()

        # we read the data, so it's self-contained
        sdata2 = SpatialData.read(f)
        assert sdata2.is_self_contained()

        # we save the data to a new location, so it's not self-contained anymore
        f2 = os.path.join(tmpdir, "data2.zarr")
        sdata2.write(f2, sdata_formats=sdata_container_format)
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


@pytest.mark.parametrize("sdata_container_format", SDATA_FORMATS)
def test_symmetric_difference_with_zarr_store(
    full_sdata: SpatialData, sdata_container_format: SpatialDataContainerFormatType
) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        f = os.path.join(tmpdir, "data.zarr")
        full_sdata.write(f, sdata_formats=sdata_container_format)

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


@pytest.mark.parametrize("sdata_container_format", SDATA_FORMATS)
def test_change_path_of_subset(full_sdata: SpatialData, sdata_container_format: SpatialDataContainerFormatType) -> None:
    """A subset SpatialData object has not Zarr path associated, show that we can reassign the path"""
    with tempfile.TemporaryDirectory() as tmpdir:
        f = os.path.join(tmpdir, "data.zarr")
        full_sdata.write(f, sdata_formats=sdata_container_format)

        subset = full_sdata.subset(["image2d", "labels2d", "points_0", "circles", "table"])

        assert subset.path is None
        subset.path = Path(f)

        assert subset.is_self_contained()
        only_in_memory, only_on_disk = subset._symmetric_difference_with_zarr_store()
        assert len(only_in_memory) == 0
        assert len(only_on_disk) > 0

        f2 = os.path.join(tmpdir, "data2.zarr")
        subset.write(f2, sdata_formats=sdata_container_format)
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
    with pytest.raises(ValueError, match="Name cannot be '.'"):
        f(".")
    with pytest.raises(ValueError, match="Name cannot be '..'"):
        f("..")
    with pytest.raises(ValueError, match="Name cannot start with '__'"):
        f("__a")
    with pytest.raises(
        ValueError,
        match="Name must contain only alphanumeric characters, underscores, dots and hyphens.",
    ):
        f("has whitespace")
    with pytest.raises(
        ValueError,
        match="Name must contain only alphanumeric characters, underscores, dots and hyphens.",
    ):
        f("this/is/not/valid")
    with pytest.raises(
        ValueError,
        match="Name must contain only alphanumeric characters, underscores, dots and hyphens.",
    ):
        f("non-alnum_#$%&()*+,?@")


def test_incremental_io_valid_name(full_sdata: SpatialData) -> None:
    _check_valid_name(full_sdata.write_element)
    _check_valid_name(full_sdata.write_metadata)
    _check_valid_name(full_sdata.write_transformations)


@pytest.mark.parametrize("sdata_container_format", SDATA_FORMATS)
def test_incremental_io_attrs(points: SpatialData, sdata_container_format: SpatialDataContainerFormatType) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        f = os.path.join(tmpdir, "data.zarr")
        my_attrs = {"a": "b", "c": 1}
        points.attrs = my_attrs
        points.write(f, sdata_formats=sdata_container_format)

        # test that the attributes are written to disk
        sdata = SpatialData.read(f)
        assert sdata.attrs == my_attrs

        # test incremental io attrs (write_attrs())
        sdata.attrs["c"] = 2
        sdata.write_attrs(sdata_format=sdata_container_format)
        sdata2 = SpatialData.read(f)
        assert sdata2.attrs["c"] == 2

        # test incremental io attrs (write_metadata())
        sdata.attrs["c"] = 3
        sdata.write_metadata(sdata_format=sdata_container_format)
        sdata2 = SpatialData.read(f)
        assert sdata2.attrs["c"] == 3


cached_sdata_blobs = blobs()


@pytest.mark.parametrize("element_name", ["image2d", "labels2d", "points_0", "circles", "table"])
@pytest.mark.parametrize("sdata_container_format", SDATA_FORMATS)
def test_delete_element_from_disk(
    full_sdata,
    element_name: str,
    sdata_container_format: SpatialDataContainerFormatType,
) -> None:
    # can't delete an element for a SpatialData object without associated Zarr store
    with pytest.raises(ValueError, match="The SpatialData object is not backed by a Zarr store."):
        full_sdata.delete_element_from_disk("image2d")

    with tempfile.TemporaryDirectory() as tmpdir:
        f = os.path.join(tmpdir, "data.zarr")
        full_sdata.write(f, sdata_formats=sdata_container_format)

        # cannot delete an element which is in-memory, but not in the Zarr store
        subset = full_sdata.subset(["points_0_1"])
        f2 = os.path.join(tmpdir, "data2.zarr")
        subset.write(f2, sdata_formats=sdata_container_format)
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
        full_sdata.write_element(element_name, sdata_formats=sdata_container_format)

        # now delete it from memory, and then show it can still be deleted on-disk
        del getattr(full_sdata, element_type)[element_name]
        full_sdata.delete_element_from_disk(element_name)
        on_disk = full_sdata.elements_paths_on_disk()
        assert element_path not in on_disk


@pytest.mark.parametrize("element_name", ["image2d", "labels2d", "points_0", "circles", "table"])
@pytest.mark.parametrize("sdata_container_format", SDATA_FORMATS)
def test_element_already_on_disk_different_type(
    full_sdata,
    element_name: str,
    sdata_container_format: SpatialDataContainerFormatType,
) -> None:
    # Constructing a corrupted object (element present both on disk and in-memory but with different type).
    # Attempting to perform and IO operation will trigger an error.
    # The checks assessed in this test will not be needed anymore after
    # https://github.com/scverse/spatialdata/issues/504 is addressed
    with tempfile.TemporaryDirectory() as tmpdir:
        f = os.path.join(tmpdir, "data.zarr")
        full_sdata.write(f, sdata_formats=sdata_container_format)

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
            full_sdata.write_element(element_name, sdata_formats=sdata_container_format)

        with pytest.raises(
            ValueError,
            match=ERROR_MSG,
        ):
            full_sdata.write_metadata(element_name, sdata_format=sdata_container_format)

        with pytest.raises(
            ValueError,
            match=ERROR_MSG,
        ):
            full_sdata.write_transformations(element_name)


def test_writing_invalid_name(tmp_path: Path):
    invalid_sdata = SpatialData()
    # Circumvent validation at construction time and check validation happens again at writing time.
    invalid_sdata.images.data[""] = next(iter(_get_images().values()))
    invalid_sdata.labels.data["."] = next(iter(_get_labels().values()))
    invalid_sdata.points.data["path/separator"] = next(iter(_get_points().values()))
    invalid_sdata.shapes.data["non-alnum_#$%&()*+,?@"] = next(iter(_get_shapes().values()))
    invalid_sdata.tables.data["has whitespace"] = _get_table(region="any")

    with pytest.raises(ValueError, match="Name (must|cannot)"):
        invalid_sdata.write(tmp_path / "data.zarr")


def test_writing_valid_table_name_invalid_table(tmp_path: Path):
    # also try with a valid table name but invalid table
    # testing just one case, all the cases are in test_table_model_invalid_names()
    invalid_sdata = SpatialData()
    invalid_sdata.tables.data["valid_name"] = AnnData(np.array([[0]]), layers={"invalid name": np.array([[0]])})
    with pytest.raises(ValueError, match="Name (must|cannot)"):
        invalid_sdata.write(tmp_path / "data.zarr")


def test_incremental_writing_invalid_name(tmp_path: Path):
    invalid_sdata = SpatialData()
    invalid_sdata.write(tmp_path / "data.zarr")

    # Circumvent validation at construction time and check validation happens again at writing time.
    invalid_sdata.images.data[""] = next(iter(_get_images().values()))
    invalid_sdata.labels.data["."] = next(iter(_get_labels().values()))
    invalid_sdata.points.data["path/separator"] = next(iter(_get_points().values()))
    invalid_sdata.shapes.data["non-alnum_#$%&()*+,?@"] = next(iter(_get_shapes().values()))
    invalid_sdata.tables.data["has whitespace"] = _get_table(region="any")

    for element_type in ["images", "labels", "points", "shapes", "tables"]:
        elements = getattr(invalid_sdata, element_type)
        for name in elements:
            with pytest.raises(ValueError, match="Name (must|cannot)"):
                invalid_sdata.write_element(name)


def test_incremental_writing_valid_table_name_invalid_table(tmp_path: Path):
    # also try with a valid table name but invalid table
    # testing just one case, all the cases are in test_table_model_invalid_names()
    invalid_sdata = SpatialData()
    invalid_sdata.write(tmp_path / "data2.zarr")
    invalid_sdata.tables.data["valid_name"] = AnnData(np.array([[0]]), layers={"invalid name": np.array([[0]])})
    with pytest.raises(ValueError, match="Name (must|cannot)"):
        invalid_sdata.write_element("valid_name")


def test_reading_invalid_name(tmp_path: Path):
    image_name, image = next(iter(_get_images().items()))
    labels_name, labels = next(iter(_get_labels().items()))
    points_name, points = next(iter(_get_points().items()))
    shapes_name, shapes = next(iter(_get_shapes().items()))
    table_name, table = "table", _get_table(region="labels2d")
    valid_sdata = SpatialData(
        images={image_name: image},
        labels={labels_name: labels},
        points={points_name: points},
        shapes={shapes_name: shapes},
        tables={table_name: table},
    )
    valid_sdata.write(tmp_path / "data.zarr")
    # Circumvent validation at construction time and check validation happens again at writing time.
    (tmp_path / "data.zarr/points" / points_name).rename(tmp_path / "data.zarr/points" / "has whitespace")
    # This one is not allowed on windows
    (tmp_path / "data.zarr/shapes" / shapes_name).rename(tmp_path / "data.zarr/shapes" / "non-alnum_#$%&()+,@")
    # We do this as the key of the element is otherwise not in the consolidated metadata, leading to an error.
    valid_sdata.write_consolidated_metadata()

    with pytest.raises(ValidationError, match="Cannot construct SpatialData") as exc_info:
        read_zarr(tmp_path / "data.zarr")

    actual_message = str(exc_info.value)
    assert "points/has whitespace" in actual_message
    assert "shapes/non-alnum_#$%&()+,@" in actual_message
    assert (
        "For renaming, please see the discussion here https://github.com/scverse/spatialdata/discussions/707"
        in actual_message
    )


@pytest.mark.parametrize("sdata_container_format", SDATA_FORMATS)
def test_write_store_unconsolidated_and_read(full_sdata, sdata_container_format: SpatialDataContainerFormatType):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "data.zarr"
        full_sdata.write(path, consolidate_metadata=False, sdata_formats=sdata_container_format)

        group = zarr.open_group(path, mode="r")
        assert group.metadata.consolidated_metadata is None
        second_read = SpatialData.read(path)
        assert_spatial_data_objects_are_identical(full_sdata, second_read)


@pytest.mark.parametrize("sdata_container_format", SDATA_FORMATS)
def test_can_read_sdata_with_reconsolidation(full_sdata, sdata_container_format: SpatialDataContainerFormatType):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "data.zarr"
        full_sdata.write(path, sdata_formats=sdata_container_format)

        if isinstance(sdata_container_format, SpatialDataContainerFormatV01):
            json_path = path / ".zmetadata"
            json_dict = json.loads(json_path.read_text())
            # TODO: this raises no exception!
            del json_dict["metadata"]["images/image2d/.zgroup"]
        else:
            json_path = path / "zarr.json"
            json_dict = json.loads(json_path.read_text())
            del json_dict["consolidated_metadata"]["metadata"]["images/image2d"]
        json_path.write_text(json.dumps(json_dict, indent=4))

        with pytest.raises(GroupNotFoundError):
            SpatialData.read(path)

        new_sdata = SpatialData.read(path, reconsolidate_metadata=True)
        assert_spatial_data_objects_are_identical(full_sdata, new_sdata)


def test_read_sdata(tmp_path: Path, points: SpatialData) -> None:
    sdata_path = tmp_path / "sdata.zarr"
    points.write(sdata_path)

    # path as Path
    sdata_from_path = SpatialData.read(sdata_path)
    assert sdata_from_path.path == sdata_path

    # path as str
    sdata_from_str = SpatialData.read(str(sdata_path))
    assert sdata_from_str.path == sdata_path

    # path as UPath
    sdata_from_upath = SpatialData.read(UPath(sdata_path))
    assert sdata_from_upath.path == sdata_path

    # path as zarr Group
    zarr_group = zarr.open_group(sdata_path, mode="r")
    sdata_from_zarr_group = SpatialData.read(zarr_group)
    assert sdata_from_zarr_group.path == sdata_path

    # Assert all read methods produce identical SpatialData objects
    assert_spatial_data_objects_are_identical(sdata_from_path, sdata_from_str)
    assert_spatial_data_objects_are_identical(sdata_from_path, sdata_from_upath)
    assert_spatial_data_objects_are_identical(sdata_from_path, sdata_from_zarr_group)


def test_sdata_with_nan_in_obs() -> None:
    """Test writing SpatialData with mixed string/NaN values in obs works correctly.

    Regression test for https://github.com/scverse/spatialdata/issues/399
    Previously this raised TypeError: expected unicode string, found nan.
    Now the write succeeds, though NaN values in object-dtype columns are
    converted to the string "nan" after round-trip.
    """
    from spatialdata.models import TableModel

    table = TableModel.parse(
        AnnData(
            obs=pd.DataFrame(
                {
                    "region": ["region1", "region2"],
                    "instance": [0, 0],
                    "column_only_region1": ["string", np.nan],
                    "column_only_region2": [np.nan, 3],
                }
            )
        ),
        region_key="region",
        instance_key="instance",
        region=["region1", "region2"],
    )
    sdata = SpatialData(tables={"table": table})
    assert sdata["table"].obs["column_only_region1"].iloc[1] is np.nan
    assert np.isnan(sdata["table"].obs["column_only_region2"].iloc[0])

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "data.zarr")
        sdata.write(path)

        sdata2 = SpatialData.read(path)
        assert "column_only_region1" in sdata2["table"].obs.columns
        assert sdata2["table"].obs["column_only_region1"].iloc[0] == "string"
        assert sdata2["table"].obs["column_only_region2"].iloc[1] == 3
        # After round-trip, NaN in object-dtype column becomes string "nan"
        assert sdata2["table"].obs["column_only_region1"].iloc[1] == "nan"
        assert np.isnan(sdata2["table"].obs["column_only_region2"].iloc[0])
