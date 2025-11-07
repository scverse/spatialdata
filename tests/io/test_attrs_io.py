"""Test attrs read/write for all SpatialData container formats."""

import tempfile
from pathlib import Path

import pytest

from spatialdata import SpatialData, read_zarr
from spatialdata._io.format import (
    SpatialDataContainerFormats,
    SpatialDataContainerFormatType,
)

FORMAT_V01 = SpatialDataContainerFormats["0.1"]
FORMAT_V02 = SpatialDataContainerFormats["0.2"]


@pytest.mark.parametrize("sdata_container_format", [FORMAT_V01, FORMAT_V02])
class TestAttrsIO:
    """Test SpatialData.attrs read/write for all container formats."""

    def test_attrs_write_and_read(
        self,
        sdata_container_format: SpatialDataContainerFormatType,
    ) -> None:
        """Test attrs with complex nested structures."""
        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / "data.zarr"

            # Set complex attrs
            my_attrs = {
                "string_key": "test_value",
                "int_key": 123,
                "float_key": 3.14,
                "bool_key": True,
                "list_key": [1, 2, 3],
                "nested_dict": {
                    "inner_key1": None,
                    "inner_key2": 456,
                    "nested_list": ["a", "b", "c"],
                },
            }
            sdata = SpatialData()
            sdata.attrs = my_attrs

            # Write to disk with sdata_container_format
            sdata.write(f, sdata_formats=sdata_container_format)

            # Read back and verify attrs
            sdata_read = read_zarr(f)
            assert sdata_read.attrs == my_attrs

    def test_attrs_incremental_write(
        self,
        sdata_container_format: SpatialDataContainerFormatType,
    ) -> None:
        """Test incremental write of attrs using write_attrs() and write_metadata()."""
        with tempfile.TemporaryDirectory() as tmpdir:
            f = Path(tmpdir) / "data.zarr"

            # Initial write with attrs using sdata_container_format
            initial_attrs = {"initial_key": "initial_value", "counter": 0}
            sdata = SpatialData()
            sdata.attrs = initial_attrs
            sdata.write(f, sdata_formats=sdata_container_format)

            # Verify initial attrs
            sdata_read1 = read_zarr(f)
            assert sdata_read1.attrs == initial_attrs

            # Update attrs using write_attrs()
            sdata_read1.attrs["counter"] = 1
            sdata_read1.attrs["new_key"] = "new_value"
            sdata_read1.write_attrs(sdata_format=sdata_container_format)

            # Read back and verify updated attrs
            sdata_read2 = read_zarr(f)
            assert sdata_read2.attrs["counter"] == 1
            assert sdata_read2.attrs["new_key"] == "new_value"
            assert sdata_read2.attrs["initial_key"] == "initial_value"

            # Update attrs using write_metadata()
            sdata_read2.attrs["counter"] = 2
            sdata_read2.attrs["another_key"] = "another_value"
            sdata_read2.write_metadata(sdata_format=sdata_container_format)

            # Read back and verify all attrs
            sdata_read3 = read_zarr(f)
            assert sdata_read3.attrs["counter"] == 2
            assert sdata_read3.attrs["new_key"] == "new_value"
            assert sdata_read3.attrs["another_key"] == "another_value"
            assert sdata_read3.attrs["initial_key"] == "initial_value"


def test_attrs_v1_to_v2() -> None:
    """Test that attrs are preserved when converting from V01 to V02."""
    with tempfile.TemporaryDirectory() as tmpdir:
        f_v1 = Path(tmpdir) / "data_v1.zarr"
        f_v2 = Path(tmpdir) / "data_v2.zarr"

        # Set attrs and write with V01
        my_attrs = {
            "test_key": "test_value",
            "counter": 1,
            "nested": {"inner": "value"},
        }
        sdata = SpatialData()
        sdata.attrs = my_attrs
        sdata.write(f_v1, sdata_formats=FORMAT_V01)

        # Read with V01
        sdata_v1 = read_zarr(f_v1)
        assert sdata_v1.attrs == my_attrs

        # Write with V02
        sdata_v1.write(f_v2, sdata_formats=FORMAT_V02)

        # Read with V02 and verify attrs are preserved
        sdata_v2 = read_zarr(f_v2)
        assert sdata_v2.attrs == my_attrs
