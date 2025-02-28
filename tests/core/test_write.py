import os
import shutil
import tempfile
from pathlib import Path

import pytest
import zarr


@pytest.fixture
def temp_zarr_path():
    """Create a temporary directory for testing, clean up afterwards."""
    temp_dir = tempfile.mkdtemp()
    try:
        zarr_path = Path(temp_dir) / "test.zarr"
        yield zarr_path
    finally:
        shutil.rmtree(temp_dir)


def test_write_encoding_metadata(full_sdata, temp_zarr_path):
    """Test that encoding metadata is written to the root group."""
    # Write the SpatialData object
    full_sdata.write(temp_zarr_path)

    # Open the Zarr store and check for encoding metadata
    store = zarr.open(temp_zarr_path)
    assert "encoding-type" in store.attrs
    assert store.attrs["encoding-type"] == "spatialdata"
    assert "encoding-version" in store.attrs
    assert store.attrs["encoding-version"] == "0.1"


def test_write_read_roundtrip(full_sdata, temp_zarr_path):
    """Test writing and reading back preserves data and metadata."""
    full_sdata.write(temp_zarr_path)

    from spatialdata import SpatialData

    read_sdata = SpatialData.read(temp_zarr_path)

    # Check that encoding metadata is preserved
    store = zarr.open(temp_zarr_path)
    assert "encoding-type" in store.attrs
    assert "encoding-version" in store.attrs

    # Check that element counts match
    assert len(read_sdata.images) == len(full_sdata.images)
    assert len(read_sdata.labels) == len(full_sdata.labels)
    assert len(read_sdata.shapes) == len(full_sdata.shapes)
    assert len(read_sdata.points) == len(full_sdata.points)
    assert len(read_sdata.tables) == len(full_sdata.tables)


def test_write_consolidated_metadata(full_sdata):
    """Test that consolidated metadata includes encoding information."""
    temp_dir = tempfile.mkdtemp()
    try:
        zarr_path = Path(temp_dir) / "test_consolidated.zarr"

        # Write with consolidated metadata
        full_sdata.write(zarr_path, consolidate_metadata=True)

        # Check for both possible metadata filenames
        # (.zmetadata is the standard, zmetadata is referenced in the comments)
        standard_path = os.path.join(zarr_path, ".zmetadata")
        alt_path = os.path.join(zarr_path, "zmetadata")

        assert os.path.exists(standard_path) or os.path.exists(alt_path), "Neither .zmetadata nor zmetadata file found"

        # Check encoding metadata in the store
        store = zarr.open(zarr_path)
        assert "encoding-type" in store.attrs
        assert store.attrs["encoding-type"] == "spatialdata"
        assert "encoding-version" in store.attrs
    finally:
        shutil.rmtree(temp_dir)


def test_write_empty_sdata(temp_zarr_path):
    """Test writing an empty SpatialData object."""
    # Create an empty SpatialData object
    from spatialdata import SpatialData

    empty_sdata = SpatialData()

    # Write the empty SpatialData object
    empty_sdata.write(temp_zarr_path)

    # Open the Zarr store and check for encoding metadata
    store = zarr.open(temp_zarr_path)
    assert "encoding-type" in store.attrs
    assert store.attrs["encoding-type"] == "spatialdata"
    assert "encoding-version" in store.attrs
