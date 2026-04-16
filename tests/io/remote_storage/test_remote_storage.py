"""Integration tests for remote-backed storage using real backend emulators.

Emulators must be running (e.g. Docker: docker run -p 5000:5000 -p 10000:10000 -p 4443:4443 spatialdata-emulators).
Ports: S3/moto 5000, Azure/Azurite 10000, GCS/fake-gcs-server 4443.
tests/io/remote_storage/conftest.py creates buckets/containers when emulators are up.

All remote paths use uuid.uuid4().hex so each test run writes to a unique location.
"""

from __future__ import annotations

import os
import uuid

import pytest
import zarr
from upath import UPath

from spatialdata import SpatialData
from spatialdata._store import make_zarr_store, open_read_store
from spatialdata.testing import assert_spatial_data_objects_are_identical

# Azure emulator connection string (Azurite default).
# https://learn.microsoft.com/en-us/azure/storage/common/storage-configure-connection-string
AZURE_CONNECTION_STRING = (
    "DefaultEndpointsProtocol=http;"
    "AccountName=devstoreaccount1;"
    "AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;"
    "BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;"
)


def _get_azure_upath(container: str = "test-container", path: str = "test.zarr") -> UPath:
    """Create Azure UPath for testing with Azurite (local emulator)."""
    return UPath(f"az://{container}/{path}", connection_string=AZURE_CONNECTION_STRING)


def _get_s3_upath(container: str = "bucket", path: str = "test.zarr") -> UPath:
    """Create S3 UPath for testing (moto emulator at 5000)."""
    endpoint = os.environ.get("AWS_ENDPOINT_URL", "http://127.0.0.1:5000")
    if endpoint:
        return UPath(
            f"s3://{container}/{path}",
            endpoint_url=endpoint,
            key=os.environ.get("AWS_ACCESS_KEY_ID", "testing"),
            secret=os.environ.get("AWS_SECRET_ACCESS_KEY", "testing"),
        )
    return UPath(f"s3://{container}/{path}", anon=True)


def _get_gcs_upath(container: str = "bucket", path: str = "test.zarr") -> UPath:
    """Create GCS UPath for testing with fake-gcs-server (port 4443)."""
    os.environ.setdefault("STORAGE_EMULATOR_HOST", "http://127.0.0.1:4443")
    return UPath(
        f"gs://{container}/{path}",
        endpoint_url=os.environ["STORAGE_EMULATOR_HOST"],
        token="anon",
        project="test",
    )


GET_UPATH_PARAMS = pytest.mark.parametrize(
    "get_upath", [_get_azure_upath, _get_s3_upath, _get_gcs_upath], ids=["azure", "s3", "gcs"]
)
REMOTE_STORAGE_PARAMS = pytest.mark.parametrize(
    "get_upath,storage_name",
    [(_get_azure_upath, "azure"), (_get_s3_upath, "s3"), (_get_gcs_upath, "gcs")],
    ids=["azure", "s3", "gcs"],
)

# Ensure buckets/containers exist on emulators before any test (see tests/io/remote_storage/conftest.py).
pytestmark = pytest.mark.usefixtures("_remote_storage_buckets_containers")


def _assert_read_identical(expected: SpatialData, upath: UPath, *, check_path: bool = True) -> None:
    """Read SpatialData from upath and assert it equals expected; optionally assert path."""
    sdata_read = SpatialData.read(upath)
    if check_path:
        assert isinstance(sdata_read.path, UPath)
        assert sdata_read.path == upath
    assert_spatial_data_objects_are_identical(expected, sdata_read)


class TestPathSetter:
    """Test SpatialData.path setter with remote UPath objects."""

    @GET_UPATH_PARAMS
    def test_path_setter_accepts_upath(self, get_upath) -> None:
        """Test that SpatialData.path setter accepts backend-configured UPath objects.

        This test fails, reproducing issue #441: SpatialData.path setter only accepts
        None | str | Path, not UPath, preventing the use of remote storage.
        """
        sdata = SpatialData()
        upath = get_upath(path=f"test-accept-{uuid.uuid4().hex}.zarr")
        sdata.path = upath
        assert sdata.path == upath

    @GET_UPATH_PARAMS
    def test_write_with_upath_sets_path(self, get_upath) -> None:
        """Test that writing to a remote UPath sets SpatialData.path correctly.

        This test fails because SpatialData.write() rejects UPath in
        _validate_can_safely_write_to_path() before it can set sdata.path.
        """
        sdata = SpatialData()
        upath = get_upath(path=f"test-write-path-{uuid.uuid4().hex}.zarr")
        sdata.write(upath)
        assert isinstance(sdata.path, UPath)

    def test_path_setter_rejects_other_types(self) -> None:
        """Test that SpatialData.path setter rejects other types."""
        sdata = SpatialData()
        with pytest.raises(TypeError, match="Path must be.*str.*Path"):
            sdata.path = 123
        with pytest.raises(TypeError, match="Path must be.*str.*Path"):
            sdata.path = {"not": "a path"}


class TestRemoteStorage:
    """Test end-to-end remote storage workflows with backend-configured UPath objects.

    Note: These tests require the backend emulators from ``tests/io/remote_storage/conftest.py``
    to be running. Tests will fail if the emulators are not available.
    """

    @REMOTE_STORAGE_PARAMS
    def test_write_read_roundtrip_remote(self, full_sdata: SpatialData, get_upath, storage_name: str) -> None:
        """Test writing and reading SpatialData to/from remote storage.

        This test verifies the full workflow:
        1. Write SpatialData to remote storage using UPath
        2. Read SpatialData from remote storage using UPath
        3. Verify data integrity (round-trip)
        """
        upath = get_upath(container=f"test-{storage_name}", path=f"roundtrip-{uuid.uuid4().hex}.zarr")
        full_sdata.write(upath, overwrite=True)
        assert isinstance(full_sdata.path, UPath)
        assert full_sdata.path == upath
        _assert_read_identical(full_sdata, upath)
        # ``str(upath)`` drops the configured filesystem object. Some backends can still be reopened
        # from ambient environment defaults, but others rely on the configured UPath, so we only
        # assert the string-URL read path for S3 here.
        if storage_name == "s3":
            sdata_str_url = SpatialData.read(str(upath))
            assert isinstance(sdata_str_url.path, UPath)
            assert_spatial_data_objects_are_identical(full_sdata, sdata_str_url)

    @REMOTE_STORAGE_PARAMS
    def test_path_setter_with_remote_then_operations(
        self, full_sdata: SpatialData, get_upath, storage_name: str
    ) -> None:
        """Test setting a remote path, then performing operations.

        This test verifies that after setting a remote path:
        1. Path is correctly stored
        2. Write operations work
        3. Read operations work
        """
        upath = get_upath(container=f"test-{storage_name}", path=f"operations-{uuid.uuid4().hex}.zarr")
        full_sdata.path = upath
        assert full_sdata.path == upath
        assert full_sdata.is_backed() is True
        full_sdata.write(overwrite=True)
        assert full_sdata.path == upath
        _assert_read_identical(full_sdata, upath)

    @REMOTE_STORAGE_PARAMS
    def test_overwrite_existing_remote_data(self, full_sdata: SpatialData, get_upath, storage_name: str) -> None:
        """Test overwriting existing data in remote storage.

        Verifies that backend-managed overwriting works and that the data remains
        intact afterwards. Round-trip is covered by ``test_write_read_roundtrip_remote``.
        """
        upath = get_upath(container=f"test-{storage_name}", path=f"overwrite-{uuid.uuid4().hex}.zarr")
        full_sdata.write(upath, overwrite=True)
        full_sdata.write(upath, overwrite=True)
        _assert_read_identical(full_sdata, upath, check_path=False)

    @REMOTE_STORAGE_PARAMS
    def test_write_element_to_remote_storage(self, full_sdata: SpatialData, get_upath, storage_name: str) -> None:
        """Test writing individual elements to remote storage using ``write_element()``.

        This test verifies that:
        1. Setting path to remote UPath works
        2. write_element() works with remote storage
        3. Written elements can be read back correctly
        """
        upath = get_upath(container=f"test-{storage_name}", path=f"write-element-{uuid.uuid4().hex}.zarr")
        # Create empty SpatialData and write to remote storage
        empty_sdata = SpatialData()
        empty_sdata.write(upath, overwrite=True)
        full_sdata.path = upath
        assert full_sdata.path == upath
        # Write each element type individually
        for _element_type, element_name, _ in full_sdata.gen_elements():
            full_sdata.write_element(element_name, overwrite=True)
        _assert_read_identical(full_sdata, upath, check_path=False)

    @REMOTE_STORAGE_PARAMS
    def test_read_from_remote_zarr_group_keeps_backing_for_followup_write(
        self, full_sdata: SpatialData, get_upath, storage_name: str
    ) -> None:
        """Test that reading from a remote zarr.Group preserves enough backing info for a later write."""
        upath = get_upath(container=f"test-{storage_name}", path=f"read-group-{uuid.uuid4().hex}.zarr")
        full_sdata.write(upath, overwrite=True)

        with open_read_store(make_zarr_store(upath)) as store:
            group = zarr.open_group(store=store, mode="r")
            sdata_from_group = SpatialData.read(group)

        assert isinstance(sdata_from_group.path, UPath)
        sdata_from_group.write(overwrite=True)
        _assert_read_identical(full_sdata, upath, check_path=False)
