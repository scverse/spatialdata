"""Unit tests for remote-storage-specific store resolution and credential handling.

Covers only code paths used when reading/writing from remote backends (Azure, S3, GCS):
- _FsspecStoreRoot resolution (used when reading elements from a remote zarr store).
- _storage_options_from_fs for Azure and GCS (used when writing parquet to remote).
"""

from __future__ import annotations

import tempfile

import pytest
from zarr.storage import FsspecStore, LocalStore, MemoryStore

from spatialdata._io._utils import _FsspecStoreRoot, _resolve_zarr_store, _storage_options_from_fs


def test_resolve_zarr_store_returns_existing_zarr_stores_unchanged() -> None:
    """StoreLike inputs must not be wrapped as FsspecStore(fs=store) — that is only for async filesystems."""
    mem = MemoryStore()
    assert _resolve_zarr_store(mem) is mem
    loc = LocalStore(tempfile.mkdtemp())
    assert _resolve_zarr_store(loc) is loc


def test_resolve_zarr_store_fsspec_store_root() -> None:
    """_FsspecStoreRoot is resolved to FsspecStore when reading from remote (e.g. points/shapes paths)."""
    import fsspec
    from fsspec.implementations.asyn_wrapper import AsyncFileSystemWrapper

    fs = fsspec.filesystem("memory")
    async_fs = AsyncFileSystemWrapper(fs, asynchronous=True)
    base = FsspecStore(async_fs, path="/")
    root = _FsspecStoreRoot(base, "/")
    store = _resolve_zarr_store(root)
    assert isinstance(store, FsspecStore)


def test_storage_options_from_fs_azure_account_key() -> None:
    """_storage_options_from_fs extracts Azure credentials for writing parquet to remote Azure Blob."""

    class AzureBlobFileSystemMock:
        protocol = "abfs"
        account_name = "dev"
        account_key = "key123"
        connection_string = None
        anon = None

    out = _storage_options_from_fs(AzureBlobFileSystemMock())
    assert out["account_name"] == "dev"
    assert out["account_key"] == "key123"


def test_storage_options_from_fs_gcs_endpoint() -> None:
    """_storage_options_from_fs extracts GCS endpoint and project for writing parquet to remote GCS."""

    class GCSFileSystemMock:
        protocol = "gs"
        token = "anon"
        _endpoint = "http://localhost:4443"
        project = "test"

    out = _storage_options_from_fs(GCSFileSystemMock())
    assert out["token"] == "anon"
    assert out["endpoint_url"] == "http://localhost:4443"
    assert out["project"] == "test"


def test_storage_options_from_fs_s3_by_protocol_not_class_name() -> None:
    """Subclasses / wrappers are accepted when ``protocol`` is s3 and attrs match s3fs-style credentials."""

    class CustomS3Wrapper:
        protocol = "s3"
        endpoint_url = "http://127.0.0.1:9000"
        key = "access"
        secret = "secret"
        anon = False

    out = _storage_options_from_fs(CustomS3Wrapper())
    assert out["endpoint_url"] == "http://127.0.0.1:9000"
    assert out["key"] == "access"
    assert out["secret"] == "secret"
    assert out["anon"] is False


def test_storage_options_from_fs_requires_object_store_protocol() -> None:
    class NoCloud:
        protocol = "file"

    with pytest.raises(ValueError, match="Cannot derive parquet storage_options"):
        _storage_options_from_fs(NoCloud())
