from __future__ import annotations

import tempfile
from pathlib import Path

import zarr
from upath import UPath
from zarr.storage import FsspecStore, LocalStore, MemoryStore

from spatialdata._io._utils import _resolve_zarr_store
from spatialdata._store import (
    normalize_path,
    open_read_store,
    open_write_store,
    parquet_fs_and_path,
    path_from_store,
    store_from_group,
)


def test_normalize_path_local_string(tmp_path: Path) -> None:
    result = normalize_path(str(tmp_path / "store.zarr"))
    assert isinstance(result, Path)


def test_normalize_path_remote_string() -> None:
    result = normalize_path("s3://bucket/store.zarr")
    assert isinstance(result, UPath)


def test_normalize_path_storage_options() -> None:
    result = normalize_path("s3://bucket/store.zarr", storage_options={"anon": True})
    assert isinstance(result, UPath)
    assert getattr(result.fs, "anon", None) is True


def test_normalize_path_passthrough_path(tmp_path: Path) -> None:
    p = tmp_path / "store.zarr"
    assert normalize_path(p) is p


def test_normalize_path_passthrough_upath() -> None:
    u = UPath("s3://bucket/store.zarr")
    assert normalize_path(u) is u


def test_open_read_and_write_store_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "store.zarr"

    with open_write_store(path) as store:
        group = zarr.create_group(store=store, overwrite=True)
        group.attrs["answer"] = 42

    with open_read_store(path) as store:
        group = zarr.open_group(store=store, mode="r")
        assert group.attrs["answer"] == 42


def test_path_from_store_local(tmp_path: Path) -> None:
    """Path derivation from a LocalStore returns the on-disk root as a Path."""
    path = tmp_path / "store.zarr"

    with open_write_store(path) as store:
        zarr.create_group(store=store, overwrite=True)
        assert path_from_store(store) == path


def test_store_from_group_local_subroots(tmp_path: Path) -> None:
    """`store_from_group` returns a LocalStore re-rooted at the group's path."""
    path = tmp_path / "store.zarr"

    with open_write_store(path) as store:
        root = zarr.create_group(store=store, overwrite=True)
        group = root.require_group("images").require_group("image")

        sub = store_from_group(group, read_only=True)
        assert isinstance(sub, LocalStore)
        assert Path(sub.root) == path / "images" / "image"
        assert sub.read_only is True


def test_parquet_fs_and_path_local(tmp_path: Path) -> None:
    """`parquet_fs_and_path` returns an fsspec LocalFileSystem and a joined local path string."""
    from fsspec.implementations.local import LocalFileSystem

    path = tmp_path / "store.zarr"
    with open_write_store(path) as store:
        root = zarr.create_group(store=store, overwrite=True)
        group = root.require_group("points").require_group("p1")

        fs, parquet_path = parquet_fs_and_path(group, "points.parquet")
        assert isinstance(fs, LocalFileSystem)
        assert parquet_path == str(path / "points" / "p1" / "points.parquet")


def test_resolve_zarr_store_returns_existing_zarr_stores_unchanged() -> None:
    """StoreLike inputs must not be wrapped as FsspecStore(fs=store) -- that is only for async filesystems."""
    mem = MemoryStore()
    assert _resolve_zarr_store(mem) is mem
    loc = LocalStore(tempfile.mkdtemp())
    assert _resolve_zarr_store(loc) is loc


def test_resolve_zarr_store_forwards_read_only_local(tmp_path: Path) -> None:
    """`_resolve_zarr_store(..., read_only=True)` must reach the LocalStore constructor."""
    store = _resolve_zarr_store(tmp_path / "store.zarr", read_only=True)
    assert isinstance(store, LocalStore)
    assert store.read_only is True


def test_resolve_zarr_store_forwards_read_only_remote() -> None:
    """`_resolve_zarr_store(..., read_only=True)` must reach the FsspecStore constructor."""
    from fsspec.implementations.memory import MemoryFileSystem

    upath = UPath("memory://ro-remote.zarr", fs=MemoryFileSystem(skip_instance_cache=True))
    store = _resolve_zarr_store(upath, read_only=True)
    assert isinstance(store, FsspecStore)
    assert store.read_only is True


def test_path_from_store_remote() -> None:
    """`path_from_store` on a remote FsspecStore yields a UPath with the original sync fs."""
    import fsspec
    from fsspec.implementations.asyn_wrapper import AsyncFileSystemWrapper

    fs = fsspec.filesystem("memory")
    async_fs = AsyncFileSystemWrapper(fs, asynchronous=True)
    store = FsspecStore(async_fs, path="/some/remote.zarr")

    result = path_from_store(store)
    assert isinstance(result, UPath)
    assert getattr(result.fs, "protocol", None) == "memory"


def test_path_from_store_memory_returns_none() -> None:
    """Stores without a meaningful filesystem path (MemoryStore, custom) return None."""
    assert path_from_store(MemoryStore()) is None
