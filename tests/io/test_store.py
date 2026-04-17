from __future__ import annotations

import tempfile
from pathlib import Path

import zarr
from upath import UPath
from zarr.storage import FsspecStore, LocalStore, MemoryStore

from spatialdata._io._utils import _resolve_zarr_store
from spatialdata._store import (
    make_zarr_store,
    make_zarr_store_from_group,
    open_read_store,
    open_write_store,
)


def test_make_zarr_store_normalizes_local_and_remote_paths(
    tmp_path: Path,
) -> None:
    local_store = make_zarr_store(str(tmp_path / "store.zarr"))
    assert isinstance(local_store.path, Path)

    remote_store = make_zarr_store("s3://bucket/store.zarr")
    assert isinstance(remote_store.path, UPath)


def test_make_zarr_store_applies_storage_options_to_remote_strings() -> None:
    zarr_store = make_zarr_store("s3://bucket/store.zarr", storage_options={"anon": True})
    assert isinstance(zarr_store.path, UPath)
    assert getattr(zarr_store.path.fs, "anon", None) is True


def test_open_read_and_write_store_roundtrip(tmp_path: Path) -> None:
    zarr_store = make_zarr_store(tmp_path / "store.zarr")

    with open_write_store(zarr_store) as store:
        group = zarr.create_group(store=store, overwrite=True)
        group.attrs["answer"] = 42

    with open_read_store(zarr_store) as store:
        group = zarr.open_group(store=store, mode="r")
        assert group.attrs["answer"] == 42


def test_make_zarr_store_from_local_group(tmp_path: Path) -> None:
    zarr_store = make_zarr_store(tmp_path / "store.zarr")

    with open_write_store(zarr_store) as store:
        root = zarr.create_group(store=store, overwrite=True)
        group = root.require_group("images").require_group("image")

    child_store = make_zarr_store_from_group(group)
    assert child_store.path == tmp_path / "store.zarr" / "images" / "image"


def test_resolve_zarr_store_returns_existing_zarr_stores_unchanged() -> None:
    """StoreLike inputs must not be wrapped as FsspecStore(fs=store) -- that is only for async filesystems."""
    mem = MemoryStore()
    assert _resolve_zarr_store(mem) is mem
    loc = LocalStore(tempfile.mkdtemp())
    assert _resolve_zarr_store(loc) is loc


def test_make_zarr_store_from_remote_group() -> None:
    """Remote zarr.Group inputs keep a usable UPath and reopen through the same protocol."""
    import fsspec
    from fsspec.implementations.asyn_wrapper import AsyncFileSystemWrapper

    fs = fsspec.filesystem("memory")
    async_fs = AsyncFileSystemWrapper(fs, asynchronous=True)
    base = FsspecStore(async_fs, path="/")
    root = zarr.open_group(store=base, mode="a")
    group = root.require_group("points").require_group("points")

    zarr_store = make_zarr_store_from_group(group)
    assert getattr(zarr_store.path.fs, "protocol", None) == "memory"

    with open_read_store(zarr_store) as store:
        assert isinstance(store, FsspecStore)
