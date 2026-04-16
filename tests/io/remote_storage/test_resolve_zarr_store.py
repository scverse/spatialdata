"""Unit tests for remote store resolution helpers.

These cover generic code paths used when reading/writing through remote backends:
- zarr.Group to ZarrStore normalization for remote-backed groups.
"""

from __future__ import annotations

import tempfile

import zarr
from zarr.storage import FsspecStore, LocalStore, MemoryStore

from spatialdata._io._utils import _resolve_zarr_store
from spatialdata._store import make_zarr_store_from_group, open_read_store


def test_resolve_zarr_store_returns_existing_zarr_stores_unchanged() -> None:
    """StoreLike inputs must not be wrapped as FsspecStore(fs=store) — that is only for async filesystems."""
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
