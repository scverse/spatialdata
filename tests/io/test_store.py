from __future__ import annotations

from pathlib import Path

import zarr
from upath import UPath

from spatialdata._store import (
    make_zarr_store,
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


def test_open_read_and_write_store_roundtrip(tmp_path: Path) -> None:
    zarr_store = make_zarr_store(tmp_path / "store.zarr")

    with open_write_store(zarr_store) as store:
        group = zarr.create_group(store=store, overwrite=True)
        group.attrs["answer"] = 42

    with open_read_store(zarr_store) as store:
        group = zarr.open_group(store=store, mode="r")
        assert group.attrs["answer"] == 42
