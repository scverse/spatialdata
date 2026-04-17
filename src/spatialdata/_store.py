from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, TypeAlias

import pyarrow.fs as pafs
import zarr
from upath import UPath
from zarr.storage import FsspecStore, LocalStore

PathLike: TypeAlias = Path | UPath


def normalize_path(path: str | PathLike, storage_options: dict[str, Any] | None = None) -> PathLike:
    if isinstance(path, str):
        return UPath(path, **(storage_options or {})) if "://" in path else Path(path)
    if isinstance(path, (Path, UPath)):
        return path
    raise TypeError("Path must be `None`, a `str`, a `Path` or a `UPath` object.")


@dataclass(frozen=True)
class ZarrStore:
    path: PathLike

    def with_path(self, path: str | PathLike) -> ZarrStore:
        return replace(self, path=normalize_path(path))

    def child(self, path: str | PathLike) -> ZarrStore:
        return self.with_path(self.path / path)

    def arrow_path(self) -> str:
        return self.path.path if isinstance(self.path, UPath) else str(self.path)

    def arrow_filesystem(self) -> pafs.FileSystem:
        if isinstance(self.path, UPath):
            return pafs.PyFileSystem(pafs.FSSpecHandler(self.path.fs))
        return pafs.LocalFileSystem()


def make_zarr_store(
    path: str | PathLike,
    *,
    storage_options: dict[str, Any] | None = None,
) -> ZarrStore:
    return ZarrStore(path=normalize_path(path, storage_options))


def make_zarr_store_from_group(group: zarr.Group) -> ZarrStore:
    from spatialdata._io._utils import join_fsspec_store_path

    # zarr v3 does not wrap stores with a ``ConsolidatedMetadataStore`` (that was a v2-only
    # concept); consolidated metadata is now a field on ``GroupMetadata``. So the group's
    # ``.store`` is already the concrete backend store -- no unwrapping required.
    store = group.store

    if isinstance(store, LocalStore):
        return make_zarr_store(Path(store.root) / group.path)
    if isinstance(store, FsspecStore):
        protocol = getattr(store.fs, "protocol", None)
        if isinstance(protocol, (list, tuple)):
            protocol = protocol[0] if protocol else "file"
        elif protocol is None:
            protocol = "file"
        # zarr's FsspecStore wraps sync filesystems via ``AsyncFileSystemWrapper`` (exposed on
        # ``.sync_fs``). Walk that chain to recover the original sync fs so we can hand it back
        # to UPath, which expects a sync filesystem.
        fs = store.fs
        while True:
            inner = getattr(fs, "sync_fs", None)
            if inner is None or inner is fs:
                break
            fs = inner
        path = join_fsspec_store_path(store.path, group.path)
        return make_zarr_store(UPath(f"{protocol}://{path}", fs=fs))
    raise ValueError(f"Unsupported store type or zarr.Group: {type(group.store)}")


@contextmanager
def open_read_store(store: ZarrStore) -> Any:
    """Open ``store`` as a read-only backend store.

    The resolved zarr store is constructed with ``read_only=True`` so that the underlying
    ``LocalStore`` / ``FsspecStore`` refuses writes at the store layer (not just at the group's
    ``mode="r"`` level). This also lets remote read-only backends (e.g. public HTTPS zarrs)
    skip any write-capability probe that fsspec may otherwise perform.
    """
    from spatialdata._io._utils import _resolve_zarr_store

    resolved_store = _resolve_zarr_store(store.path, read_only=True)
    try:
        yield resolved_store
    finally:
        resolved_store.close()


@contextmanager
def open_write_store(store: ZarrStore) -> Any:
    """Open ``store`` as a writable backend store (``read_only=False``)."""
    from spatialdata._io._utils import _resolve_zarr_store

    resolved_store = _resolve_zarr_store(store.path, read_only=False)
    try:
        yield resolved_store
    finally:
        resolved_store.close()
