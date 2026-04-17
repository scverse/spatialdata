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
        # Recover the original SYNC filesystem from ``store.fs``. zarr v3's FsspecStore requires
        # an async fs, so when callers pass a sync fs (e.g. ``MemoryFileSystem``) we wrap it via
        # ``AsyncFileSystemWrapper``, which preserves the original on ``.sync_fs``. We must
        # unwrap here because the resulting UPath flows into ``ZarrStore.arrow_filesystem()``,
        # i.e. ``pafs.FSSpecHandler(fs)`` -- and pyarrow's handler is strictly sync. Feeding it
        # an async-wrapped fs raises ``RuntimeError: Loop is not running`` at read/write time.
        # The ``while`` loop tolerates (hypothetical) multi-layer wrapping across zarr versions.
        #
        # TODO(async-pyarrow-fs): drop this unwrap once either (a) pyarrow's FSSpecHandler learns
        # to run an async fs under its own event loop, or (b) zarr exposes the original sync fs
        # on FsspecStore without the AsyncFileSystemWrapper indirection (tracked at
        # https://github.com/zarr-developers/zarr-python/issues/2073). At that point ``fs`` can be
        # assigned directly from ``store.fs`` and the getattr probe can go.
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


def open_zarr_for_read(store: Any, *, as_group: bool = True) -> Any:
    """Open a zarr group or node for reading with remote-friendly defaults.

    Prefers the fast path: pinned ``zarr_format=3`` (we only ever write v3 stores, so skipping
    v2-metadata auto-probes saves up to five small GETs per open on remote backends) and
    ``use_consolidated=True`` (requires the root / element ``zarr.json`` to carry the
    ``consolidated_metadata`` field produced by ``_write_consolidated_metadata``). Falls back
    to ``zarr.open*`` with no format/consolidation hints for legacy or third-party stores that
    predate either convention.

    Parameters
    ----------
    store
        A ``zarr.storage.StoreLike`` -- typically the value yielded by ``open_read_store``.
    as_group
        If ``True`` (default) use ``zarr.open_group``; if ``False`` use ``zarr.open`` which
        returns either a ``Group`` or an ``Array`` based on the metadata at the store root.
    """
    fn = zarr.open_group if as_group else zarr.open
    try:
        return fn(store, mode="r", zarr_format=3, use_consolidated=True)
    except (ValueError, FileNotFoundError):
        return fn(store, mode="r")


@contextmanager
def open_write_store(store: ZarrStore) -> Any:
    """Open ``store`` as a writable backend store (``read_only=False``)."""
    from spatialdata._io._utils import _resolve_zarr_store

    resolved_store = _resolve_zarr_store(store.path, read_only=False)
    try:
        yield resolved_store
    finally:
        resolved_store.close()
