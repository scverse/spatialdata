from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypeAlias

import pyarrow.fs as pafs
import zarr
from upath import UPath
from zarr.storage import FsspecStore, LocalStore

PathLike: TypeAlias = Path | UPath


def normalize_path(path: str | PathLike, storage_options: dict[str, Any] | None = None) -> PathLike:
    """Normalize a path-like input to ``Path`` (local) or ``UPath`` (remote)."""
    if isinstance(path, str):
        return UPath(path, **(storage_options or {})) if "://" in path else Path(path)
    if isinstance(path, (Path, UPath)):
        return path
    raise TypeError("Path must be a `str`, `Path`, or `UPath` object.")


def path_from_group(group: zarr.Group) -> PathLike:
    """Derive a ``PathLike`` from an open ``zarr.Group``, including its subgroup path.

    Handles both ``LocalStore`` and ``FsspecStore`` backends. For fsspec stores,
    unwraps any ``AsyncFileSystemWrapper`` to recover the original sync filesystem
    so that the returned ``UPath`` can be passed to pyarrow's ``FSSpecHandler``
    (which is strictly synchronous).

    TODO(async-pyarrow-fs): drop the sync-unwrap once pyarrow's FSSpecHandler learns
    to drive an async fs, or zarr exposes the original sync fs directly on FsspecStore
    (tracked at https://github.com/zarr-developers/zarr-python/issues/2073).
    """
    from spatialdata._io._utils import join_fsspec_store_path

    store = group.store

    if isinstance(store, LocalStore):
        return Path(store.root) / group.path

    if isinstance(store, FsspecStore):
        protocol = getattr(store.fs, "protocol", None)
        if isinstance(protocol, (list, tuple)):
            protocol = protocol[0] if protocol else "file"
        elif protocol is None:
            protocol = "file"
        fs = store.fs
        while True:
            inner = getattr(fs, "sync_fs", None)
            if inner is None or inner is fs:
                break
            fs = inner
        path = join_fsspec_store_path(store.path, group.path)
        return UPath(f"{protocol}://{path}", fs=fs)

    raise ValueError(f"Unsupported store type for zarr.Group: {type(group.store)}")


def arrow_path(path: PathLike) -> str:
    """Return the raw path string suitable for a pyarrow filesystem call."""
    return path.path if isinstance(path, UPath) else str(path)


def arrow_filesystem(path: PathLike) -> pafs.FileSystem:
    """Return a pyarrow filesystem matching the fsspec backend embedded in ``path``."""
    if isinstance(path, UPath):
        return pafs.PyFileSystem(pafs.FSSpecHandler(path.fs))
    return pafs.LocalFileSystem()


def path_from_store(store: Any) -> PathLike | None:
    """Derive the user-facing path from a zarr store, or ``None`` for stores without one.

    - ``LocalStore`` → ``Path(store.root)``
    - ``FsspecStore`` → ``UPath("{protocol}://{store.path}", fs=sync_fs)``
    - Anything else (``MemoryStore``, custom stores) → ``None`` (no meaningful path)

    The fsspec branch unwraps ``AsyncFileSystemWrapper`` so the returned ``UPath``
    carries the original sync filesystem (required by pyarrow's ``FSSpecHandler``).
    """
    if isinstance(store, LocalStore):
        return Path(store.root)
    if isinstance(store, FsspecStore):
        protocol = getattr(store.fs, "protocol", None)
        if isinstance(protocol, (list, tuple)):
            protocol = protocol[0] if protocol else "file"
        elif protocol is None:
            protocol = "file"
        fs = store.fs
        while True:
            inner = getattr(fs, "sync_fs", None)
            if inner is None or inner is fs:
                break
            fs = inner
        return UPath(f"{protocol}://{store.path}", fs=fs)
    return None


def store_from_group(group: zarr.Group, *, read_only: bool = True) -> Any:
    """Return a zarr store re-rooted at ``group.path``.

    For consumers (e.g. ``ome_zarr.io.ZarrLocation``, ``anndata.read_zarr``) that
    need a store and don't understand sub-paths inside one. Falls back to returning
    the parent store unchanged for stores we cannot re-root (``MemoryStore`` etc.).
    """
    from spatialdata._io._utils import join_fsspec_store_path

    parent = group.store
    if isinstance(parent, LocalStore):
        return LocalStore(Path(parent.root) / group.path, read_only=read_only)
    if isinstance(parent, FsspecStore):
        return FsspecStore(
            parent.fs,
            path=join_fsspec_store_path(parent.path, group.path),
            read_only=read_only,
        )
    return parent


def arrow_fs_and_path(group: zarr.Group, *child_parts: str) -> tuple[pafs.FileSystem, str]:
    """Derive a pyarrow filesystem + path for I/O at ``group/<child_parts>``.

    Used by parquet readers/writers (points, shapes) so they can locate their backing
    file without going through a ``UPath``: the zarr group already carries its store
    (and thus the filesystem), so we derive both directly.

    TODO(async-pyarrow-fs): drop the ``sync_fs`` unwrap when either pyarrow's
    FSSpecHandler learns to drive an async fs, or zarr exposes the original sync fs
    directly (https://github.com/zarr-developers/zarr-python/issues/2073).
    """
    from spatialdata._io._utils import join_fsspec_store_path

    store = group.store
    child = "/".join(child_parts)

    if isinstance(store, LocalStore):
        local_path = Path(store.root) / group.path / child if child else Path(store.root) / group.path
        return pafs.LocalFileSystem(), str(local_path)

    if isinstance(store, FsspecStore):
        fs = store.fs
        while True:
            inner = getattr(fs, "sync_fs", None)
            if inner is None or inner is fs:
                break
            fs = inner
        sub = f"{group.path}/{child}" if child else group.path
        return pafs.PyFileSystem(pafs.FSSpecHandler(fs)), join_fsspec_store_path(store.path, sub)

    raise ValueError(f"Cannot derive a pyarrow filesystem for store of type {type(store).__name__}")


@contextmanager
def open_read_store(path: PathLike) -> Any:
    """Open *path* as a read-only zarr backend store.

    The store is constructed with ``read_only=True`` so the underlying
    ``LocalStore`` / ``FsspecStore`` refuses writes at the store layer (not just
    at the group ``mode="r"`` level). This also lets public HTTPS zarrs skip any
    write-capability probe that fsspec may otherwise perform.
    """
    from spatialdata._io._utils import _resolve_zarr_store

    resolved_store = _resolve_zarr_store(path, read_only=True)
    try:
        yield resolved_store
    finally:
        resolved_store.close()


def open_zarr_for_read(store: Any, *, as_group: bool = True) -> Any:
    """Open a zarr group or node for reading with remote-friendly defaults.

    Prefers the fast path: pinned ``zarr_format=3`` (we only ever write v3 stores,
    so skipping v2-metadata auto-probes saves up to five small GETs per open on
    remote backends) and ``use_consolidated=True`` (requires the root / element
    ``zarr.json`` to carry the ``consolidated_metadata`` field produced by
    ``_write_consolidated_metadata``). Falls back to ``zarr.open*`` with no
    format/consolidation hints for legacy or third-party stores that predate
    either convention.

    Parameters
    ----------
    store
        A ``zarr.storage.StoreLike`` -- typically the value yielded by
        ``open_read_store``.
    as_group
        If ``True`` (default) use ``zarr.open_group``; if ``False`` use
        ``zarr.open`` which returns either a ``Group`` or an ``Array`` based on
        the metadata at the store root.
    """
    fn = zarr.open_group if as_group else zarr.open
    try:
        return fn(store, mode="r", zarr_format=3, use_consolidated=True)
    except (ValueError, FileNotFoundError):
        return fn(store, mode="r")


@contextmanager
def open_write_store(path: PathLike) -> Any:
    """Open *path* as a writable zarr backend store (``read_only=False``)."""
    from spatialdata._io._utils import _resolve_zarr_store

    resolved_store = _resolve_zarr_store(path, read_only=False)
    try:
        yield resolved_store
    finally:
        resolved_store.close()
