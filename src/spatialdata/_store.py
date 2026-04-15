from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import zarr
from upath import UPath

PathLike = Path | UPath


def _normalize_path(path: str | PathLike) -> PathLike:
    if isinstance(path, str):
        return UPath(path) if "://" in path else Path(path)
    if isinstance(path, (Path, UPath)):
        return path
    raise TypeError("Path must be `None`, a `str`, a `Path` or a `UPath` object.")


@dataclass(frozen=True)
class ZarrStore:
    path: PathLike
    storage_options: dict[str, Any] = field(default_factory=dict)

    def with_path(self, path: str | PathLike) -> ZarrStore:
        return replace(self, path=_normalize_path(path))


def make_zarr_store(
    path: str | PathLike,
    *,
    storage_options: dict[str, Any] | None = None,
) -> ZarrStore:
    return ZarrStore(
        path=_normalize_path(path),
        storage_options={} if storage_options is None else dict(storage_options),
    )


@contextmanager
def open_read_store(store: ZarrStore) -> Any:
    from spatialdata._io._utils import _resolve_zarr_store

    resolved_store = _resolve_zarr_store(store.path, **store.storage_options)
    try:
        yield resolved_store
    finally:
        resolved_store.close()


@contextmanager
def open_write_store(store: ZarrStore) -> Any:
    from spatialdata._io._utils import _resolve_zarr_store

    resolved_store = _resolve_zarr_store(store.path, **store.storage_options)
    try:
        yield resolved_store
    finally:
        resolved_store.close()


def open_group_from_store(
    store: zarr.storage.StoreLike,
    *,
    mode: str,
    use_consolidated: bool | None = None,
) -> zarr.Group:
    kwargs: dict[str, Any] = {"store": store, "mode": mode}
    if use_consolidated is not None:
        kwargs["use_consolidated"] = use_consolidated
    return zarr.open_group(**kwargs)
