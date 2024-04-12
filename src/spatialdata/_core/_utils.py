from typing import Any

import zarr
from upath import UPath
from zarr.storage import FSStore


# Any instead of SpatialData to avoid circular import
def _find_common_table_keys(sdatas: list[Any]) -> set[str]:
    """
    Find table keys present in more than one SpatialData object.

    Parameters
    ----------
    sdatas
        A list of SpatialData objects.

    Returns
    -------
    A set of common keys that are present in the tables of more than one SpatialData object.
    """
    common_keys = set(sdatas[0].tables.keys())

    for sdata in sdatas[1:]:
        common_keys.intersection_update(sdata.tables.keys())

    return common_keys


def _open_zarr_store(path: UPath, **kwargs: Any) -> zarr.storage.BaseStore:
    """
    Open a zarr store (on-disk or remote) and return the corresponding zarr.storage.BaseStore object.

    Parameters
    ----------
    path
        Path to the zarr store (on-disk or remote).
    kwargs
        Additional keyword arguments to pass to the zarr.storage.FSStore constructor.

    Returns
    -------
    The zarr.storage.BaseStorage object.
    """
    return FSStore(url=path.path, fs=path.fs, **kwargs)
