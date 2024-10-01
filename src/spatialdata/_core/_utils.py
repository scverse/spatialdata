from collections.abc import Iterable

from spatialdata._core.spatialdata import SpatialData


def _find_common_table_keys(sdatas: Iterable[SpatialData]) -> set[str]:
    """
    Find table keys present in more than one SpatialData object.

    Parameters
    ----------
    sdatas
        An `Iterable` of SpatialData objects.

    Returns
    -------
    A set of common keys that are present in the tables of more than one SpatialData object.
    """
    common_keys: set[str] = set()

    for sdata in sdatas:
        if len(common_keys) == 0:
            common_keys = set(sdata.tables.keys())
        else:
            common_keys.intersection_update(sdata.tables.keys())

    return common_keys
