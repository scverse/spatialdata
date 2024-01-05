from spatialdata._core.spatialdata import SpatialData


def _find_common_table_keys(sdatas: list[SpatialData]) -> set[str]:
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
