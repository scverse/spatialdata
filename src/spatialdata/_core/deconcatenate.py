from __future__ import annotations

from collections.abc import Iterable

from spatialdata._core.query.relational_query import match_sdata_to_table
from spatialdata._core.spatialdata import SpatialData


def deconcatenate(
    full_sdata: SpatialData,
    by: str | Iterable[str],
    target_coordinate_system: str,
    full_sdata_table_name: str = "table",
    sdatas_table_names: str | Iterable[str] = "table",
    region_key: str = "region",
    join: str = "right",
) -> SpatialData | list[SpatialData]:
    """
    From a `SpatialData` object containing multiple regions, returns `SpatialData` objects specific to each region identified in `by`.
    """
    if full_sdata_table_name not in full_sdata.tables:
        raise KeyError("Missing table")

    sdata_table = full_sdata[full_sdata_table_name]

    is_single_region = isinstance(by, str)
    deconcat_regions = [by] if is_single_region else list(by)
    sdatas_table_names = (
        [sdatas_table_names] * len(deconcat_regions)
        if isinstance(sdatas_table_names, str)
        else list(sdatas_table_names)
    )

    sdatas = []
    for region, table_name in zip(deconcat_regions, sdatas_table_names):
        deconcat_table = sdata_table[sdata_table.obs[region_key] == region]
        deconcat_sdata = match_sdata_to_table(full_sdata, table=deconcat_table, table_name=table_name, how=join)

        sdatas.append(deconcat_sdata)

    return sdatas[0] if is_single_region else sdatas
