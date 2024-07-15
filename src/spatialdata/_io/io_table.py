from __future__ import annotations

import os

import numpy as np
import zarr
from anndata import AnnData
from anndata import read_zarr as read_anndata_zarr
from anndata._io.specs import read_elem
from anndata._io.specs import write_elem as write_adata
from ome_zarr.format import Format

from spatialdata._io.format import CurrentTablesFormat, TablesFormats
from spatialdata._logging import logger
from spatialdata.models import TableModel


def _read_table(
    zarr_store_path: str, group: zarr.Group, subgroup: zarr.Group, tables: dict[str, AnnData]
) -> dict[str, AnnData]:
    """
    Read in tables in the tables Zarr.group of a SpatialData Zarr store.

    Parameters
    ----------
    zarr_store_path
        The path to the Zarr store.
    group
        The parent group containing the subgroup.
    subgroup
        The subgroup containing the tables.
    tables
        A dictionary of tables.

    Returns
    -------
    The modified dictionary with the tables.
    """
    count = 0
    for table_name in subgroup:
        f_elem = subgroup[table_name]
        f_elem_store = os.path.join(zarr_store_path, f_elem.path)
        if isinstance(group.store, zarr.storage.ConsolidatedMetadataStore):
            # TODO: read version and get table format
            tables[table_name] = read_elem(f_elem)
            # we can replace read_elem with read_anndata_zarr after this PR gets into a release (>= 0.6.5)
            # https://github.com/scverse/anndata/pull/1057#pullrequestreview-1530623183
            # table = read_anndata_zarr(f_elem)
        else:
            # TODO: read version and get table format
            tables[table_name] = read_anndata_zarr(f_elem_store)
        # replace with format from above
        version = "0.1"
        format = TablesFormats[version]
        if TableModel.ATTRS_KEY in tables[table_name].uns:
            # fill out eventual missing attributes that has been omitted because their value was None
            attrs = tables[table_name].uns[TableModel.ATTRS_KEY]
            if "region" not in attrs:
                attrs["region"] = None
            if "region_key" not in attrs:
                attrs["region_key"] = None
            if "instance_key" not in attrs:
                attrs["instance_key"] = None
            # fix type for region
            if "region" in attrs and isinstance(attrs["region"], np.ndarray):
                attrs["region"] = attrs["region"].tolist()

        count += 1

    logger.debug(f"Found {count} elements in {subgroup}")
    return tables


def write_table(
    table: AnnData,
    group: zarr.Group,
    name: str,
    group_type: str = "ngff:regions_table",
    format: Format = CurrentTablesFormat(),
) -> None:
    if TableModel.ATTRS_KEY in table.uns:
        region = table.uns["spatialdata_attrs"]["region"]
        region_key = table.uns["spatialdata_attrs"].get("region_key", None)
        instance_key = table.uns["spatialdata_attrs"].get("instance_key", None)
        format.validate_table(table, region_key, instance_key)
    else:
        region, region_key, instance_key = (None, None, None)
    write_adata(group, name, table)  # creates group[name]
    tables_group = group[name]
    tables_group.attrs["spatialdata-encoding-type"] = group_type
    tables_group.attrs["region"] = region
    tables_group.attrs["region_key"] = region_key
    tables_group.attrs["instance_key"] = instance_key
    tables_group.attrs["spatialdata_format_version"] = format.spatialdata_format_version
