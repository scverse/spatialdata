from __future__ import annotations

import os
from json import JSONDecodeError
from typing import Literal

import numpy as np
import zarr
from anndata import AnnData
from anndata import read_zarr as read_anndata_zarr
from anndata._io.specs import write_elem as write_adata
from ome_zarr.format import Format
from zarr.errors import ArrayNotFoundError

from spatialdata._io._utils import BadFileHandleMethod, handle_read_errors
from spatialdata._io.format import CurrentTablesFormat, TablesFormats, _parse_version
from spatialdata._logging import logger
from spatialdata.models import TableModel


def _read_table(
    zarr_store_path: str,
    group: zarr.Group,
    subgroup: zarr.Group,
    tables: dict[str, AnnData],
    on_bad_files: Literal[BadFileHandleMethod.ERROR, BadFileHandleMethod.WARN] = BadFileHandleMethod.ERROR,
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
    on_bad_files
        Specifies what to do upon encountering a bad file, e.g. corrupted, invalid or missing files.

    Returns
    -------
    The modified dictionary with the tables.
    """
    count = 0
    for table_name in subgroup:
        f_elem = subgroup[table_name]
        f_elem_store = os.path.join(zarr_store_path, f_elem.path)

        with handle_read_errors(
            on_bad_files=on_bad_files,
            location=f"{subgroup.path}/{table_name}",
            exc_types=(JSONDecodeError, KeyError, ValueError, ArrayNotFoundError),
        ):
            tables[table_name] = read_anndata_zarr(f_elem_store)

            f = zarr.open(f_elem_store, mode="r")
            version = _parse_version(f, expect_attrs_key=False)
            assert version is not None
            # since have just one table format, we currently read it but do not use it; if we ever change the format
            # we can rename the two _ to format and implement the per-format read logic (as we do for shapes)
            _ = TablesFormats[version]
            f.store.close()

            # # replace with format from above
            # version = "0.1"
            # format = TablesFormats[version]
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
    tables_group.attrs["version"] = format.spatialdata_format_version
