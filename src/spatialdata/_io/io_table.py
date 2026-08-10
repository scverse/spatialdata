from __future__ import annotations

import warnings
from importlib.metadata import version
from pathlib import Path

import numpy as np
import zarr
from anndata import AnnData
from anndata import read_zarr as read_anndata_zarr
from anndata._io.specs import write_elem as write_adata
from ome_zarr.format import Format
from packaging.version import Version

from spatialdata._io._utils import _resolve_zarr_store
from spatialdata._io.exceptions import FormatVersionUnknownError, WritingToZarrV2DeprecationWarning
from spatialdata._io.format import (
    CurrentTablesFormat,
    TablesFormats,
    TablesFormatV01,
    TablesFormatV02,
    _parse_version,
)
from spatialdata.models import TableModel, get_table_keys


def _read_table(store: str | Path) -> AnnData:
    table = read_anndata_zarr(str(store))

    f = zarr.open(Path(store), mode="r")  # Path avoids zarr v3 URL-parsing special chars (e.g. #) in names
    version = _parse_version(f, expect_attrs_key=False)
    assert version is not None
    table_format = TablesFormats[version]

    f.store.close()

    if isinstance(table_format, TablesFormatV01 | TablesFormatV02):
        if TableModel.ATTRS_KEY in table.uns:
            # fill out eventual missing attributes that has been omitted because their value was None
            attrs = table.uns[TableModel.ATTRS_KEY]
            if "region" not in attrs:
                attrs["region"] = None
            if "region_key" not in attrs:
                attrs["region_key"] = None
            if "instance_key" not in attrs:
                attrs["instance_key"] = None
            # fix type for region
            if "region" in attrs and isinstance(attrs["region"], np.ndarray):
                attrs["region"] = attrs["region"].tolist()
    else:
        raise ValueError(
            f"Unsupported table format: {type(table_format)}. Supported formats are: {TablesFormats.values()}"
        )
    return table


def write_table(
    table: AnnData,
    group: zarr.Group,
    name: str,
    group_type: str = "ngff:regions_table",
    element_format: Format = CurrentTablesFormat(),
) -> None:
    if element_format.zarr_format == 2:
        warnings.warn(
            message=WritingToZarrV2DeprecationWarning.message, category=WritingToZarrV2DeprecationWarning, stacklevel=2
        )

    if TableModel.ATTRS_KEY in table.uns:
        region, region_key, instance_key = get_table_keys(table)
        TableModel.validate(table)
    else:
        region, region_key, instance_key = (None, None, None)

    # Ensure the table group exists
    table_group = group.require_group(name=name)

    if element_format not in TablesFormats.values():
        raise FormatVersionUnknownError(element_type="table", version_encountered=element_format)

    if element_format.zarr_format == 3 and Version(version("anndata")) >= Version("0.13"):
        # `write_zarr` in anndata v0.13 and above can only write to zarr v3
        # solution of passing resolved store directly roughly based on:
        # https://github.com/scverse/anndata/issues/1548#issuecomment-2199801855

        # resolve the store from the group
        resolved_store = _resolve_zarr_store(table_group)

        # Write the table to the path of the table group
        table.write_zarr(store=resolved_store, consolidate_metadata=False)

        table_group = group[name]
    else:
        table.strings_to_categoricals()
        write_adata(group, name, table)
        table_group = group[name]

    table_group.attrs["spatialdata-encoding-type"] = group_type
    table_group.attrs["region"] = region
    table_group.attrs["region_key"] = region_key
    table_group.attrs["instance_key"] = instance_key
    table_group.attrs["version"] = element_format.spatialdata_format_version
