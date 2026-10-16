from __future__ import annotations

import warnings
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import zarr
from anndata import read_zarr as read_anndata_zarr
from anndata._io.specs import read_elem
from anndata._io.specs import write_elem as write_adata
from mudata import MuData
from mudata import read_zarr as read_mudata_zarr
from packaging.version import Version

from spatialdata._core.validation import MUDATA_MODALITY_SEPARATOR
from spatialdata._io._utils import _resolve_zarr_store
from spatialdata._io.exceptions import FormatVersionUnknownError, WritingToZarrV2DeprecationWarning
from spatialdata._io.format import (
    CurrentTablesFormat,
    TablesFormats,
    TablesFormatType,
    TablesFormatV01,
    TablesFormatV02,
    _parse_version,
)
from spatialdata._types import Table_T
from spatialdata._utils import _copy_table, _restore_mudata_global_annotations
from spatialdata.models import TableModel, get_table_keys

# value of the `encoding-type` attribute that mudata writes to the root group of a MuData object
MUDATA_ENCODING_TYPE = "MuData"
# The columns of the global obs/var of a MuData table may contain ":" ("<modality>:<column>"), which is not allowed in
# Windows file names, and each column is stored as a zarr array named after it; ":" is therefore stored percent-encoded.
# "%" is not allowed in the names of the columns of a table, so the encoded names are decoded without ambiguity.
_ESCAPED_MUDATA_MODALITY_SEPARATOR = "%3A"


def _escape_column_names(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=lambda c: c.replace(MUDATA_MODALITY_SEPARATOR, _ESCAPED_MUDATA_MODALITY_SEPARATOR))


def _unescape_column_names(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns=lambda c: c.replace(_ESCAPED_MUDATA_MODALITY_SEPARATOR, MUDATA_MODALITY_SEPARATOR))


def _read_table(store: str | Path) -> Table_T:
    f = zarr.open(Path(store), mode="r")  # Path avoids zarr v3 URL-parsing special chars (e.g. #) in names
    if not isinstance(f, zarr.Group):
        raise TypeError(f"Expected a zarr group holding a table element, got {type(f).__name__}.")

    # read the table before parsing the version: an incomplete table group then raises a reading error (e.g. a
    # KeyError), which `on_bad_files="warn"` turns into a warning
    table: Table_T
    if f.attrs.get("encoding-type") == MUDATA_ENCODING_TYPE:
        # pass the store of the group instead of a str, for the same reason as above
        mdata = read_mudata_zarr(f.store)
        obs, var = read_elem(f["obs"]), read_elem(f["var"])
        if not isinstance(mdata, MuData) or not isinstance(obs, pd.DataFrame) or not isinstance(var, pd.DataFrame):
            raise TypeError(f"Expected the zarr group {store} to hold a MuData table.")
        obs, var = _unescape_column_names(obs), _unescape_column_names(var)
        # undo the dtype changes that MuData applies to the global annotations when it is constructed
        _restore_mudata_global_annotations(mdata, obs=obs, var=var)
        mdata.obs, mdata.var = _unescape_column_names(mdata.obs), _unescape_column_names(mdata.var)
        table = mdata
    else:
        table = read_anndata_zarr(str(store))

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


def _write_mudata(group: zarr.Group, name: str, mdata: MuData) -> None:
    """
    Write a MuData table to the subgroup `name` of `group`, in the MuData zarr format.

    `mudata.write_zarr()` is not used because it converts the strings of the table to categoricals in place (also in
    the modalities, and on disk regardless of `convert_strings_to_categoricals`), rebuilds the global annotations of
    the table once written, and fails on views. Here, the elements are written as `anndata` writes the elements of an
    AnnData table.
    """
    import mudata

    if mdata.is_view:
        mdata = _copy_table(mdata)
    table_group = group.create_group(name, overwrite=True)
    for attr in ("obs", "var"):
        write_adata(table_group, attr, _escape_column_names(getattr(mdata, attr)))
    for attr in ("obsm", "varm", "obsp", "varp", "uns", "obsmap", "varmap"):
        write_adata(table_group, attr, dict(getattr(mdata, attr)))
    mod_group = table_group.require_group("mod")
    for mod_name, modality in mdata.mod.items():
        write_adata(mod_group, mod_name, modality)
    mod_group.attrs["mod-order"] = list(mdata.mod)
    table_group.attrs.update(
        {"axis": mdata.axis, "encoding-type": MUDATA_ENCODING_TYPE, "encoding-version": mudata.__mudataversion__}
    )


def write_table(
    table: Table_T,
    group: zarr.Group,
    name: str,
    group_type: str = "ngff:regions_table",
    element_format: TablesFormatType = CurrentTablesFormat(),
    convert_strings_to_categoricals: bool = False,
) -> None:
    """
    Write a table to a Zarr store.

    Parameters
    ----------
    table
        The table to write, either an AnnData or a MuData object.
    group
        The table will be written into a subgroup of this group
    name
        The name of the subgroup of `group` to which table is to be written.
    group_type
        The type of the group.
    element_format
        The format to use for writing the table.
    convert_strings_to_categoricals
        If True, convert string columns to categoricals before writing.
        Note that this will have a side effect of modifying dtypes of the input table in place.
    """
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

    if isinstance(table, MuData):
        if convert_strings_to_categoricals:
            table.strings_to_categoricals()
        _write_mudata(group, name, table)

    elif element_format.zarr_format == 3 and Version(version("anndata")) >= Version("0.13"):
        # `write_zarr` in anndata v0.13 and above can only write to zarr v3
        # solution of passing resolved store directly roughly based on:
        # https://github.com/scverse/anndata/issues/1548#issuecomment-2199801855

        # resolve the store from the group
        resolved_store = _resolve_zarr_store(table_group)

        # Write the table to the path of the table group
        table.write_zarr(
            store=resolved_store,
            consolidate_metadata=False,
            convert_strings_to_categoricals=convert_strings_to_categoricals,
        )

    else:
        if convert_strings_to_categoricals:
            table.strings_to_categoricals()

        write_adata(group, name, table)

    # Re-fetch the group before setting the attributes below: the handle obtained above was cached while the group
    # was still empty, and zarr writes attributes as a whole document based on the handle's cached view, so writing
    # through the stale handle would erase the `encoding-type`/`encoding-version` metadata that anndata just wrote
    # (https://github.com/scverse/spatialdata/issues/1183).
    written_table_group = group[name]
    if not isinstance(written_table_group, zarr.Group):
        raise TypeError(f"Expected a zarr group holding the table {name!r}, got {type(written_table_group).__name__}.")
    table_group = written_table_group

    table_group.attrs["spatialdata-encoding-type"] = group_type
    table_group.attrs["region"] = region
    table_group.attrs["region_key"] = region_key
    table_group.attrs["instance_key"] = instance_key
    table_group.attrs["version"] = element_format.spatialdata_format_version
