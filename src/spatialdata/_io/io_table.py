from pathlib import Path

import numpy as np
import zarr
from anndata import AnnData
from anndata import read_zarr as read_anndata_zarr
from anndata._io.specs import write_elem as write_adata
from ome_zarr.format import Format

from spatialdata._io.format import CurrentTablesFormat, TablesFormats, TablesFormatV01, TablesFormatV02, _parse_version
from spatialdata.models import TableModel, get_table_keys


def _read_table(store: str | Path, lazy: bool = False) -> AnnData:
    """
    Read a table from a zarr store.

    Parameters
    ----------
    store
        Path to the zarr store containing the table.
    lazy
        If True, read the table lazily using anndata.experimental.read_lazy.
        This requires anndata >= 0.12. If the installed version does not support
        lazy reading, a warning is raised and the table is read eagerly.

    Returns
    -------
    The AnnData table, either lazily loaded or in-memory.
    """
    if lazy:
        try:
            from anndata.experimental import read_lazy

            table = read_lazy(str(store))
        except ImportError:
            import warnings

            warnings.warn(
                "Lazy reading of tables requires anndata >= 0.12. "
                "Falling back to eager reading. To enable lazy reading, "
                "upgrade anndata with: pip install 'anndata>=0.12'",
                UserWarning,
                stacklevel=2,
            )
            table = read_anndata_zarr(str(store))
    else:
        table = read_anndata_zarr(str(store))

    f = zarr.open(store, mode="r")
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
    if TableModel.ATTRS_KEY in table.uns:
        region, region_key, instance_key = get_table_keys(table)
        TableModel().validate(table)
    else:
        region, region_key, instance_key = (None, None, None)

    write_adata(group, name, table)
    tables_group = group[name]
    tables_group.attrs["spatialdata-encoding-type"] = group_type
    tables_group.attrs["region"] = region
    tables_group.attrs["region_key"] = region_key
    tables_group.attrs["instance_key"] = instance_key
    tables_group.attrs["version"] = element_format.spatialdata_format_version
