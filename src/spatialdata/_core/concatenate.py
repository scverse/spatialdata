from __future__ import annotations

from copy import copy  # Should probably go up at the top
from itertools import chain
from typing import TYPE_CHECKING, Any

import numpy as np
from anndata import AnnData

if TYPE_CHECKING:
    from spatialdata._core.spatialdata import SpatialData

from spatialdata.models import TableModel

__all__ = [
    "concatenate",
]


def _concatenate_tables(
    tables: list[AnnData],
    region_key: str | None = None,
    instance_key: str | None = None,
    **kwargs: Any,
) -> AnnData:
    import anndata as ad

    region_keys = [table.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY_KEY] for table in tables]
    instance_keys = [table.uns[TableModel.ATTRS_KEY][TableModel.INSTANCE_KEY] for table in tables]
    regions = [table.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY] for table in tables]

    if len(set(region_keys)) == 1:
        region_key = list(region_keys)[0]
    else:
        if region_key is None:
            raise ValueError("`region_key` must be specified if tables have different region keys")

    # get unique regions from list of lists or str
    regions_unique = list(chain(*[[i] if isinstance(i, str) else i for i in regions]))
    if len(set(regions_unique)) != len(regions_unique):
        raise ValueError(f"Two or more tables seems to annotate regions with the same name: {regions_unique}")

    if len(set(instance_keys)) == 1:
        instance_key = list(instance_keys)[0]
    else:
        if instance_key is None:
            raise ValueError("`instance_key` must be specified if tables have different instance keys")

    tables_l = []
    for table_region_key, table_instance_key, table in zip(region_keys, instance_keys, tables):
        rename_dict = {}
        if table_region_key != region_key:
            rename_dict[table_region_key] = region_key
        if table_instance_key != instance_key:
            rename_dict[table_instance_key] = instance_key
        if len(rename_dict) > 0:
            table = copy(table)  # Shallow copy
            table.obs = table.obs.rename(columns=rename_dict, copy=False)
        tables_l.append(table)

    merged_table = ad.concat(tables_l, **kwargs)
    attrs = {
        TableModel.REGION_KEY: merged_table.obs[TableModel.REGION_KEY].unique().tolist(),
        TableModel.REGION_KEY_KEY: region_key,
        TableModel.INSTANCE_KEY: instance_key,
    }
    merged_table.uns[TableModel.ATTRS_KEY] = attrs

    return TableModel().validate(merged_table)


def concatenate(
    sdatas: list[SpatialData],
    region_key: str | None = None,
    instance_key: str | None = None,
    **kwargs: Any,
) -> SpatialData:
    """
    Concatenate a list of spatial data objects.

    Parameters
    ----------
    sdatas
        The spatial data objects to concatenate.
    region_key
        The key to use for the region column in the concatenated object.
        If all region_keys are the same, the `region_key` is used.
    instance_key
        The key to use for the instance column in the concatenated object.
    kwargs
        See :func:`anndata.concat` for more details.

    Returns
    -------
    The concatenated :class:`spatialdata.SpatialData` object.
    """
    from spatialdata import SpatialData

    merged_images = {**{k: v for sdata in sdatas for k, v in sdata.images.items()}}
    if len(merged_images) != np.sum([len(sdata.images) for sdata in sdatas]):
        raise KeyError("Images must have unique names across the SpatialData objects to concatenate")
    merged_labels = {**{k: v for sdata in sdatas for k, v in sdata.labels.items()}}
    if len(merged_labels) != np.sum([len(sdata.labels) for sdata in sdatas]):
        raise KeyError("Labels must have unique names across the SpatialData objects to concatenate")
    merged_points = {**{k: v for sdata in sdatas for k, v in sdata.points.items()}}
    if len(merged_points) != np.sum([len(sdata.points) for sdata in sdatas]):
        raise KeyError("Points must have unique names across the SpatialData objects to concatenate")
    merged_shapes = {**{k: v for sdata in sdatas for k, v in sdata.shapes.items()}}
    if len(merged_shapes) != np.sum([len(sdata.shapes) for sdata in sdatas]):
        raise KeyError("Shapes must have unique names across the SpatialData objects to concatenate")

    assert type(sdatas) == list, "sdatas must be a list"
    assert len(sdatas) > 0, "sdatas must be a non-empty list"

    merged_table = _concatenate_tables(
        [sdata.table for sdata in sdatas if sdata.table is not None], region_key, instance_key, **kwargs
    )

    return SpatialData(
        images=merged_images,
        labels=merged_labels,
        points=merged_points,
        shapes=merged_shapes,
        table=merged_table,
    )


def _filter_table_in_coordinate_systems(table: AnnData, coordinate_systems: list[str]) -> AnnData:
    table_mapping_metadata = table.uns[TableModel.ATTRS_KEY]
    region_key = table_mapping_metadata[TableModel.REGION_KEY_KEY]
    new_table = table[table.obs[region_key].isin(coordinate_systems)].copy()
    new_table.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY] = new_table.obs[region_key].unique().tolist()
    return new_table
