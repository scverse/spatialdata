from collections import defaultdict
from collections.abc import Callable, Iterable
from copy import copy  # Should probably go up at the top
from itertools import chain
from typing import Any
from warnings import warn

import numpy as np
from anndata import AnnData
from anndata._core.merge import StrategiesLiteral, resolve_merge_strategy

from spatialdata._core._utils import _find_common_table_keys
from spatialdata._core.spatialdata import SpatialData
from spatialdata.models import TableModel, get_table_keys
from spatialdata.transformations import (
    get_transformation,
    remove_transformation,
    set_transformation,
)

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

    if not all(TableModel.ATTRS_KEY in table.uns for table in tables):
        raise ValueError("Not all tables are annotating a spatial element")
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
    for table_region_key, table_instance_key, table in zip(region_keys, instance_keys, tables, strict=True):
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
        TableModel.REGION_KEY: merged_table.obs[region_key].unique().tolist(),
        TableModel.REGION_KEY_KEY: region_key,
        TableModel.INSTANCE_KEY: instance_key,
    }
    merged_table.uns[TableModel.ATTRS_KEY] = attrs
    return TableModel().parse(merged_table)


def concatenate(
    sdatas: Iterable[SpatialData] | dict[str, SpatialData],
    region_key: str | None = None,
    instance_key: str | None = None,
    concatenate_tables: bool = False,
    obs_names_make_unique: bool = True,
    modify_tables_inplace: bool = False,
    merge_coordinate_systems_on_name: bool = False,
    attrs_merge: (StrategiesLiteral | Callable[[list[dict[Any, Any]]], dict[Any, Any]] | None) = None,
    **kwargs: Any,
) -> SpatialData:
    """
    Concatenate a list of spatial data objects.

    Parameters
    ----------
    sdatas
        The spatial data objects to concatenate. The names of the elements across the `SpatialData` objects must be
        unique. If they are not unique, you can pass a dictionary with the suffixes as keys and the spatial data objects
        as values. This will rename the names of each `SpatialElement` to ensure uniqueness of names across
        `SpatialData` objects. See more on the notes.
    region_key
        The key to use for the region column in the concatenated object.
        If `None` and all region_keys are the same, the `region_key` is used.
    instance_key
        The key to use for the instance column in the concatenated object.
        If `None` and all instance_keys are the same, the `instance_key` is used.
    concatenate_tables
        Whether to merge the tables in case of having the same element name.
    obs_names_make_unique
        Whether to make the `obs_names` unique by calling `AnnData.obs_names_make_unique()` on each table of the
        concatenated object. If you passed a dictionary with the suffixes as keys and the `SpatialData` objects as
        values and if `concatenate_tables` is `True`, the `obs_names` will be made unique by adding the corresponding
        suffix instead.
    modify_tables_inplace
        Whether to modify the tables in place. If `True`, the tables will be modified in place. If `False`, the tables
        will be copied before modification. Copying is enabled by default but can be disabled for performance reasons.
    merge_coordinate_systems_on_name
        Whether to keep coordinate system names unchanged (True) or add suffixes (False).
    attrs_merge
        How the elements of `.attrs` are selected. Uses the same set of strategies as the `uns_merge` argument of [anndata.concat](https://anndata.readthedocs.io/en/latest/generated/anndata.concat.html)
    kwargs
        See :func:`anndata.concat` for more details.

    Returns
    -------
    The concatenated :class:`spatialdata.SpatialData` object.

    Notes
    -----
    If you pass a dictionary with the suffixes as keys and the `SpatialData` objects as values, the names of each
    `SpatialElement` will be renamed to ensure uniqueness of names across `SpatialData` objects by adding the
    corresponding suffix. To ensure the matching between existing table annotations, the `region` metadata of each
    table, and the values of the `region_key` column in each table, will be altered by adding the suffix. In addition,
    the `obs_names` of each table will be altered (a suffix will be added). Finally, a suffix will be added to the name
    of each table iff `rename_tables` is `False`.

    If you need more control in the renaming, please give us feedback, as we are still trying to find the right balance
    between ergonomics and control. Also, you are welcome to copy and adjust the code of
    `_fix_ensure_unique_element_names()` directly.
    """
    if not isinstance(sdatas, Iterable):
        raise TypeError("`sdatas` must be a `Iterable`")

    if isinstance(sdatas, dict):
        sdatas = _fix_ensure_unique_element_names(
            sdatas,
            rename_tables=not concatenate_tables,
            rename_obs_names=obs_names_make_unique and concatenate_tables,
            modify_tables_inplace=modify_tables_inplace,
            merge_coordinate_systems_on_name=merge_coordinate_systems_on_name,
        )
    elif merge_coordinate_systems_on_name:
        raise ValueError("`merge_coordinate_systems_on_name` can only be used if `sdatas` is a dictionary")

    ERROR_STR = (
        " must have unique names across the SpatialData objects to concatenate. Please pass a `dict[str, SpatialData]`"
        " to `concatenate()` to address this (see docstring)."
    )

    merged_images = {**{k: v for sdata in sdatas for k, v in sdata.images.items()}}
    if len(merged_images) != np.sum([len(sdata.images) for sdata in sdatas]):
        raise KeyError("Images" + ERROR_STR)
    merged_labels = {**{k: v for sdata in sdatas for k, v in sdata.labels.items()}}
    if len(merged_labels) != np.sum([len(sdata.labels) for sdata in sdatas]):
        raise KeyError("Labels" + ERROR_STR)
    merged_points = {**{k: v for sdata in sdatas for k, v in sdata.points.items()}}
    if len(merged_points) != np.sum([len(sdata.points) for sdata in sdatas]):
        raise KeyError("Points" + ERROR_STR)
    merged_shapes = {**{k: v for sdata in sdatas for k, v in sdata.shapes.items()}}
    if len(merged_shapes) != np.sum([len(sdata.shapes) for sdata in sdatas]):
        raise KeyError("Shapes" + ERROR_STR)

    if not concatenate_tables:
        key_counts: dict[str, int] = defaultdict(int)
        for sdata in sdatas:
            for k in sdata.tables:
                key_counts[k] += 1

        if any(value > 1 for value in key_counts.values()):
            warn(
                "Duplicate table names found. Tables will be added with integer suffix. Set `concatenate_tables` to "
                "`True` if concatenation is wished for instead.",
                UserWarning,
                stacklevel=2,
            )
        merged_tables = {}
        count_dict: dict[str, int] = defaultdict(int)

        for sdata in sdatas:
            for k, v in sdata.tables.items():
                new_key = f"{k}_{count_dict[k]}" if key_counts[k] > 1 else k
                count_dict[k] += 1
                merged_tables[new_key] = v
    else:
        common_keys = _find_common_table_keys(sdatas)
        merged_tables = {}
        for sdata in sdatas:
            for k, v in sdata.tables.items():
                if k in common_keys and merged_tables.get(k) is not None:
                    merged_tables[k] = _concatenate_tables([merged_tables[k], v], region_key, instance_key, **kwargs)
                else:
                    merged_tables[k] = v

    attrs_merge = resolve_merge_strategy(attrs_merge)
    attrs = attrs_merge([sdata.attrs for sdata in sdatas])

    sdata = SpatialData(
        images=merged_images,
        labels=merged_labels,
        points=merged_points,
        shapes=merged_shapes,
        tables=merged_tables,
        attrs=attrs,
    )
    if obs_names_make_unique:
        for table in sdata.tables.values():
            table.obs_names_make_unique()
    return sdata


def _filter_table_in_coordinate_systems(table: AnnData, coordinate_systems: list[str]) -> AnnData:
    table_mapping_metadata = table.uns[TableModel.ATTRS_KEY]
    region_key = table_mapping_metadata[TableModel.REGION_KEY_KEY]
    new_table = table[table.obs[region_key].isin(coordinate_systems)].copy()
    new_table.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY] = new_table.obs[region_key].unique().tolist()
    return new_table


def _fix_ensure_unique_element_names(
    sdatas: dict[str, SpatialData],
    rename_tables: bool,
    rename_obs_names: bool,
    modify_tables_inplace: bool,
    merge_coordinate_systems_on_name: bool,
) -> list[SpatialData]:
    sdatas_fixed = []
    for suffix, sdata in sdatas.items():
        # Create new elements dictionary with suffixed names
        elements = {}
        for _, name, el in sdata.gen_spatial_elements():
            new_element_name = f"{name}-{suffix}"
            if not merge_coordinate_systems_on_name:
                # Set new transformations with suffixed coordinate system names
                transformations = get_transformation(el, get_all=True)
                assert isinstance(transformations, dict)

                remove_transformation(el, remove_all=True)
                for cs, t in transformations.items():
                    new_cs = f"{cs}-{suffix}"
                    set_transformation(el, t, to_coordinate_system=new_cs)

            elements[new_element_name] = el

        # Handle tables with suffix
        tables = {}
        for name, table in sdata.tables.items():
            if not modify_tables_inplace:
                table = table.copy()

            # fix the region_key column
            region, region_key, _ = get_table_keys(table)
            table.obs[region_key] = (table.obs[region_key].astype("str") + f"-{suffix}").astype("category")
            new_region: str | list[str]
            if isinstance(region, str):
                new_region = f"{region}-{suffix}"
            else:
                assert isinstance(region, list)
                new_region = [f"{r}-{suffix}" for r in region]
            table.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY] = new_region

            # fix the obs names
            if rename_obs_names:
                table.obs.index = table.obs.index.to_series().apply(lambda x, suffix=suffix: f"{x}-{suffix}")

            # fix the table name
            new_name = f"{name}-{suffix}" if rename_tables else name
            tables[new_name] = table

        # Create new SpatialData object with suffixed elements and tables
        sdata_fixed = SpatialData.init_from_elements(elements | tables)
        sdatas_fixed.append(sdata_fixed)
    return sdatas_fixed
