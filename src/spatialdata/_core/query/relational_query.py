from __future__ import annotations

import math
import warnings
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from functools import partial
from typing import Any, Literal

import dask.array as da
import numpy as np
import pandas as pd
from anndata import AnnData
from dask.dataframe.core import DataFrame as DaskDataFrame
from multiscale_spatial_image import MultiscaleSpatialImage
from spatial_image import SpatialImage

from spatialdata._core.spatialdata import SpatialData
from spatialdata._utils import _inplace_fix_subset_categorical_obs
from spatialdata.models import (
    Labels2DModel,
    Labels3DModel,
    PointsModel,
    ShapesModel,
    SpatialElement,
    TableModel,
    get_model,
    get_table_keys,
)


def _get_element_annotators(sdata: SpatialData, element_name: str) -> set[str]:
    """
    Retrieve names of tables that annotate a SpatialElement in a SpatialData object.

    Parameters
    ----------
    sdata
        SpatialData object.
    element_name
        The name of the SpatialElement.

    Returns
    -------
    The names of the tables annotating the SpatialElement.
    """
    table_names = set()
    for name, table in sdata.tables.items():
        if table.uns.get(TableModel.ATTRS_KEY):
            regions, _, _ = get_table_keys(table)
            if element_name in regions:
                table_names.add(name)
    return table_names


def _filter_table_by_element_names(table: AnnData | None, element_names: str | list[str]) -> AnnData | None:
    """
    Filter an AnnData table to keep only the rows that are in the coordinate system.

    Parameters
    ----------
    table
        The table to filter; if None, returns None
    element_names
        The element_names to keep in the tables obs.region column

    Returns
    -------
    The filtered table, or None if the input table was None
    """
    if table is None or not table.uns.get(TableModel.ATTRS_KEY):
        return None
    table_mapping_metadata = table.uns[TableModel.ATTRS_KEY]
    region_key = table_mapping_metadata[TableModel.REGION_KEY_KEY]
    table.obs = pd.DataFrame(table.obs)
    table = table[table.obs[region_key].isin(element_names)].copy()
    table.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY] = table.obs[region_key].unique().tolist()
    return table


def _get_unique_label_values_as_index(element: SpatialElement) -> pd.Index:
    if isinstance(element, SpatialImage):
        # get unique labels value (including 0 if present)
        instances = da.unique(element.data).compute()
    else:
        assert isinstance(element, MultiscaleSpatialImage)
        v = element["scale0"].values()
        assert len(v) == 1
        xdata = next(iter(v))
        # can be slow
        instances = da.unique(xdata.data).compute()
    return pd.Index(np.sort(instances))


# TODO: replace function use throughout repo by `join_sdata_spatialelement_table`
def _filter_table_by_elements(
    table: AnnData | None, elements_dict: dict[str, dict[str, Any]], match_rows: bool = False
) -> AnnData | None:
    """
    Filter an AnnData table to keep only the rows that are in the elements.

    Parameters
    ----------
    table
        The table to filter; if None, returns None
    elements_dict
        The elements to use to filter the table
    match_rows
        If True, reorder the table rows to match the order of the elements

    Returns
    -------
    The filtered table (eventually with reordered rows), or None if the input table was None.
    """
    assert set(elements_dict.keys()).issubset({"images", "labels", "shapes", "points"})
    assert len(elements_dict) > 0, "elements_dict must not be empty"
    assert any(
        len(elements) > 0 for elements in elements_dict.values()
    ), "elements_dict must contain at least one dict which contains at least one element"
    if table is None:
        return None
    to_keep = np.zeros(len(table), dtype=bool)
    region_key = table.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY_KEY]
    instance_key = table.uns[TableModel.ATTRS_KEY][TableModel.INSTANCE_KEY]
    instances = None
    for _, elements in elements_dict.items():
        for name, element in elements.items():
            if get_model(element) == Labels2DModel or get_model(element) == Labels3DModel:
                if isinstance(element, SpatialImage):
                    # get unique labels value (including 0 if present)
                    instances = da.unique(element.data).compute()
                else:
                    assert isinstance(element, MultiscaleSpatialImage)
                    v = element["scale0"].values()
                    assert len(v) == 1
                    xdata = next(iter(v))
                    # can be slow
                    instances = da.unique(xdata.data).compute()
                instances = np.sort(instances)
            elif get_model(element) == ShapesModel:
                instances = element.index.to_numpy()
            else:
                continue
            indices = ((table.obs[region_key] == name) & (table.obs[instance_key].isin(instances))).to_numpy()
            to_keep = to_keep | indices
    original_table = table
    table.obs = pd.DataFrame(table.obs)
    table = table[to_keep, :]
    if match_rows:
        assert instances is not None
        assert isinstance(instances, np.ndarray)
        assert np.sum(to_keep) != 0, "No row matches in the table annotates the element"
        if np.sum(to_keep) != len(instances):
            if len(elements_dict) > 1 or len(elements_dict) == 1 and len(next(iter(elements_dict.values()))) > 1:
                raise NotImplementedError("Sorting is not supported when filtering by multiple elements")
            # case in which the instances in the table and the instances in the element don't correspond
            assert "element" in locals()
            assert "name" in locals()
            n0 = np.setdiff1d(instances, table.obs[instance_key].to_numpy())
            n1 = np.setdiff1d(table.obs[instance_key].to_numpy(), instances)
            assert len(n1) == 0, f"The table contains {len(n1)} instances that are not in the element: {n1}"
            # some instances have not a corresponding row in the table
            instances = np.setdiff1d(instances, n0)
        assert np.sum(to_keep) == len(instances)
        assert sorted(set(instances.tolist())) == sorted(set(table.obs[instance_key].tolist()))
        table_df = pd.DataFrame({instance_key: table.obs[instance_key], "position": np.arange(len(instances))})
        merged = pd.merge(table_df, pd.DataFrame(index=instances), left_on=instance_key, right_index=True, how="right")
        matched_positions = merged["position"].to_numpy()
        table = table[matched_positions, :]
    _inplace_fix_subset_categorical_obs(subset_adata=table, original_adata=original_table)
    table = table.copy()
    table.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY] = table.obs[region_key].unique().tolist()
    return table


def _get_joined_table_indices(
    joined_indices: pd.Index | None,
    element_indices: pd.RangeIndex,
    table_instance_key_column: pd.Series,
    match_rows: Literal["left", "no", "right"],
) -> pd.Index:
    """
    Get indices of the table that are present in element_indices.

    Parameters
    ----------
    joined_indices
        Current indices that have been found to match indices of an element
    element_indices
        Element indices to match against table_instance_key_column.
    table_instance_key_column
        The column of a table containing the instance ids.
    match_rows
        Whether to match the indices of the element and table and if so how. If left, element_indices take priority and
        if right table instance ids take priority.

    Returns
    -------
        The indices that of the table that match the SpatialElement indices.
    """
    mask = table_instance_key_column.isin(element_indices)
    if joined_indices is None:
        if match_rows == "left":
            joined_indices = _match_rows(table_instance_key_column, mask, element_indices, match_rows)
        else:
            joined_indices = table_instance_key_column[mask].index
    else:
        if match_rows == "left":
            add_indices = _match_rows(table_instance_key_column, mask, element_indices, match_rows)
            joined_indices = joined_indices.append(add_indices)
        # in place append does not work with pd.Index
        else:
            joined_indices = joined_indices.append(table_instance_key_column[mask].index)
    return joined_indices


def _get_masked_element(
    element_indices: pd.RangeIndex,
    element: SpatialElement,
    table_instance_key_column: pd.Series,
    match_rows: Literal["left", "no", "right"],
) -> SpatialElement:
    """
    Get element rows matching the instance ids in the table_instance_key_column.

    Parameters
    ----------
    element_indices
        The indices of an element.
    element
        The spatial element to be masked.
    table_instance_key_column
        The column of a table containing the instance ids
    match_rows
         Whether to match the indices of the element and table and if so how. If left, element_indices take priority and
        if right table instance ids take priority.

    Returns
    -------
    The masked spatial element based on the provided indices and match rows.
    """
    mask = table_instance_key_column.isin(element_indices)
    masked_table_instance_key_column = table_instance_key_column[mask]
    mask_values = mask_values if len(mask_values := masked_table_instance_key_column.values) != 0 else None
    if match_rows == "right":
        mask_values = _match_rows(table_instance_key_column, mask, element_indices, match_rows)

    return element.loc[mask_values, :]


def _right_exclusive_join_spatialelement_table(
    element_dict: dict[str, dict[str, Any]], table: AnnData, match_rows: Literal["left", "no", "right"]
) -> tuple[dict[str, Any], AnnData | None]:
    regions, region_column_name, instance_key = get_table_keys(table)
    groups_df = table.obs.groupby(by=region_column_name)
    mask = []
    for element_type, name_element in element_dict.items():
        for name, element in name_element.items():
            if name in regions:
                group_df = groups_df.get_group(name)
                table_instance_key_column = group_df[instance_key]
                if element_type in ["points", "shapes"]:
                    element_indices = element.index
                else:
                    element_indices = _get_unique_label_values_as_index(element)

                element_dict[element_type][name] = None
                submask = ~table_instance_key_column.isin(element_indices)
                mask.append(submask)
            else:
                warnings.warn(
                    f"The element `{name}` is not annotated by the table. Skipping", UserWarning, stacklevel=2
                )
                element_dict[element_type][name] = None
                continue

    if len(mask) != 0:
        mask = pd.concat(mask)
        exclusive_table = table[mask, :].copy() if mask.sum() != 0 else None  # type: ignore[attr-defined]
    else:
        exclusive_table = None

    return element_dict, exclusive_table


def _right_join_spatialelement_table(
    element_dict: dict[str, dict[str, Any]], table: AnnData, match_rows: Literal["left", "no", "right"]
) -> tuple[dict[str, Any], AnnData]:
    if match_rows == "left":
        warnings.warn("Matching rows 'left' is not supported for 'right' join.", UserWarning, stacklevel=2)
    regions, region_column_name, instance_key = get_table_keys(table)
    groups_df = table.obs.groupby(by=region_column_name)
    for element_type, name_element in element_dict.items():
        for name, element in name_element.items():
            if name in regions:
                group_df = groups_df.get_group(name)
                table_instance_key_column = group_df[instance_key]
                if element_type in ["points", "shapes"]:
                    element_indices = element.index
                else:
                    warnings.warn(
                        f"Element type `labels` not supported for 'right' join. Skipping `{name}`",
                        UserWarning,
                        stacklevel=2,
                    )
                    continue

                masked_element = _get_masked_element(element_indices, element, table_instance_key_column, match_rows)
                element_dict[element_type][name] = masked_element
            else:
                warnings.warn(
                    f"The element `{name}` is not annotated by the table. Skipping", UserWarning, stacklevel=2
                )
                continue
    return element_dict, table


def _inner_join_spatialelement_table(
    element_dict: dict[str, dict[str, Any]], table: AnnData, match_rows: Literal["left", "no", "right"]
) -> tuple[dict[str, Any], AnnData]:
    regions, region_column_name, instance_key = get_table_keys(table)
    groups_df = table.obs.groupby(by=region_column_name)
    joined_indices = None
    for element_type, name_element in element_dict.items():
        for name, element in name_element.items():
            if name in regions:
                group_df = groups_df.get_group(name)
                table_instance_key_column = group_df[instance_key]  # This is always a series
                if element_type in ["points", "shapes"]:
                    element_indices = element.index
                else:
                    warnings.warn(
                        f"Element type `labels` not supported for 'inner' join. Skipping `{name}`",
                        UserWarning,
                        stacklevel=2,
                    )
                    continue

                masked_element = _get_masked_element(element_indices, element, table_instance_key_column, match_rows)
                element_dict[element_type][name] = masked_element

                joined_indices = _get_joined_table_indices(
                    joined_indices, element_indices, table_instance_key_column, match_rows
                )
            else:
                warnings.warn(
                    f"The element `{name}` is not annotated by the table. Skipping", UserWarning, stacklevel=2
                )
                element_dict[element_type][name] = None
                continue

    joined_table = table[joined_indices, :].copy() if joined_indices is not None else None
    return element_dict, joined_table


def _left_exclusive_join_spatialelement_table(
    element_dict: dict[str, dict[str, Any]], table: AnnData, match_rows: Literal["left", "no", "right"]
) -> tuple[dict[str, Any], AnnData | None]:
    regions, region_column_name, instance_key = get_table_keys(table)
    groups_df = table.obs.groupby(by=region_column_name)
    for element_type, name_element in element_dict.items():
        for name, element in name_element.items():
            if name in regions:
                group_df = groups_df.get_group(name)
                table_instance_key_column = group_df[instance_key]
                if element_type in ["points", "shapes"]:
                    mask = np.full(len(element), True, dtype=bool)
                    mask[table_instance_key_column.values] = False
                    masked_element = element.loc[mask, :] if mask.sum() != 0 else None
                    element_dict[element_type][name] = masked_element
                else:
                    warnings.warn(
                        f"Element type `labels` not supported for left exclusive join. Skipping `{name}`",
                        UserWarning,
                        stacklevel=2,
                    )
                    continue
            else:
                warnings.warn(
                    f"The element `{name}` is not annotated by the table. Skipping", UserWarning, stacklevel=2
                )
                continue

    return element_dict, None


def _left_join_spatialelement_table(
    element_dict: dict[str, dict[str, Any]], table: AnnData, match_rows: Literal["left", "no", "right"]
) -> tuple[dict[str, Any], AnnData]:
    if match_rows == "right":
        warnings.warn("Matching rows 'right' is not supported for 'left' join.", UserWarning, stacklevel=2)
    regions, region_column_name, instance_key = get_table_keys(table)
    groups_df = table.obs.groupby(by=region_column_name)
    joined_indices = None
    for element_type, name_element in element_dict.items():
        for name, element in name_element.items():
            if name in regions:
                group_df = groups_df.get_group(name)
                table_instance_key_column = group_df[instance_key]  # This is always a series
                if element_type in ["points", "shapes"]:
                    element_indices = element.index
                else:
                    element_indices = _get_unique_label_values_as_index(element)

                joined_indices = _get_joined_table_indices(
                    joined_indices, element_indices, table_instance_key_column, match_rows
                )
            else:
                warnings.warn(
                    f"The element `{name}` is not annotated by the table. Skipping", UserWarning, stacklevel=2
                )
                continue

    joined_indices = joined_indices.dropna() if joined_indices is not None else None
    joined_table = table[joined_indices, :].copy() if joined_indices is not None else None

    return element_dict, joined_table


def _match_rows(
    table_instance_key_column: pd.Series,
    mask: pd.Series,
    element_indices: pd.RangeIndex,
    match_rows: str,
) -> pd.Index:
    instance_id_df = pd.DataFrame(
        {"instance_id": table_instance_key_column[mask].values, "index_right": table_instance_key_column[mask].index}
    )
    element_index_df = pd.DataFrame({"index_left": element_indices})
    index_col = "index_left" if match_rows == "right" else "index_right"

    merged_df = pd.merge(
        element_index_df, instance_id_df, left_on="index_left", right_on="instance_id", how=match_rows
    )[index_col]

    # With labels it can be that index 0 is NaN
    if isinstance(merged_df.iloc[0], float) and math.isnan(merged_df.iloc[0]):
        merged_df = merged_df.iloc[1:]

    return pd.Index(merged_df)


class JoinTypes(Enum):
    """Available join types for matching elements to tables and vice versa."""

    left = partial(_left_join_spatialelement_table)
    left_exclusive = partial(_left_exclusive_join_spatialelement_table)
    inner = partial(_inner_join_spatialelement_table)
    right = partial(_right_join_spatialelement_table)
    right_exclusive = partial(_right_exclusive_join_spatialelement_table)

    def __call__(self, *args: Any) -> tuple[dict[str, Any], AnnData]:
        return self.value(*args)


class MatchTypes(Enum):
    """Available match types for matching rows of elements and tables."""

    left = "left"
    right = "right"
    no = "no"


def join_sdata_spatialelement_table(
    sdata: SpatialData,
    spatial_element_name: str | list[str],
    table_name: str,
    how: str = "left",
    match_rows: Literal["no", "left", "right"] = "no",
) -> tuple[dict[str, Any], AnnData]:
    """Join SpatialElement(s) and table together in SQL like manner.

    The function allows the user to perform SQL like joins of SpatialElements and a table. The elements are not
    returned together in one dataframe like structure, but instead filtered elements are returned. To determine matches,
    for the SpatialElement the index is used and for the table the region key column and instance key column. The
    elements are not overwritten in the `SpatialData` object.

    The following joins are supported: ``'left'``, ``'left_exclusive'``, ``'inner'``, ``'right'`` and
    ``'right_exclusive'``. In case of a ``'left'`` join the SpatialElements are returned in a dictionary as is
    while the table is filtered to only include matching rows. In case of ``'left_exclusive'`` join None is returned
    for table while the SpatialElements returned are filtered to only include indices not present in the table. The
    cases for ``'right'`` joins are symmetric to the ``'left'`` joins. In case of an ``'inner'`` join of
    SpatialElement(s) and a table, for each an element is returned only containing the rows that are present in
    both the SpatialElement and table.

    For Points and Shapes elements every valid join for argument how is supported. For Labels elements only
     the ``'left'`` and ``'right_exclusive'`` joins are supported.

    Parameters
    ----------
    sdata
        The SpatialData object containing the tables and spatial elements.
    spatial_element_name
        The name(s) of the spatial elements to be joined with the table.
    table_name
        The name of the table to join with the spatial elements.
    how
        The type of SQL like join to perform, default is ``'left'``. Options are ``'left'``, ``'left_exclusive'``,
        ``'inner'``, ``'right'`` and ``'right_exclusive'``.
    match_rows
        Whether to match the indices of the element and table and if so how. If ``'left'``, element_indices take
        priority and if ``'right'`` table instance ids take priority.

    Returns
    -------
    A tuple containing the joined elements as a dictionary and the joined table as an AnnData object.

    Raises
    ------
    AssertionError
        If no table with the given table_name exists in the SpatialData object.
    ValueError
        If the provided join type is not supported.
    """
    assert sdata.tables.get(table_name), f"No table with `{table_name}` exists in the SpatialData object."
    table = sdata.tables[table_name]
    if isinstance(spatial_element_name, str):
        spatial_element_name = [spatial_element_name]

    elements_dict: dict[str, dict[str, Any]] = defaultdict(lambda: defaultdict(dict))
    for name in spatial_element_name:
        if name in sdata.tables:
            warnings.warn(
                f"Tables: `{', '.join(elements_dict['tables'].keys())}` given in spatial_element_names cannot be "
                f"joined with a table using this function.",
                UserWarning,
                stacklevel=2,
            )
        elif name in sdata.images:
            warnings.warn(
                f"Images: `{', '.join(elements_dict['images'].keys())}` cannot be joined with a table",
                UserWarning,
                stacklevel=2,
            )
        else:
            element_type, _, element = sdata._find_element(name)
            elements_dict[element_type][name] = element

    assert any(key in elements_dict for key in ["labels", "shapes", "points"]), (
        "No valid element to join in spatial_element_name. Must provide at least one of either `labels`, `points` or "
        "`shapes`."
    )

    if match_rows not in MatchTypes.__dict__["_member_names_"]:
        raise TypeError(
            f"`{match_rows}` is an invalid argument for `match_rows`. Can be either `no`, ``'left'`` or ``'right'``"
        )
    if how in JoinTypes.__dict__["_member_names_"]:
        elements_dict, table = JoinTypes[how](elements_dict, table, match_rows)
    else:
        raise TypeError(f"`{how}` is not a valid type of join.")

    elements_dict = {
        name: element for outer_key, dict_val in elements_dict.items() for name, element in dict_val.items()
    }
    return elements_dict, table


def match_table_to_element(sdata: SpatialData, element_name: str, table_name: str = "table") -> AnnData:
    """
    Filter the table and reorders the rows to match the instances (rows/labels) of the specified SpatialElement.

    Parameters
    ----------
    sdata
        SpatialData object
    element_name
        The name of the spatial elements to be joined with the table.
    table_name
        The name of the table to match to the element.

    Returns
    -------
    Table with the rows matching the instances of the element
    """
    # TODO: refactor this to make use of the new join_sdata_spatialelement_table function.
    # if table_name is None:
    #     warnings.warn(
    #         "Assumption of table with name `table` being present is being deprecated in SpatialData v0.1. "
    #         "Please provide the name of the table as argument to table_name.",
    #         DeprecationWarning,
    #         stacklevel=2,
    #     )
    #     table_name = "table"
    # _, table = join_sdata_spatialelement_table(sdata, element_name, table_name, "left", match_rows="left")
    # return table
    assert sdata[table_name] is not None, "No table found in the SpatialData"
    element_type, _, element = sdata._find_element(element_name)
    assert element_type in ["labels", "shapes"], f"Element {element_name} ({element_type}) is not supported"
    elements_dict = {element_type: {element_name: element}}
    return _filter_table_by_elements(sdata[table_name], elements_dict, match_rows=True)


def match_element_to_table(
    sdata: SpatialData, element_name: str | list[str], table_name: str
) -> tuple[dict[str, Any], AnnData]:
    """
    Filter the elements and make the indices match those in the table.

    Parameters
    ----------
    sdata
       SpatialData object
    element_name
       The name(s) of the spatial elements to be joined with the table. Not supported for Label elements.
    table_name
       The name of the table to join with the spatial elements.

    Returns
    -------
    A tuple containing the joined elements as a dictionary and the joined table as an AnnData object.
    """
    element_dict, table = join_sdata_spatialelement_table(sdata, element_name, table_name, "right", match_rows="right")
    return element_dict, table


@dataclass
class _ValueOrigin:
    origin: str
    is_categorical: bool
    value_key: str


def _get_element(element: SpatialElement | None, sdata: SpatialData | None, element_name: str | None) -> SpatialElement:
    if element is None:
        assert sdata is not None
        assert element_name is not None
        return sdata[element_name]
    assert sdata is None
    assert element_name is None
    return element


def _locate_value(
    value_key: str,
    element: SpatialElement | None = None,
    sdata: SpatialData | None = None,
    element_name: str | None = None,
    table_name: str | None = None,
) -> list[_ValueOrigin]:
    el = _get_element(element=element, sdata=sdata, element_name=element_name)
    origins = []
    model = get_model(el)
    if model not in [PointsModel, ShapesModel, Labels2DModel, Labels3DModel]:
        raise ValueError(f"Cannot get value from {model}")
    # adding from the dataframe columns
    if model in [PointsModel, ShapesModel] and value_key in el.columns:
        value = el[value_key]
        is_categorical = pd.api.types.is_categorical_dtype(value)
        origins.append(_ValueOrigin(origin="df", is_categorical=is_categorical, value_key=value_key))

    # adding from the obs columns or var
    if model in [ShapesModel, Labels2DModel, Labels3DModel] and sdata is not None:
        table = sdata.tables.get(table_name) if table_name is not None else None
        if table is not None:
            # check if the table is annotating the element
            region = table.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY]
            if element_name in region:
                # check if the value_key is in the table
                if value_key in table.obs.columns:
                    value = table.obs[value_key]
                    is_categorical = pd.api.types.is_categorical_dtype(value)
                    origins.append(_ValueOrigin(origin="obs", is_categorical=is_categorical, value_key=value_key))
                # check if the value_key is in the var
                elif value_key in table.var_names:
                    origins.append(_ValueOrigin(origin="var", is_categorical=False, value_key=value_key))
    return origins


def get_values(
    value_key: str | list[str],
    element: SpatialElement | None = None,
    sdata: SpatialData | None = None,
    element_name: str | None = None,
    table_name: str | None = None,
) -> pd.DataFrame:
    """
    Get the values from the element, from any location: df columns, obs or var columns (table).

    Parameters
    ----------
    value_key
        Name of the column/channel name to get the values from
    element
        SpatialElement object; either element or (sdata, element_name) must be provided
    sdata
        SpatialData object; either element or (sdata, element_name) must be provided
    element_name
        Name of the element; either element or (sdata, element_name) must be provided
    table_name
        Name of the table to get the values from.

    Returns
    -------
    DataFrame with the values requested.

    Notes
    -----
    - The index of the returned dataframe is the instance_key of the table for the specified element.
    - If the element is a labels, the eventual background (0) is not included in the dataframe of returned values.
    """
    el = _get_element(element=element, sdata=sdata, element_name=element_name)
    value_keys = [value_key] if isinstance(value_key, str) else value_key
    locations = []
    for vk in value_keys:
        origins = _locate_value(
            value_key=vk, element=element, sdata=sdata, element_name=element_name, table_name=table_name
        )
        if len(origins) > 1:
            raise ValueError(
                f"{vk} has been found in multiple locations of (element, sdata, element_name) = "
                f"{(element, sdata, element_name)}: {origins}"
            )
        if len(origins) == 0:
            raise ValueError(
                f"{vk} has not been found in (element, sdata, element_name) = {(element, sdata, element_name)}"
            )
        locations.append(origins[0])
    categorical_values = {x.is_categorical for x in locations}
    origin_values = {x.origin for x in locations}
    value_key_values = [x.value_key for x in locations]
    if len(categorical_values) == 2:
        raise ValueError("Cannot mix categorical and non-categorical values. Please call aggregate() multiple times.")
    if len({x.origin for x in locations}) > 1 and categorical_values.__iter__().__next__() is True:
        raise ValueError(
            "Can only aggregate one categorical column at the time. Please call aggregate() multiple times."
        )
    if len(origin_values) > 1:
        raise ValueError(
            f"Cannot mix values from different origins: {origin_values}. Please call aggregate() multiple times."
        )
    origin = origin_values.__iter__().__next__()
    if origin == "df":
        df = el[value_key_values]
        if isinstance(el, DaskDataFrame):
            df = df.compute()
        return df
    if sdata is not None and table_name is not None:
        assert element_name is not None
        matched_table = match_table_to_element(sdata=sdata, element_name=element_name, table_name=table_name)
        region_key = matched_table.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY_KEY]
        instance_key = matched_table.uns[TableModel.ATTRS_KEY][TableModel.INSTANCE_KEY]
        obs = matched_table.obs
        assert obs[region_key].nunique() == 1
        assert obs[instance_key].nunique() == len(matched_table)
        if origin == "obs":
            df = obs[value_key_values].copy()
        if origin == "var":
            matched_table.obs = pd.DataFrame(obs)
            x = matched_table[:, value_key_values].X
            import scipy

            if isinstance(x, scipy.sparse.csr_matrix):
                x = x.todense()
            df = pd.DataFrame(x, columns=value_key_values)
        df.index = obs[instance_key]
        return df
    raise ValueError(f"Unknown origin {origin}")
