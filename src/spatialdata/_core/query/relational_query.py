import math
import warnings
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from functools import partial, singledispatch
from typing import Any, Literal

import dask.array as da
import numpy as np
import pandas as pd
from anndata import AnnData
from annsel.core.typing import Predicates
from dask.dataframe import DataFrame as DaskDataFrame
from geopandas import GeoDataFrame
from xarray import DataArray, DataTree

from spatialdata._core.spatialdata import SpatialData
from spatialdata._types import ArrayLike
from spatialdata._utils import _inplace_fix_subset_categorical_obs
from spatialdata.models import (
    Image2DModel,
    Image3DModel,
    Labels2DModel,
    Labels3DModel,
    PointsModel,
    ShapesModel,
    SpatialElement,
    TableModel,
    get_model,
    get_table_keys,
)


def get_element_annotators(sdata: SpatialData, element_name: str) -> set[str]:
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


@singledispatch
def get_element_instances(
    element: SpatialElement,
    return_background: bool = False,
) -> pd.Index:
    """
    Get the instances (index values) of the SpatialElement.

    Parameters
    ----------
    element
        The SpatialElement.
    return_background
        If True, the background label (0) is included in the output.

    Returns
    -------
    pd.Series with the instances (index values) of the SpatialElement.
    """
    raise ValueError(f"The object type {type(element)} is not supported.")


@get_element_instances.register(DataArray)
@get_element_instances.register(DataTree)
def _(
    element: DataArray | DataTree,
    return_background: bool = False,
) -> pd.Index:
    model = get_model(element)
    assert model in [Labels2DModel, Labels3DModel], "Expected a `Labels` element. Found an `Image` instead."
    if isinstance(element, DataArray):
        # get unique labels value (including 0 if present)
        instances = da.unique(element.data).compute()
    else:
        assert isinstance(element, DataTree)
        v = element["scale0"].values()
        assert len(v) == 1
        xdata = next(iter(v))
        # can be slow
        instances = da.unique(xdata.data).compute()
    index = pd.Index(np.sort(instances))
    if not return_background and 0 in index:
        return index.drop(0)  # drop the background label
    return index


@get_element_instances.register(GeoDataFrame)
def _(
    element: GeoDataFrame,
) -> pd.Index:
    return element.index


@get_element_instances.register(DaskDataFrame)
def _(
    element: DaskDataFrame,
) -> pd.Index:
    return element.index


# TODO: replace function use throughout repo by `join_sdata_spatialelement_table`
# TODO: benchmark against join operations before removing
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
    assert any(len(elements) > 0 for elements in elements_dict.values()), (
        "elements_dict must contain at least one dict which contains at least one element"
    )
    if table is None:
        return None
    to_keep = np.zeros(len(table), dtype=bool)
    region_key = table.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY_KEY]
    instance_key = table.uns[TableModel.ATTRS_KEY][TableModel.INSTANCE_KEY]
    instances = None
    for _, elements in elements_dict.items():
        for name, element in elements.items():
            if get_model(element) == Labels2DModel or get_model(element) == Labels3DModel:
                if isinstance(element, DataArray):
                    # get unique labels value (including 0 if present)
                    instances = da.unique(element.data).compute()
                else:
                    assert isinstance(element, DataTree)
                    v = element["scale0"].values()
                    assert len(v) == 1
                    xdata = next(iter(v))
                    # can be slow
                    instances = da.unique(xdata.data).compute()
                instances = np.sort(instances)
            elif get_model(element) == ShapesModel:
                instances = element.index.to_numpy()
            elif get_model(element) == PointsModel:
                instances = element.compute().index.to_numpy()
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
        assert sorted(set(instances.tolist())) == sorted(set(table.obs[instance_key].tolist()))  # type: ignore[type-var]
        table_df = pd.DataFrame({instance_key: table.obs[instance_key], "position": np.arange(len(instances))})
        merged = pd.merge(table_df, pd.DataFrame(index=instances), left_on=instance_key, right_index=True, how="right")
        matched_positions = merged["position"].to_numpy()
        table = table[matched_positions, :]
    table = table.copy()
    _inplace_fix_subset_categorical_obs(subset_adata=table, original_adata=original_table)
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
            _, joined_indices = _match_rows(table_instance_key_column, mask, element_indices, match_rows)
        else:
            joined_indices = table_instance_key_column[mask].index
    else:
        if match_rows == "left":
            _, add_indices = _match_rows(table_instance_key_column, mask, element_indices, match_rows)
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
    if match_rows in ["left", "right"]:
        left_index, _ = _match_rows(table_instance_key_column, mask, element_indices, match_rows)

        if mask_values is not None and len(left_index) != len(mask_values):
            mask = left_index.isin(mask_values)
            mask_values = left_index[mask]
        else:
            mask_values = left_index

    if isinstance(element, DaskDataFrame):
        return element.map_partitions(lambda df: df.loc[mask_values], meta=element)
    return element.loc[mask_values, :]


def _right_exclusive_join_spatialelement_table(
    element_dict: dict[str, dict[str, Any]], table: AnnData, match_rows: Literal["left", "no", "right"]
) -> tuple[dict[str, Any], AnnData | None]:
    regions, region_column_name, instance_key = get_table_keys(table)
    if isinstance(regions, str):
        regions = [regions]
    groups_df = table.obs.groupby(by=region_column_name, observed=False)
    mask = []
    for element_type, name_element in element_dict.items():
        for name, element in name_element.items():
            if name in regions:
                group_df = groups_df.get_group(name)
                table_instance_key_column = group_df[instance_key]
                if element_type in ["points", "shapes"]:
                    element_indices = element.index
                else:
                    element_indices = get_element_instances(element)

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

    _inplace_fix_subset_categorical_obs(subset_adata=exclusive_table, original_adata=table)
    return element_dict, exclusive_table


def _right_join_spatialelement_table(
    element_dict: dict[str, dict[str, Any]], table: AnnData, match_rows: Literal["left", "no", "right"]
) -> tuple[dict[str, Any], AnnData]:
    if match_rows == "left":
        warnings.warn("Matching rows 'left' is not supported for 'right' join.", UserWarning, stacklevel=2)
    regions, region_column_name, instance_key = get_table_keys(table)
    if isinstance(regions, str):
        regions = [regions]
    groups_df = table.obs.groupby(by=region_column_name, observed=False)
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
    if isinstance(regions, str):
        regions = [regions]
    obs = table.obs.reset_index()
    groups_df = obs.groupby(by=region_column_name, observed=False)
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
                    joined_indices, masked_element.index, table_instance_key_column, match_rows
                )
            else:
                warnings.warn(
                    f"The element `{name}` is not annotated by the table. Skipping", UserWarning, stacklevel=2
                )
                element_dict[element_type][name] = None
                continue

    if joined_indices is not None:
        joined_indices = joined_indices.dropna() if any(joined_indices.isna()) else joined_indices

    joined_table = table[joined_indices.tolist(), :].copy() if joined_indices is not None else None

    _inplace_fix_subset_categorical_obs(subset_adata=joined_table, original_adata=table)
    return element_dict, joined_table


def _left_exclusive_join_spatialelement_table(
    element_dict: dict[str, dict[str, Any]], table: AnnData, match_rows: Literal["left", "no", "right"]
) -> tuple[dict[str, Any], AnnData | None]:
    regions, region_column_name, instance_key = get_table_keys(table)
    if isinstance(regions, str):
        regions = [regions]
    groups_df = table.obs.groupby(by=region_column_name, observed=False)
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
    if isinstance(regions, str):
        regions = [regions]
    obs = table.obs.reset_index()
    groups_df = obs.groupby(by=region_column_name, observed=False)
    joined_indices = None
    for element_type, name_element in element_dict.items():
        for name, element in name_element.items():
            if name in regions:
                group_df = groups_df.get_group(name)
                table_instance_key_column = group_df[instance_key]  # This is always a series
                if element_type in ["points", "shapes"]:
                    element_indices = element.index
                else:
                    element_indices = get_element_instances(element)

                joined_indices = _get_joined_table_indices(
                    joined_indices, element_indices, table_instance_key_column, match_rows
                )
            else:
                warnings.warn(
                    f"The element `{name}` is not annotated by the table. Skipping", UserWarning, stacklevel=2
                )
                continue

    if joined_indices is not None:
        joined_indices = joined_indices.dropna()
        # if nan were present, the dtype would have been changed to float
        if joined_indices.dtype == float:
            joined_indices = joined_indices.astype(int)
    joined_table = table[joined_indices.tolist(), :].copy() if joined_indices is not None else None
    _inplace_fix_subset_categorical_obs(subset_adata=joined_table, original_adata=table)

    return element_dict, joined_table


def _match_rows(
    table_instance_key_column: pd.Series,
    mask: pd.Series,
    element_indices: pd.RangeIndex,
    match_rows: str,
) -> tuple[pd.Index, pd.Index]:
    instance_id_df = pd.DataFrame(
        {"instance_id": table_instance_key_column[mask].values, "index_right": table_instance_key_column[mask].index}
    )
    element_index_df = pd.DataFrame({"index_left": element_indices})

    merged_df = pd.merge(element_index_df, instance_id_df, left_on="index_left", right_on="instance_id", how=match_rows)
    index_left = merged_df["index_left"]
    index_right = merged_df["index_right"]

    # With labels it can be that index 0 is NaN
    if isinstance(index_left.iloc[0], float) and math.isnan(index_left.iloc[0]):
        index_left = index_left.iloc[1:]

    if isinstance(index_right.iloc[0], float) and math.isnan(index_right.iloc[0]):
        index_right = index_right.iloc[1:]

    return pd.Index(index_left), pd.Index(index_right)


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


def _create_sdata_elements_dict_for_join(
    sdata: SpatialData, spatial_element_name: str | list[str]
) -> dict[str, dict[str, Any]]:
    elements_dict: dict[str, dict[str, Any]] = defaultdict(lambda: defaultdict(dict))
    for name in spatial_element_name:
        element_type, _, element = sdata._find_element(name)
        elements_dict[element_type][name] = element
    return elements_dict


def _validate_element_types_for_join(
    sdata: SpatialData | None,
    spatial_element_names: list[str],
    spatial_elements: list[SpatialElement] | None,
    table: AnnData | None,
) -> None:
    if sdata is not None:
        elements_to_check = []
        for name in spatial_element_names:
            elements_to_check.append(sdata[name])
    else:
        assert spatial_elements is not None
        elements_to_check = spatial_elements

    for element in elements_to_check:
        model = get_model(element)
        if model in [Image2DModel, Image3DModel, TableModel]:
            raise ValueError(f"Element type `{model}` not supported for join operation.")


def join_spatialelement_table(
    sdata: SpatialData | None = None,
    spatial_element_names: str | list[str] | None = None,
    spatial_elements: SpatialElement | list[SpatialElement] | None = None,
    table_name: str | None = None,
    table: AnnData | None = None,
    how: Literal["left", "left_exclusive", "inner", "right", "right_exclusive"] = "left",
    match_rows: Literal["no", "left", "right"] = "no",
) -> tuple[dict[str, Any], AnnData]:
    """
    Join SpatialElement(s) and table together in SQL like manner.

    The function allows the user to perform SQL like joins of SpatialElements and a table. The elements are not
    returned together in one dataframe-like structure, but instead filtered elements are returned. To determine matches,
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
    For Labels, the background label (0) is not included in the output and it will not be returned.

    Parameters
    ----------
    sdata
        SpatialData object containing all the elements and tables. This parameter can be `None`; in such case the both
        the names and values for the elements and the table must be provided.
    spatial_element_names
        Required. The name(s) of the spatial elements to be joined with the table. If a list of names, and if sdata is
         `None`, the indices must match with the list of SpatialElements passed on by the argument elements.
    spatial_elements
        This parameter should be speficied exactly when `sdata` is `None`. The SpatialElement(s) to be joined with the
        table. In case of a list of SpatialElements the indices must match exactly with the indices in the list of
        `spatial_element_name`.
    table_name
        The name of the table to join with the spatial elements. Optional, `table` can be provided instead.
    table
        The table to join with the spatial elements. When `sdata` is not `None`, `table_name` can be used instead.
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
    ValueError
        If `spatial_element_names` is not provided.
    ValueError
        If sdata is `None` but `spatial_elements` is not `None`; if `sdata` is not `None`, but `spatial_elements` is
        `None`.
    ValueError
        If `table_name` is provided but not present in the `SpatialData` object, or if `table_name` is provided but
        `sdata` is `None`.
    ValueError
        If not exactly one of `table_name` and `table` is provided.
    ValueError
        If no valid elements are provided for the join operation.
    ValueError
        If the provided join type is not supported.
    ValueError
        If an incorrect value is given for `match_rows`.

    Notes
    -----
    For a graphical representation of the join operations, see the
    `Tables tutorial <https://spatialdata.scverse.org/en/stable/tutorials/notebooks/notebooks/examples/tables.html>`_.

    See Also
    --------
    match_element_to_table : Function to match elements to a table.
    join_spatialelement_table : Function to join spatial elements with a table.
    """
    if spatial_element_names is None:
        raise ValueError("`spatial_element_names` must be provided.")
    if sdata is None and (spatial_elements is None or table is None):
        raise ValueError("If `sdata` is not provided, both `spatial_elements` and `table` must be provided.")
    if sdata is not None and (spatial_elements is not None):
        raise ValueError(
            "If `sdata` is provided, `spatial_elements` must not be provided; use `spatial_elements_name` instead."
        )
    if table is None and table_name is None or table is not None and table_name is not None:
        raise ValueError("Exactly one of `table_name` and `table` must be provided.")
    if sdata is not None and table_name is not None:
        if table_name not in sdata.tables:
            raise ValueError(f"No table with name `{table_name}` found in the SpatialData object.")
        table = sdata[table_name]
    spatial_element_names = (
        spatial_element_names if isinstance(spatial_element_names, list) else [spatial_element_names]
    )
    spatial_elements = spatial_elements if isinstance(spatial_elements, list) else [spatial_elements]
    _validate_element_types_for_join(sdata, spatial_element_names, spatial_elements, table)

    elements_dict: dict[str, dict[str, Any]]
    if sdata is not None:
        elements_dict = _create_sdata_elements_dict_for_join(sdata, spatial_element_names)
    else:
        derived_sdata = SpatialData.init_from_elements(dict(zip(spatial_element_names, spatial_elements, strict=True)))
        element_types = ["labels", "shapes", "points"]
        elements_dict = defaultdict(lambda: defaultdict(dict))
        for element_type in element_types:
            for name, element in getattr(derived_sdata, element_type).items():
                elements_dict[element_type][name] = element

    elements_dict_joined, table = _call_join(elements_dict, table, how, match_rows)
    return elements_dict_joined, table


def _call_join(
    elements_dict: dict[str, dict[str, Any]], table: AnnData, how: str, match_rows: Literal["no", "left", "right"]
) -> tuple[dict[str, Any], AnnData]:
    assert any(key in elements_dict for key in ["labels", "shapes", "points"]), (
        "No valid element to join in spatial_element_name. Must provide at least one of either `labels`, `points` or "
        "`shapes`."
    )

    if match_rows not in MatchTypes.__dict__["_member_names_"]:
        raise TypeError(
            f"`{match_rows}` is an invalid argument for `match_rows`. Can be either `no`, ``'left'`` or ``'right'``"
        )
    # bug with Python 3.13 (https://github.com/scverse/spatialdata/issues/852)
    # if how in JoinTypes.__dict__["_member_names_"]:
    # hotfix for bug with Python 3.13:
    if how in JoinTypes.__dict__:
        elements_dict, table = getattr(JoinTypes, how)(elements_dict, table, match_rows)
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

    Notes
    -----
    For a graphical representation of the join operations, see the
    `Tables tutorial <https://spatialdata.scverse.org/en/stable/tutorials/notebooks/notebooks/examples/tables.html>`_.

    See Also
    --------
    match_element_to_table : Function to match a spatial element to a table.
    join_spatialelement_table : General function, to join spatial elements with a table with more control.
    """
    _, table = join_spatialelement_table(
        sdata=sdata, spatial_element_names=element_name, table_name=table_name, how="left", match_rows="left"
    )
    return table


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

    Notes
    -----
    For a graphical representation of the join operations, see the
    `Tables tutorial <https://spatialdata.scverse.org/en/stable/tutorials/notebooks/notebooks/examples/tables.html>`_.

    See Also
    --------
    match_table_to_element : Function to match a table to a spatial element.
    join_spatialelement_table : General function, to join spatial elements with a table with more control.
    """
    element_dict, table = join_spatialelement_table(
        sdata=sdata, spatial_element_names=element_name, table_name=table_name, how="right", match_rows="right"
    )
    return element_dict, table


def match_sdata_to_table(
    sdata: SpatialData,
    table_name: str,
    table: AnnData | None = None,
    how: Literal["left", "left_exclusive", "inner", "right", "right_exclusive"] = "right",
) -> SpatialData:
    """
    Filter the elements of a SpatialData object to match only the rows present in the table.

    Parameters
    ----------
    sdata
        SpatialData object containing all the elements and tables.
    table
        The table to join with the spatial elements. Has precedence over `table_name`.
    table_name
        The name of the table to join with the SpatialData object if `table` is not provided. If table is provided,
        `table_name` is used to name the table in the returned `SpatialData` object.
    how
        The type of join to perform. See :func:`spatialdata.join_spatialelement_table`. Default is "right".

    Notes
    -----
    For a graphical representation of the join operations, see the
    `Tables tutorial <https://spatialdata.scverse.org/en/stable/tutorials/notebooks/notebooks/examples/tables.html>`_.
    """
    if table is None:
        table = sdata[table_name]
    _, region_key, instance_key = get_table_keys(table)
    annotated_regions = SpatialData.get_annotated_regions(table)
    filtered_elements, filtered_table = join_spatialelement_table(
        sdata, spatial_element_names=annotated_regions, table=table, how=how
    )
    filtered_table = TableModel.parse(
        filtered_table,
        region=annotated_regions,
        region_key=region_key,
        instance_key=instance_key,
        overwrite_metadata=True,
    )
    return SpatialData.init_from_elements(filtered_elements | {table_name: filtered_table})


def filter_by_table_query(
    sdata: SpatialData,
    table_name: str,
    filter_tables: bool = True,
    element_names: list[str] | None = None,
    obs_expr: Predicates | None = None,
    var_expr: Predicates | None = None,
    x_expr: Predicates | None = None,
    obs_names_expr: Predicates | None = None,
    var_names_expr: Predicates | None = None,
    layer: str | None = None,
    how: Literal["left", "left_exclusive", "inner", "right", "right_exclusive"] = "right",
) -> SpatialData:
    """Filter the SpatialData object based on a set of table queries.

    Parameters
    ----------
    sdata
        The SpatialData object to filter.
    table_name
        The name of the table to filter the SpatialData object by.
    filter_tables
        If True (default), the table is filtered to only contain rows that are annotating regions
        contained within the element_names.
    element_names
        The names of the elements to filter the SpatialData object by.
    obs_expr
        A Predicate or an iterable of `annsel` `Predicates` to filter :attr:`anndata.AnnData.obs` by.
    var_expr
        A Predicate or an iterable of `annsel` `Predicates` to filter :attr:`anndata.AnnData.var` by.
    x_expr
        A Predicate or an iterable of `annsel` `Predicates` to filter :attr:`anndata.AnnData.X` by.
    obs_names_expr
        A Predicate or an iterable of `annsel` `Predicates` to filter :attr:`anndata.AnnData.obs_names` by.
    var_names_expr
        A Predicate or an iterable of `annsel` `Predicates` to filter :attr:`anndata.AnnData.var_names` by.
    layer
        The layer of the :class:`anndata.AnnData` to filter the SpatialData object by, only used with `x_expr`.
    how
        The type of join to perform. See :func:`spatialdata.join_spatialelement_table`. Default is "right".

    Returns
    -------
        The filtered SpatialData object.

    Notes
    -----
    You can also use :func:`spatialdata.SpatialData.filter_by_table_query` with the convenience that `sdata` is the
    current `SpatialData` object.

    For a graphical representation of the join operations, see the
    `Tables tutorial <https://spatialdata.scverse.org/en/stable/tutorials/notebooks/notebooks/examples/tables.html>`_.

    For more examples on table queries, see the
    `Table queries tutorial <https://spatialdata.scverse.org/en/stable/tutorials/notebooks/notebooks/examples/table_queries.html>`_.
    """
    sdata_subset: SpatialData = (
        sdata.subset(element_names=element_names, filter_tables=filter_tables) if element_names else sdata
    )

    filtered_table: AnnData = sdata_subset.tables[table_name].an.filter(
        obs=obs_expr, var=var_expr, x=x_expr, obs_names=obs_names_expr, var_names=var_names_expr, layer=layer
    )

    return match_sdata_to_table(sdata=sdata_subset, table_name=table_name, table=filtered_table, how=how)


@dataclass
class _ValueOrigin:
    origin: str
    is_categorical: bool
    value_key: str


def _get_element(
    element: SpatialElement | AnnData | None, sdata: SpatialData | None, element_name: str | None
) -> SpatialElement | AnnData:
    if element is None:
        assert sdata is not None
        assert element_name is not None
        return sdata[element_name]
    assert sdata is None
    if not isinstance(element, AnnData):
        assert element_name is None
    return element


def _get_table_origins(
    element: SpatialElement | AnnData, value_key: str, origins: list[_ValueOrigin]
) -> list[_ValueOrigin]:
    if value_key in element.obs.columns:
        value = element.obs[value_key]
        is_categorical = isinstance(value.dtype, pd.CategoricalDtype)
        origins.append(_ValueOrigin(origin="obs", is_categorical=is_categorical, value_key=value_key))
    # check if the value_key is in the var
    elif value_key in element.var_names:
        origins.append(_ValueOrigin(origin="var", is_categorical=False, value_key=value_key))
    elif value_key in element.obsm:
        origins.append(_ValueOrigin(origin="obsm", is_categorical=False, value_key=value_key))
    return origins


def _locate_value(
    value_key: str,
    element: SpatialElement | AnnData | None = None,
    sdata: SpatialData | None = None,
    element_name: str | None = None,
    table_name: str | None = None,
) -> list[_ValueOrigin]:
    el = _get_element(element=element, sdata=sdata, element_name=element_name)
    origins = []
    model = get_model(el)
    if model not in [PointsModel, ShapesModel, Labels2DModel, Labels3DModel, TableModel]:
        raise ValueError(f"Cannot get value from {model}")
    # adding from the dataframe columns
    if model in [PointsModel, ShapesModel] and value_key in el.columns:
        value = el[value_key]
        is_categorical = isinstance(value.dtype, pd.CategoricalDtype)
        origins.append(_ValueOrigin(origin="df", is_categorical=is_categorical, value_key=value_key))
    if model == TableModel:
        origins = _get_table_origins(element=el, value_key=value_key, origins=origins)

    # adding from the obs columns or var
    if model in [PointsModel, ShapesModel, Labels2DModel, Labels3DModel] and sdata is not None:
        table = sdata.tables.get(table_name) if table_name is not None else None
        if table is not None:
            # check if the table is annotating the element
            region = table.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY]
            if element_name in region:
                # check if the value_key is in the table
                origins = _get_table_origins(element=table, value_key=value_key, origins=origins)

    return origins


def get_values(
    value_key: str | list[str],
    element: SpatialElement | AnnData | None = None,
    sdata: SpatialData | None = None,
    element_name: str | None = None,
    table_name: str | None = None,
    table_layer: str | None = None,
    return_obsm_as_is: bool = False,
) -> pd.DataFrame | ArrayLike:
    """
    Get the values from the element, from any location: df columns, obs or var columns (table).

    Parameters
    ----------
    value_key
        Name of the column/channel name to get the values from
    element
        SpatialElement object or AnnData table; either element or (sdata, element_name) must be provided
    sdata
        SpatialData object; either element or (sdata, element_name) must be provided
    element_name
        Name of the element; either element or (sdata, element_name) must be provided. In case of element being
        an AnnData table, element_name can also be provided to subset the AnnData table to only include those rows
        annotating the element_name.
    table_name
        Name of the table to get the values from.
    table_layer
        Layer of the table to get the values from. If None, the values are taken from X.
    return_obsm_as_is
        In case the value is in obsm the value of the key can be returned as is if return_obsm_as_is is True, otherwise
        creates a dataframe and returns it.

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
    if (sdata is not None and table_name is not None) or isinstance(element, AnnData):
        if sdata is not None and table_name is not None:
            assert element_name is not None
            matched_table = match_table_to_element(sdata=sdata, element_name=element_name, table_name=table_name)
            region_key = matched_table.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY_KEY]
            instance_key = matched_table.uns[TableModel.ATTRS_KEY][TableModel.INSTANCE_KEY]
            obs = matched_table.obs
            assert obs[region_key].nunique() == 1
            assert obs[instance_key].nunique() == len(matched_table)
        else:
            assert isinstance(element, AnnData)
            matched_table = element
            instance_key = matched_table.uns[TableModel.ATTRS_KEY][TableModel.INSTANCE_KEY]
            region_key = matched_table.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY_KEY]
            if element_name is not None:
                matched_table = matched_table[matched_table.obs[region_key] == element_name]
            obs = matched_table.obs

        if origin == "obs":
            df = obs[value_key_values].copy()
        if origin == "var":
            matched_table.obs = pd.DataFrame(obs)
            if table_layer is None:
                x = matched_table[:, value_key_values].X
            else:
                if table_layer not in matched_table.layers:
                    raise ValueError(f"Layer {table_layer} was not found.")
                x = matched_table[:, value_key_values].layers[table_layer]
            import scipy

            if isinstance(x, scipy.sparse.csr_matrix | scipy.sparse.csc_matrix | scipy.sparse.coo_matrix):
                x = x.todense()
            df = pd.DataFrame(x, columns=value_key_values)
        if origin == "obsm":
            data = {}
            for key in value_key_values:
                data_values = matched_table.obsm[key]
                if len(value_key_values) == 1 and return_obsm_as_is:
                    return data_values
                if len(value_key_values) > 1 and return_obsm_as_is:
                    warnings.warn(
                        "Multiple value_keys are specified. If you want to return an array only 1 should be specified",
                        UserWarning,
                        stacklevel=2,
                    )
                for i in range(data_values.shape[1]):
                    data[key + f"_{i}"] = data_values[:, i]
            df = pd.DataFrame(data)
        df.index = obs[instance_key]
        return df

    raise ValueError(f"Unknown origin {origin}")
