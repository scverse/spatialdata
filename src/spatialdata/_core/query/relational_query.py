from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import dask.array as da
import numpy as np
import pandas as pd
from anndata import AnnData
from dask.dataframe.core import DataFrame as DaskDataFrame
from multiscale_spatial_image import MultiscaleSpatialImage
from spatial_image import SpatialImage

from spatialdata._utils import _inplace_fix_subset_categorical_obs
from spatialdata.models import (
    Labels2DModel,
    Labels3DModel,
    PointsModel,
    ShapesModel,
    SpatialElement,
    TableModel,
    get_model,
)

if TYPE_CHECKING:
    from spatialdata import SpatialData


def _filter_table_by_coordinate_system(table: AnnData | None, coordinate_system: str | list[str]) -> AnnData | None:
    """
    Filter an AnnData table to keep only the rows that are in the coordinate system.

    Parameters
    ----------
    table
        The table to filter; if None, returns None
    coordinate_system
        The coordinate system to keep

    Returns
    -------
    The filtered table, or None if the input table was None
    """
    if table is None:
        return None
    table_mapping_metadata = table.uns[TableModel.ATTRS_KEY]
    region_key = table_mapping_metadata[TableModel.REGION_KEY_KEY]
    table = table[table.obs[region_key].isin(coordinate_system)].copy()
    table.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY] = table.obs[region_key].unique().tolist()
    return table


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
            elif get_model(element) == ShapesModel:
                instances = element.index.to_numpy()
            else:
                continue
            indices = ((table.obs[region_key] == name) & (table.obs[instance_key].isin(instances))).to_numpy()
            to_keep = to_keep | indices
    original_table = table
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
            n0 = np.setdiff1d(element.index.to_numpy(), table.obs["cell_id"].to_numpy())
            n1 = np.setdiff1d(table.obs["cell_id"].to_numpy(), element.index.to_numpy())
            raise ValueError(
                f"Instances in the table and in the element don't correspond: found {len(n0)} indices in the "
                f"element {name} but not in the table and found {len(n1)} indices in the table but not in the "
                "element"
            )
        assert sorted(set(instances.tolist())) == sorted(set(table.obs[instance_key].tolist()))
        table_df = pd.DataFrame({instance_key: table.obs[instance_key], "position": np.arange(len(instances))})
        merged = pd.merge(table_df, pd.DataFrame(index=instances), left_on=instance_key, right_index=True, how="right")
        matched_positions = merged["position"].to_numpy()
        table = table[matched_positions, :]
    _inplace_fix_subset_categorical_obs(subset_adata=table, original_adata=original_table)
    table = table.copy()
    table.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY] = table.obs[region_key].unique().tolist()
    return table


def match_table_to_element(sdata: SpatialData, element_name: str) -> AnnData:
    """
    Filter the table and reorders the rows to match the instances (rows/labels) of the specified SpatialElement.

    Parameters
    ----------
    sdata
        SpatialData object
    element_name
        Name of the element to match the table to

    Returns
    -------
    Table with the rows matching the instances of the element
    """
    assert sdata.table is not None, "No table found in the SpatialData"
    element_type, _, element = sdata._find_element(element_name)
    assert element_type in ["labels", "shapes"], f"Element {element_name} ({element_type}) is not supported"
    elements_dict = {element_type: {element_name: element}}
    return _filter_table_by_elements(sdata.table, elements_dict, match_rows=True)


@dataclass
class _ValueOrigin:
    origin: str
    is_categorical: bool
    value_key: str


def _locate_value(
    value_key: str,
    element: SpatialElement | None = None,
    sdata: SpatialData | None = None,
    element_name: str | None = None,
) -> list[_ValueOrigin]:
    assert (element is None) ^ (sdata is None and element_name is None)
    if element is not None:
        el = element
    else:
        assert sdata is not None
        assert element_name is not None
        el = sdata[element_name]
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
        table = sdata.table
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

    Returns
    -------
    DataFrame with the values requested.
    """
    assert (element is None) ^ (sdata is None and element_name is None)
    if element is not None:
        el = element
    else:
        assert sdata is not None
        assert element_name is not None
        el = sdata[element_name]
    value_keys = [value_key] if isinstance(value_key, str) else value_key
    locations = []
    for vk in value_keys:
        origins = _locate_value(value_key=vk, element=element, sdata=sdata, element_name=element_name)
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
    if sdata is not None:
        assert element_name is not None
        matched_table = match_table_to_element(sdata=sdata, element_name=element_name)
        if origin == "obs":
            return matched_table.obs[value_key_values]
        if origin == "var":
            x = matched_table[:, value_key_values].X
            import scipy

            if isinstance(x, scipy.sparse.csr_matrix):
                x = x.todense()
            return pd.DataFrame(x, columns=value_key_values)
    raise ValueError(f"Unknown origin {origin}")
