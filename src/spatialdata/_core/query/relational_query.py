from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Union

import dask.array as da
import numpy as np
import pandas as pd
from anndata import AnnData
from multiscale_spatial_image import MultiscaleSpatialImage
from spatial_image import SpatialImage

from spatialdata.models import Labels2DModel, Labels3DModel, PointsModel, ShapesModel, TableModel, get_model

if TYPE_CHECKING:
    from spatialdata import SpatialData


def _filter_table_by_coordinate_system(table: AnnData, coordinate_system: Union[str, list[str]]) -> Optional[AnnData]:
    if table is None:
        return None
    table_mapping_metadata = table.uns[TableModel.ATTRS_KEY]
    region_key = table_mapping_metadata[TableModel.REGION_KEY_KEY]
    table = table[table.obs[region_key].isin(coordinate_system)].copy()
    table.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY] = table.obs[region_key].unique().tolist()
    return table


def _filter_table_by_elements(table: AnnData, elements_dict: dict[str, dict[str, Any]]) -> Optional[AnnData]:
    assert set(elements_dict.keys()).issubset({"images", "labels", "shapes", "points"})
    if table is None:
        return None
    to_keep = np.zeros(len(table), dtype=bool)
    region_key = table.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY_KEY]
    instance_key = table.uns[TableModel.ATTRS_KEY][TableModel.INSTANCE_KEY]
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
            indices = (table.obs[region_key] == name) & (table.obs[instance_key].isin(instances))
            to_keep = to_keep | indices
    table = table[to_keep, :].copy()
    table.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY] = table.obs[region_key].unique().tolist()
    return table


@dataclass
class _ValueOrigin:
    name: str
    origin: str
    is_categorical: bool
    value_key: str


def locate_value(sdata: SpatialData, element_name: str, value_key: str) -> list[_ValueOrigin]:
    origins = []
    model = get_model(sdata[element_name])
    if model not in [PointsModel, ShapesModel, Labels2DModel, Labels3DModel]:
        raise ValueError(f"Cannot get value from {model}")
    el = sdata[element_name]
    # adding from the dataframe columns
    if model in [PointsModel, ShapesModel] and value_key in el.columns:
        value = el[value_key]
        is_categorical = pd.api.types.is_categorical_dtype(value)
        origins.append(_ValueOrigin(name=element_name, origin="df", is_categorical=is_categorical, value_key=value_key))

    # adding from the obs columns or var
    if model in [ShapesModel, Labels2DModel, Labels3DModel]:
        table = sdata.table
        if table is not None:
            # check if the table is annotating the element
            region = table.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY]
            if element_name in region:
                # check if the value_key is in the table
                if value_key in table.obs.columns:
                    value = table.obs[value_key]
                    is_categorical = pd.api.types.is_categorical_dtype(value)
                    origins.append(
                        _ValueOrigin(
                            name=element_name, origin="obs", is_categorical=is_categorical, value_key=value_key
                        )
                    )
                # check if the value_key is in the var
                elif value_key in table.var_names:
                    origins.append(
                        _ValueOrigin(name=element_name, origin="var", is_categorical=False, value_key=value_key)
                    )
    return origins


def get_values(sdata: SpatialData, element_name: str, value_key: str | list[str]) -> pd.DataFrame | np.ndarray:
    value_keys = [value_key] if isinstance(value_key, str) else value_key
    locations = []
    for vk in value_keys:
        origins = locate_value(sdata, element_name, vk)
        if len(origins) > 1:
            raise ValueError(f"{vk} has been found in multiple locations of {element_name}: {origins}")
        if len(origins) == 0:
            raise ValueError(f"{vk} has not been found in {element_name}")
        locations.append(origins[0])
    categorical_values = {x.is_categorical for x in locations}
    origin_values = {x.origin for x in locations}
    name_values = {x.name for x in locations}
    assert len(name_values) == 1
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
        el = sdata[element_name]
        return el[value_key_values]
    if origin == "obs":
        return sdata.table.obs[value_key_values]
    if origin == "var":
        return sdata.table[:, value_key_values].X
    raise ValueError(f"Unknown origin {origin}")
