from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Union

import dask.array as da
import numpy as np
from anndata import AnnData
from multiscale_spatial_image import MultiscaleSpatialImage
from spatial_image import SpatialImage

from spatialdata.models import Labels2DModel, Labels3DModel, ShapesModel, TableModel, get_model

if TYPE_CHECKING:
    pass


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
