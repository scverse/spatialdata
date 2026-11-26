from __future__ import annotations

from copy import deepcopy as _deepcopy
from functools import singledispatch

from anndata import AnnData
from dask.array.core import Array as DaskArray
from dask.array.core import from_array
from dask.dataframe import DataFrame as DaskDataFrame
from geopandas import GeoDataFrame
from xarray import DataArray, DataTree

from spatialdata._core.spatialdata import SpatialData
from spatialdata.models._utils import SpatialElement
from spatialdata.models.models import (
    Image2DModel,
    Image3DModel,
    Labels2DModel,
    Labels3DModel,
    PointsModel,
    RasterSchema,
    get_model,
)


@singledispatch
def deepcopy(element: SpatialData | SpatialElement | AnnData) -> SpatialData | SpatialElement | AnnData:
    """
    Deepcopy a SpatialData or SpatialElement object.

    Deepcopy will load the data in memory. Using this function for large Dask-backed objects is discouraged. In that
    case, please save the SpatialData object to a different disk location and read it back again.

    Parameters
    ----------
    element
        The SpatialData or SpatialElement object to deepcopy

    Returns
    -------
    A deepcopy of the SpatialData or SpatialElement object

    Notes
    -----
    The order of the columns for a deepcopied points element may be differ from the original one, please see more here:
    https://github.com/scverse/spatialdata/issues/486
    """
    raise RuntimeError(f"Wrong type for deepcopy: {type(element)}")


# In the implementations below, when the data is loaded from Dask, we first use compute() and then we deepcopy the data.
# This leads to double copying the data, but since we expect the data to be small, this is acceptable.
@deepcopy.register(SpatialData)
def _(sdata: SpatialData) -> SpatialData:
    elements_dict: dict[str, SpatialElement | AnnData] = {}
    for _, element_name, element in sdata.gen_elements():
        copied = deepcopy(element)
        assert not isinstance(copied, SpatialData)
        elements_dict[element_name] = copied
    deepcopied_attrs = _deepcopy(sdata.attrs)
    return SpatialData.init_from_elements(elements_dict, attrs=deepcopied_attrs)


@deepcopy.register(DataArray)
def _(element: DataArray) -> DataArray:
    model = get_model(element)
    if isinstance(element.data, DaskArray):
        element = element.compute()
    assert issubclass(model, RasterSchema)
    if model in [Image2DModel, Image3DModel]:
        return model.parse(element.copy(deep=True), c_coords=list(element["c"].to_numpy()))
    assert model in [Labels2DModel, Labels3DModel]
    return model.parse(element.copy(deep=True))


@deepcopy.register(DataTree)
def _(element: DataTree) -> DataTree:
    # TODO: now that multiscale_spatial_image 1.0.0 is supported, this code can probably be simplified. Check
    # https://github.com/scverse/spatialdata/pull/587/files#diff-c74ebf49cb8cbddcfaec213defae041010f2043cfddbded24175025b6764ef79
    # to understand the original motivation.
    model = get_model(element)
    for key in element:
        node = element[key]
        assert isinstance(node, DataTree)
        ds = node.ds
        assert len(ds) == 1
        variable = str(next(iter(ds)))
        if isinstance(node[variable].data, DaskArray):
            node[variable] = node[variable].compute()
    msi = element.copy(deep=True)
    for key in msi:
        copied_node = msi[key]
        assert isinstance(copied_node, DataTree)
        original_node = element[key]
        assert isinstance(original_node, DataTree)
        variable = str(next(iter(copied_node.ds)))
        copied_node[variable].data = from_array(copied_node[variable].data)
        original_node[variable].data = from_array(original_node[variable].data)
    assert model in [Image2DModel, Image3DModel, Labels2DModel, Labels3DModel]
    model.validate(msi)
    return msi


@deepcopy.register(GeoDataFrame)
def _(gdf: GeoDataFrame) -> GeoDataFrame:
    new_gdf = _deepcopy(gdf)
    # temporary fix for https://github.com/scverse/spatialdata/issues/286.
    new_gdf.attrs = _deepcopy(gdf.attrs)
    return new_gdf


@deepcopy.register(DaskDataFrame)
def _(df: DaskDataFrame) -> DaskDataFrame:
    # bug: the parser may change the order of the columns
    compute_df = df.compute().copy(deep=True)
    new_ddf = PointsModel.parse(compute_df)
    # the problem is not .copy(deep=True), but the parser, which discards some metadata https://github.com/scverse/spatialdata/issues/503#issuecomment-2015275322
    # We need to use the compute_df here as with deepcopy, df._attrs does not exist anymore.
    # print(type(new_ddf.attrs))
    new_ddf.attrs.update(_deepcopy(compute_df.attrs))
    return new_ddf


@deepcopy.register(AnnData)
def _(adata: AnnData) -> AnnData:
    return adata.copy()
