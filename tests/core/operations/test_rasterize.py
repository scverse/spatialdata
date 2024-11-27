from __future__ import annotations

import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from dask.dataframe import DataFrame as DaskDataFrame
from geopandas import GeoDataFrame
from multiscale_spatial_image import MultiscaleSpatialImage
from shapely import MultiPolygon, box
from spatial_image import SpatialImage
from xarray import DataArray, DataTree

from spatialdata import SpatialData, get_extent
from spatialdata._core.operations.rasterize import rasterize
from spatialdata._core.query.relational_query import get_element_instances
from spatialdata._utils import get_pyramid_levels
from spatialdata.models import PointsModel, ShapesModel, TableModel, get_axes_names
from spatialdata.models._utils import get_spatial_axes
from spatialdata.transformations import MapAxis
from tests.conftest import _get_images, _get_labels


# testing the two equivalent ways of calling rasterize, one with annotations in the element and passing the element,
# and the other with annotations in the table and passing the element name and the SpatialData object.
def _rasterize_test_alternative_calls(
    element: DaskDataFrame | GeoDataFrame | SpatialImage | MultiscaleSpatialImage,
    sdata: SpatialData,
    element_name: str,
    **kwargs,
) -> SpatialImage:
    kwargs0 = kwargs.copy()
    kwargs0["data"] = element

    kwargs1 = kwargs.copy()
    kwargs1["data"] = element_name
    kwargs1["sdata"] = sdata

    res0 = rasterize(**kwargs0)
    res1 = rasterize(**kwargs1)
    assert res0.equals(res1)

    return res0


@pytest.mark.parametrize("_get_raster", [_get_images, _get_labels])
def test_rasterize_raster(_get_raster):
    rasters = _get_raster()
    sdata = SpatialData.init_from_elements(rasters)

    def _rasterize(element: DataArray | DataTree, element_name: str, **kwargs) -> DataArray:
        return _rasterize_test_alternative_calls(element=element, sdata=sdata, element_name=element_name, **kwargs)

    def _get_data_of_largest_scale(raster):
        if isinstance(raster, DataArray):
            return raster.data.compute()

        xdata = get_pyramid_levels(raster, n=0)
        return xdata.data.compute()

    for element_name, raster in rasters.items():
        dims = get_axes_names(raster)
        all_slices = {"c": slice(0, 1000), "z": slice(0, 1000), "y": slice(5, 20), "x": slice(0, 5)}
        slices = [all_slices[d] for d in dims]

        data = _get_data_of_largest_scale(raster)
        data[tuple(slices)] = 1

        for kwargs in [
            {"target_unit_to_pixels": 2.0},
            {"target_width": 10.0},
            {"target_height": 10.0},
            {"target_depth": 10.0},
        ]:
            if "z" not in dims and "target_depth" in kwargs:
                continue
            spatial_dims = get_spatial_axes(dims)

            kwargs |= {
                "axes": spatial_dims,
                "min_coordinate": [0] * len(spatial_dims),
                "max_coordinate": [10] * len(spatial_dims),
                "target_coordinate_system": "global",
                "return_regions_as_labels": True,
            }
            result = _rasterize(element=raster, element_name=element_name, **kwargs)

            if "c" in raster.coords:
                assert np.array_equal(raster.coords["c"].values, result.coords["c"].values)

            result_data = _get_data_of_largest_scale(result)
            n_equal = result_data[tuple(slices)] == 1
            ratio = np.sum(n_equal) / np.prod(n_equal.shape)

            # 0.1: the z dim of the data is 2, the z dim of the target image is 20, because target_unit_to_pixels
            # is 2 and the bounding box is a square with size 10 x 10
            # 0.2: the z dim of the data is 2, the z dim of the target image is 10, because target_width is 10 and
            # the bounding box is a square
            target_ratio = (0.1 if "target_unit_to_pixels" in kwargs else 0.2) if "z" in dims else 1

            # image case (not labels)
            if "c" in dims:
                # this number approximately takes into account for this interpolation error, that makes some pixel
                # not match. The problem is described here: https://github.com/scverse/spatialdata/issues/166
                target_ratio *= 0.66
            # this approximately takes into account for the pixel offset problem, that makes some pixels not match.
            # The problem is described here: https://github.com/scverse/spatialdata/issues/165
            target_ratio *= 0.73

            EPS = 0.01
            if ratio < target_ratio - EPS:
                raise AssertionError(
                    "ratio is too small; ideally this number would be 100% but there is an offset error that needs "
                    "to be addressed. Also to get 100% we need to disable interpolation"
                )


def test_rasterize_labels_value_key_specified():
    element_name = "labels2d_multiscale_xarray"
    value_key = "background"
    table_name = "my_table"
    raster = _get_labels()[element_name]
    spatial_dims = get_spatial_axes(get_axes_names(raster))
    labels_indices = get_element_instances(raster)
    obs = pd.DataFrame(
        {
            "region": [element_name] * len(labels_indices),
            "instance_id": labels_indices,
            value_key: [True] * 10 + [False] * (len(labels_indices) - 10),
        }
    )
    table = TableModel.parse(
        AnnData(shape=(len(labels_indices), 0), obs=obs),
        region=element_name,
        region_key="region",
        instance_key="instance_id",
    )
    sdata = SpatialData.init_from_elements({element_name: raster}, tables={table_name: table})
    result = rasterize(
        data=element_name,
        sdata=sdata,
        axes=spatial_dims,
        min_coordinate=[0] * len(spatial_dims),
        max_coordinate=[10] * len(spatial_dims),
        target_coordinate_system="global",
        target_unit_to_pixels=2.0,
        value_key=value_key,
        table_name=table_name,
        return_regions_as_labels=True,
    )
    assert result.shape == (20, 20)
    # background pixels
    values = set(np.unique(result).tolist())
    assert values == {True, False}, values


def test_rasterize_points_shapes_with_string_index(points, shapes):
    sdata = SpatialData.init_from_elements({"points_0": points["points_0"], "circles": shapes["circles"]})

    # make the indices of the points_0 and circles dataframes strings
    sdata["points_0"]["str_index"] = dd.from_pandas(pd.Series([str(i) for i in sdata["points_0"].index]), npartitions=1)
    sdata["points_0"] = sdata["points_0"].set_index("str_index")
    sdata["circles"].index = [str(i) for i in sdata["circles"].index]

    data_extent = get_extent(sdata)

    for element_name in ["points_0", "circles"]:
        _ = rasterize(
            data=element_name,
            sdata=sdata,
            axes=("x", "y"),
            min_coordinate=[data_extent["x"][0], data_extent["y"][0]],
            max_coordinate=[data_extent["x"][1], data_extent["y"][1]],
            target_coordinate_system="global",
            target_unit_to_pixels=1,
            return_regions_as_labels=True,
            return_single_channel=True,
        )


def test_rasterize_shapes():
    box_one = box(0, 10, 20, 40)
    box_two = box(5, 35, 15, 45)
    box_three = box(0, 0, 2, 2)
    box_four = box(0, 2, 2, 4)

    gdf = GeoDataFrame(geometry=[box_one, MultiPolygon([box_two, box_three]), box_four])
    gdf["values"] = [0.1, 0.3, 0]
    gdf["cat_values"] = ["gene_a", "gene_a", "gene_b"]
    gdf["cat_values"] = gdf["cat_values"].astype("category")
    gdf = ShapesModel.parse(gdf, transformations={"global": MapAxis({"y": "x", "x": "y"})})

    element_name = "shapes"
    adata = AnnData(
        X=np.arange(len(gdf)).reshape(-1, 1),
        obs=pd.DataFrame(
            {
                "region": [element_name] * len(gdf),
                "instance_id": gdf.index,
                "values": gdf["values"],
                "cat_values": gdf["cat_values"],
            }
        ),
    )
    adata.obs["cat_values"] = adata.obs["cat_values"].astype("category")
    adata = TableModel.parse(adata, region=element_name, region_key="region", instance_key="instance_id")
    sdata = SpatialData.init_from_elements({element_name: gdf[["geometry"]]}, table=adata)

    def _rasterize(element: GeoDataFrame, **kwargs) -> SpatialImage:
        return _rasterize_test_alternative_calls(element=element, sdata=sdata, element_name=element_name, **kwargs)

    res = _rasterize(
        gdf,
        axes=("x", "y"),
        min_coordinate=[0, 0],
        max_coordinate=[50, 40],
        target_coordinate_system="global",
        target_unit_to_pixels=1,
    ).data.compute()

    assert res[0, 0, 0] == 2
    assert res[0, 30, 10] == 0
    assert res[0, 10, 30] == 1
    assert res[0, 10, 37] == 2

    res = _rasterize(
        gdf,
        axes=("x", "y"),
        min_coordinate=[0, 0],
        max_coordinate=[50, 40],
        target_coordinate_system="global",
        target_unit_to_pixels=1,
        return_single_channel=False,
    ).data.compute()

    assert res.shape == (3, 40, 50)
    assert res.max() == 1

    res = _rasterize(
        gdf,
        axes=("x", "y"),
        min_coordinate=[0, 0],
        max_coordinate=[50, 40],
        target_coordinate_system="global",
        target_unit_to_pixels=1,
        return_regions_as_labels=True,
    ).data.compute()

    assert res.shape == (40, 50)

    res = _rasterize(
        gdf,
        axes=("x", "y"),
        min_coordinate=[0, 0],
        max_coordinate=[50, 40],
        target_coordinate_system="global",
        target_unit_to_pixels=1,
        value_key="values",
    ).data.compute()

    assert res[0, 0, 0] == 0.3
    assert res[0, 30, 10] == 0
    assert res[0, 10, 30] == 0.1
    assert res[0, 10, 37] == 0.4

    res = _rasterize(
        gdf,
        axes=("x", "y"),
        min_coordinate=[0, 0],
        max_coordinate=[50, 40],
        target_coordinate_system="global",
        target_unit_to_pixels=1,
        value_key="cat_values",
    ).data.compute()

    assert res[0, 0, 3] == 2
    assert res[0, 10, 37] == 1

    res = rasterize(
        gdf,
        axes=("x", "y"),
        min_coordinate=[0, 0],
        max_coordinate=[50, 40],
        target_coordinate_system="global",
        target_unit_to_pixels=1,
        value_key="cat_values",
        return_single_channel=False,
    ).data.compute()

    assert res.shape == (2, 40, 50)
    assert res[0].max() == 2
    assert res[1].max() == 1


def test_rasterize_points():
    data = {
        "x": [0, 1, 0, 1, 2, 3, 3, 5.1],
        "y": [0, 0, 1, 1, 1, 1, 1, 5.1],
        "gene": ["A", "A", "B", "B", "C", "C", "C", "D"],
        "value": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.6, 0.8],
    }
    df = pd.DataFrame(data)
    df["gene"] = df["gene"].astype("category")

    ddf = dd.from_pandas(df, npartitions=2)
    ddf = PointsModel.parse(ddf)

    element_name = "points"
    adata = AnnData(
        X=np.arange(len(ddf)).reshape(-1, 1),
        obs=pd.DataFrame(
            {
                "region": [element_name] * len(ddf),
                "instance_id": ddf.index,
                "gene": data["gene"],
                "value": data["value"],
            }
        ),
    )
    adata.obs["gene"] = adata.obs["gene"].astype("category")
    adata = TableModel.parse(adata, region=element_name, region_key="region", instance_key="instance_id")
    sdata = SpatialData.init_from_elements({element_name: ddf[["x", "y"]]}, table=adata)

    def _rasterize(element: DaskDataFrame, **kwargs) -> SpatialImage:
        return _rasterize_test_alternative_calls(element=element, sdata=sdata, element_name=element_name, **kwargs)

    res = _rasterize(
        ddf,
        axes=("x", "y"),
        min_coordinate=[0, 0],
        max_coordinate=[5, 5],
        target_coordinate_system="global",
        target_unit_to_pixels=1.0,
    ).data.compute()

    assert res.max() == 2
    assert res[0, 1, 3] == 2
    assert res[0, -1, -1] == 0

    res = _rasterize(
        ddf,
        axes=("x", "y"),
        min_coordinate=[0, 0],
        max_coordinate=[5, 5],
        target_coordinate_system="global",
        target_unit_to_pixels=0.5,
    ).data.compute()

    assert res[0, 0, 0] == 5
    assert res[0, 0, 1] == 2

    res = _rasterize(
        ddf,
        axes=("x", "y"),
        min_coordinate=[0, 0],
        max_coordinate=[5, 5],
        target_coordinate_system="global",
        target_unit_to_pixels=1.0,
        return_single_channel=False,
        value_key="gene",
    ).data.compute()

    assert res[0].max() == 1
    assert res[1].max() == 1
    assert res[2].max() == 2

    res = _rasterize(
        ddf,
        axes=("x", "y"),
        min_coordinate=[0, 0],
        max_coordinate=[5, 5],
        target_coordinate_system="global",
        target_unit_to_pixels=1.0,
        value_key="gene",
    ).data.compute()

    assert res[0, 0, 0] == 1
    assert res[0, 1, 0] == 2
    assert res[0, 1, 2] == 3

    res = _rasterize(
        ddf,
        axes=("x", "y"),
        min_coordinate=[0, 0],
        max_coordinate=[5, 5],
        target_coordinate_system="global",
        target_unit_to_pixels=1.0,
        value_key="value",
    ).data.compute()

    assert res[0, 0, 1] == 0.2
    assert res[0, 1, 3] == 1.2


def test_rasterize_spatialdata(full_sdata):
    sdata = full_sdata.subset(
        ["image2d", "image2d_multiscale", "labels2d", "labels2d_multiscale", "points_0", "circles"]
    )
    _ = rasterize(
        data=sdata,
        axes=("x", "y"),
        min_coordinate=[0, 0],
        max_coordinate=[5, 5],
        target_coordinate_system="global",
        target_width=1000,
    )
