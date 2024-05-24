import numpy as np
import pytest
from geopandas import GeoDataFrame
from shapely import MultiPolygon, box
from spatial_image import SpatialImage
from spatialdata._core.operations.rasterize import rasterize
from spatialdata._io._utils import _iter_multiscale
from spatialdata.models import ShapesModel, get_axes_names
from spatialdata.models._utils import get_spatial_axes
from spatialdata.transformations import MapAxis

from tests.conftest import _get_images, _get_labels


@pytest.mark.parametrize("_get_raster", [_get_images, _get_labels])
def test_rasterize_raster(_get_raster):
    def _get_data_of_largest_scale(raster):
        if isinstance(raster, SpatialImage):
            return raster.data.compute()

        xdata = next(iter(_iter_multiscale(raster, None)))
        return xdata.data.compute()

    rasters = _get_raster()
    for raster in rasters.values():
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
            result = rasterize(
                raster,
                axes=spatial_dims,
                min_coordinate=[0] * len(spatial_dims),
                max_coordinate=[10] * len(spatial_dims),
                target_coordinate_system="global",
                **kwargs,
            )

            result_data = _get_data_of_largest_scale(result)
            n_equal = result_data[tuple(slices)] == 1
            ratio = np.sum(n_equal) / np.prod(n_equal.shape)

            # 0.1: the z dim of the data is 2, the z dim of the target image is 20, because target_unit_to_pixels
            # is 2 and the bounding box is a square with size 10 x 10
            # 0.2: the z dim of the data is 2, the z dim of the target image is 10, because target_width is 10 and
            # the boundigbox is a square
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


def test_rasterize_shapes():
    square_one = box(0, 10, 20, 40)
    square_two = box(5, 35, 15, 45)
    square_three = box(0, 0, 2, 2)
    square_four = box(0, 2, 2, 4)

    gdf = GeoDataFrame(geometry=[square_one, MultiPolygon([square_two, square_three]), square_four])
    gdf["values"] = [0.1, 0.3, 0]
    gdf["cat_values"] = ["gene_a", "gene_a", "gene_b"]
    gdf["cat_values"] = gdf["cat_values"].astype("category")
    gdf = ShapesModel.parse(gdf, transformations={"global": MapAxis({"y": "x", "x": "y"})})

    res = rasterize(gdf, ["x", "y"], [0, 0], [50, 40], "global", target_unit_to_pixels=1).data.compute()

    assert res[0, 0, 0] == 1
    assert res[0, 30, 10] == 0
    assert res[0, 10, 30] == 1
    assert res[0, 10, 37] == 2

    res = rasterize(
        gdf, ["x", "y"], [0, 0], [50, 40], "global", target_unit_to_pixels=1, instance_key_as_default_value_key=True
    ).data.compute()

    assert res[0, 0, 0] == 2
    assert res[0, 30, 10] == 0
    assert res[0, 10, 30] == 1
    assert res[0, 10, 37] == 2

    res = rasterize(
        gdf,
        ["x", "y"],
        [0, 0],
        [50, 40],
        "global",
        target_unit_to_pixels=1,
        instance_key_as_default_value_key=True,
        return_single_channel=False,
    ).data.compute()

    assert res.shape == (3, 40, 50)
    assert res.max() == 1

    res = rasterize(
        gdf, ["x", "y"], [0, 0], [50, 40], "global", target_unit_to_pixels=1, return_as_labels=True
    ).data.compute()

    assert res.shape == (40, 50)

    res = rasterize(
        gdf, ["x", "y"], [0, 0], [50, 40], "global", target_unit_to_pixels=1, value_key="values"
    ).data.compute()

    assert res[0, 0, 0] == 0.3
    assert res[0, 30, 10] == 0
    assert res[0, 10, 30] == 0.1
    assert res[0, 10, 37] == 0.4

    res = rasterize(
        gdf, ["x", "y"], [0, 0], [50, 40], "global", target_unit_to_pixels=1, value_key="cat_values"
    ).data.compute()

    assert res[0, 0, 3] == 2
    assert res[0, 10, 37] == 1

    res = rasterize(
        gdf,
        ["x", "y"],
        [0, 0],
        [50, 40],
        "global",
        target_unit_to_pixels=1,
        value_key="cat_values",
        return_single_channel=False,
    ).data.compute()

    assert res.shape == (2, 40, 50)
    assert res[0].max() == 2
    assert res[1].max() == 1


@pytest.mark.skip(reason="Not implemented yet")
def test_rasterize_points(points):
    pass


@pytest.mark.skip(reason="Not implemented yet")
def test_rasterize_spatialdata(full_sdata):
    pass
