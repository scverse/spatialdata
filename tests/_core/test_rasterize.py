import numpy as np
import pytest
from spatial_image import SpatialImage

from spatialdata._core._rasterize import rasterize
from spatialdata._core.core_utils import get_dims, get_spatial_axes
from spatialdata._io.write import _iter_multiscale
from tests.conftest import _get_images, _get_labels


@pytest.mark.parametrize("_get_raster", [_get_images, _get_labels])
def test_rasterize_raster(_get_raster):
    def _get_data_of_largest_scale(raster):
        if isinstance(raster, SpatialImage):
            data = raster.data.compute()
        else:
            xdata = next(iter(_iter_multiscale(raster, None)))
            data = xdata.data.compute()
        return data

    rasters = _get_raster()
    for raster in rasters.values():
        dims = get_dims(raster)
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
            if "z" in dims:
                if "target_unit_to_pixels" in kwargs:
                    # the z dim of the data is 2, the z dim of the target image is 20, because target_unit_to_pixels
                    # is 2 and the boundigbox is a square with size 10 x 10
                    target_ratio = 0.1
                else:
                    # the z dim of the data is 2, the z dim of the target image is 10, because target_width is 10 and
                    # the boundigbox is a square
                    target_ratio = 0.2
            else:
                target_ratio = 1

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
                print(f"ratio = {ratio}")
                print(
                    f"dims: {dims}, element: {type(raster)}, type data: {type(data)}, type result: {type(result_data)}, "
                    f"shape data: {data.shape}, shape result: {result_data.shape}"
                )
                print(f"kwargs = {kwargs}")
                raise AssertionError(
                    "ratio is too small; ideally this number would be 100% but there is an offset error that needs "
                    "to be addressed. Also to get 100% we need to disable interpolation"
                )

            # from napari_spatialdata import Interactive
            # Interactive(SpatialData.from_elements_dict({"raster": raster, "result": result}))


@pytest.mark.skip(reason="Not implemented yet")
def test_rasterize_shapes(shapes):
    pass


@pytest.mark.skip(reason="Not implemented yet")
def test_rasterize_points(points):
    pass


@pytest.mark.skip(reason="Not implemented yet")
def test_rasterize_spatialdata(full_sdata):
    pass
