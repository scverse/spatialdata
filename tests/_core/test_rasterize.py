import pytest

from spatialdata._core._rasterize import rasterize
from spatialdata._core.core_utils import get_dims, get_spatial_axes
from tests.conftest import _get_images, _get_labels


@pytest.mark.parametrize("_get_raster", [_get_images, _get_labels])
def test_rasterize_raster(_get_raster):
    rasters = _get_raster()
    for raster in rasters.values():
        dims = get_dims(raster)
        spatial_dims = get_spatial_axes(dims)
        for kwargs in [
            {"target_unit_to_pixels": 2.0},
            {"target_width": 10.0},
            {"target_height": 10.0},
            {"target_depth": 10.0},
        ]:
            if "z" not in dims and "target_depth" in kwargs:
                continue
            rasterize(
                raster,
                axes=spatial_dims,
                min_coordinate=[0] * len(spatial_dims),
                max_coordinate=[1] * len(spatial_dims),
                target_coordinate_system="global",
                **kwargs,
            )


@pytest.mark.skip(reason="Not implemented yet")
def test_rasterize_shapes(shapes):
    pass


@pytest.mark.skip(reason="Not implemented yet")
def test_rasterize_points(points):
    pass


@pytest.mark.skip(reason="Not implemented yet")
def test_rasterize_spatialdata(full_sdata):
    pass
