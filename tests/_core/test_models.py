from typing import Any, Callable

import numpy as np
import pytest
from spatial_image import SpatialImage, to_spatial_image

from spatialdata._core.models import Image2DModel, RasterSchema


class TestModels:
    @pytest.mark.parametrize(
        "converter",
        [
            # lambda _: _,
            # from_array,
            to_spatial_image,
            # DataArray,
        ],
    )
    @pytest.mark.parametrize(
        "model",
        [
            Image2DModel,
            # Labels2DModel,
            # Labels3DModel,
        ],  # TODO: Image3DModel once fixed.
    )
    def test_raster_schema(self, converter: Callable[..., Any], model: RasterSchema) -> None:
        n_dims = len(model.dims.dims)
        if n_dims == 2:
            image: np.ndarray = np.random.rand(10, 10)
        elif n_dims == 3:
            image = np.random.rand(3, 10, 10)
        image = converter(image)
        spatial_image = model.parse(image)

        assert isinstance(spatial_image, SpatialImage)
        assert spatial_image.shape == image.shape
        assert spatial_image.data.shape == image.shape
        assert spatial_image.data.dtype == image.dtype
        np.testing.assert_array_equal(spatial_image.data, image)
