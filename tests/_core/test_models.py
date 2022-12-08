from functools import partial
from typing import Any, Callable

import numpy as np
import pytest
from dask.array.core import from_array
from numpy.random import default_rng
from spatial_image import SpatialImage, to_spatial_image
from xarray import DataArray

from spatialdata._core.models import (
    Image2DModel,
    Labels2DModel,
    Labels3DModel,
    RasterSchema,
)


class TestModels:
    @pytest.mark.parametrize(
        "converter",
        [
            lambda _: _,
            from_array,
            DataArray,
            to_spatial_image,
        ],
    )
    @pytest.mark.parametrize(
        "model",
        [
            Image2DModel,
            Labels2DModel,
            Labels3DModel,
        ],  # TODO: Image3DModel once fixed.
    )
    @pytest.mark.parametrize(
        "permute",
        [
            True,
            False,
        ],
    )
    def test_raster_schema(self, converter: Callable[..., Any], model: RasterSchema, permute: bool) -> None:
        rng = default_rng()
        dims = np.array(model.dims.dims).tolist()
        if permute:
            rng.shuffle(dims)
        n_dims = len(dims)

        if converter is DataArray:
            converter = partial(converter, dims=dims)
        elif converter is to_spatial_image:
            converter = partial(converter, dims=model.dims.dims)
        if n_dims == 2:
            image: np.ndarray = np.random.rand(10, 10)
        elif n_dims == 3:
            image = np.random.rand(3, 10, 10)
        image = converter(image)
        spatial_image = model.parse(image)

        assert isinstance(spatial_image, SpatialImage)
        if not permute:
            assert spatial_image.shape == image.shape
            assert spatial_image.data.shape == image.shape
            np.testing.assert_array_equal(spatial_image.data, image)
        else:
            assert set(spatial_image.shape) == set(spatial_image.shape)
            assert set(spatial_image.data.shape) == set(image.shape)
        assert spatial_image.data.dtype == image.dtype
