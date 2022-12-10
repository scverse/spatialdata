from functools import partial
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import pytest
from dask.array.core import from_array
from numpy.random import default_rng
from shapely.io import to_ragged_array
from spatial_image import SpatialImage, to_spatial_image
from xarray import DataArray

from spatialdata._core.models import (
    Image2DModel,
    Labels2DModel,
    Labels3DModel,
    PointsModel,
    PolygonsModel,
    RasterSchema,
)
from tests._core.conftest import MULTIPOLYGON_PATH, POLYGON_PATH

RNG = default_rng()


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
        dims = np.array(model.dims.dims).tolist()
        if permute:
            RNG.shuffle(dims)
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
            assert set(spatial_image.shape) == set(image.shape)
            assert set(spatial_image.data.shape) == set(image.shape)
        assert spatial_image.data.dtype == image.dtype

    @pytest.mark.parametrize("model", [PolygonsModel])
    @pytest.mark.parametrize("path", [POLYGON_PATH, MULTIPOLYGON_PATH])
    def test_polygons_model(self, model: PolygonsModel, path: Path) -> None:
        poly = model.parse(path)
        assert PolygonsModel.GEOMETRY_KEY in poly
        assert PolygonsModel.TRANSFORM_KEY in poly.attrs

        geometry, data, offsets = to_ragged_array(poly.geometry.values)
        other_poly = model.parse(data, offsets, geometry)
        assert poly.equals(other_poly)

        other_poly = model.parse(poly)
        assert poly.equals(other_poly)

    @pytest.mark.parametrize("model", [PointsModel])
    @pytest.mark.parametrize(
        "annotations",
        [
            None,
            pd.DataFrame(RNG.integers(0, 100, size=(10, 3)), columns=["A", "B", "C"]),
        ],
    )
    def test_points_model(
        self,
        model: PointsModel,
        annotations: pd.DataFrame,
    ) -> None:
        coords = RNG.normal(size=(10, 2))
        if annotations is not None:
            annotations["A"] = annotations["A"].astype(str)
        points = model.parse(coords, annotations)
        assert PointsModel.GEOMETRY_KEY in points
        assert PointsModel.TRANSFORM_KEY in points.attrs
