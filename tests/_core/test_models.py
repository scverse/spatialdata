from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from anndata import AnnData
from dask.array.core import from_array
from numpy.random import default_rng
from pandas.api.types import is_categorical_dtype
from shapely.io import to_ragged_array
from spatial_image import SpatialImage, to_spatial_image
from xarray import DataArray

from spatialdata._core.models import (
    Image2DModel,
    Image3DModel,
    Labels2DModel,
    Labels3DModel,
    PointsModel,
    PolygonsModel,
    RasterSchema,
    ShapesModel,
    TableModel,
    get_schema,
)
from tests._core.conftest import MULTIPOLYGON_PATH, POLYGON_PATH
from tests.conftest import (
    _get_images,
    _get_labels,
    _get_points,
    _get_polygons,
    _get_shapes,
    _get_table,
)

RNG = default_rng()


class TestModels:
    @pytest.mark.parametrize("converter", [lambda _: _, from_array, DataArray, to_spatial_image])
    @pytest.mark.parametrize("model", [Image2DModel, Labels2DModel, Labels3DModel])  # TODO: Image3DModel once fixed.
    @pytest.mark.parametrize("permute", [True, False])
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

    @pytest.mark.skip("Waiting for the new points implementation")
    @pytest.mark.parametrize("model", [PointsModel])
    @pytest.mark.parametrize(
        "annotations",
        [None, pd.DataFrame(RNG.integers(0, 101, size=(10, 3)), columns=["A", "B", "C"])],
    )
    def test_points_model(
        self,
        model: PointsModel,
        annotations: pd.DataFrame,
    ) -> None:
        coords = RNG.normal(size=(10, 2))
        if annotations is not None:
            annotations["A"] = annotations["A"].astype(str)
        points = model.parse(coords, None if annotations is None else pa.Table.from_pandas(annotations))
        assert PointsModel.TRANSFORM_KEY.encode("utf-8") in points.schema.metadata

    @pytest.mark.parametrize("model", [ShapesModel])
    @pytest.mark.parametrize("shape_type", [None, "Circle", "Square"])
    @pytest.mark.parametrize("shape_size", [None, RNG.normal(size=(10,)), 0.3])
    def test_shapes_model(
        self,
        model: ShapesModel,
        shape_type: Optional[str],
        shape_size: Optional[Union[int, float, np.ndarray]],
    ) -> None:
        coords = RNG.normal(size=(10, 2))
        shapes = model.parse(coords, shape_type, shape_size)
        assert ShapesModel.COORDS_KEY in shapes.obsm
        assert ShapesModel.TRANSFORM_KEY in shapes.uns
        assert ShapesModel.SIZE_KEY in shapes.obs
        if shape_size is not None:
            assert shapes.obs[ShapesModel.SIZE_KEY].dtype == np.float64
            if isinstance(shape_size, np.ndarray):
                assert shapes.obs[ShapesModel.SIZE_KEY].shape == shape_size.shape
            elif isinstance(shape_size, float):
                assert shapes.obs[ShapesModel.SIZE_KEY].unique() == shape_size
            else:
                raise ValueError(f"Unexpected shape_size: {shape_size}")
        assert ShapesModel.ATTRS_KEY in shapes.uns
        assert ShapesModel.TYPE_KEY in shapes.uns[ShapesModel.ATTRS_KEY]
        assert shape_type == shapes.uns[ShapesModel.ATTRS_KEY][ShapesModel.TYPE_KEY]

    @pytest.mark.parametrize("model", [TableModel])
    @pytest.mark.parametrize("region", ["sample", RNG.choice([1, 2], size=10).tolist()])
    def test_table_model(
        self,
        model: TableModel,
        region: Union[str, np.ndarray],
    ) -> None:
        region_key = "reg"
        obs = pd.DataFrame(RNG.integers(0, 100, size=(10, 3)), columns=["A", "B", "C"])
        obs["A"] = obs["A"].astype(str)  # instance_key
        obs[region_key] = region
        adata = AnnData(RNG.normal(size=(10, 2)), obs=obs)
        if not isinstance(region, str):
            table = model.parse(adata, region=region, region_key=region_key, instance_key="A")
            assert region_key in table.obs
            assert is_categorical_dtype(table.obs[region_key])
            assert table.obs[region_key].cat.categories.tolist() == np.unique(region).tolist()
            assert table.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY_KEY] == region_key
        else:
            table = model.parse(adata, region=region, instance_key="A")
        assert TableModel.ATTRS_KEY in table.uns
        assert TableModel.REGION_KEY in table.uns[TableModel.ATTRS_KEY]
        assert TableModel.REGION_KEY_KEY in table.uns[TableModel.ATTRS_KEY]
        assert table.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY] == region


@pytest.mark.skip("Waiting for the new points implementation")
def test_get_schema():
    images = _get_images()
    labels = _get_labels()
    polygons = _get_polygons()
    points = _get_points()
    shapes = _get_shapes()
    table = _get_table(region="sample1")
    for k, v in images.items():
        schema = get_schema(v)
        if "2d" in k:
            assert schema == Image2DModel
        elif "3d" in k:
            assert schema == Image3DModel
        else:
            raise ValueError(f"Unexpected key: {k}")
    for k, v in labels.items():
        schema = get_schema(v)
        if "2d" in k:
            assert schema == Labels2DModel
        elif "3d" in k:
            assert schema == Labels3DModel
        else:
            raise ValueError(f"Unexpected key: {k}")
    for v in polygons.values():
        schema = get_schema(v)
        assert schema == PolygonsModel
    for v in points.values():
        schema = get_schema(v)
        assert schema == PointsModel
    for v in shapes.values():
        schema = get_schema(v)
        assert schema == ShapesModel
    schema = get_schema(table)
    assert schema == TableModel
