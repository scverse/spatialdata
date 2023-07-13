import os
import pathlib
import tempfile
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional, Union

import dask.array.core
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from dask.array.core import from_array
from dask.dataframe.core import DataFrame as DaskDataFrame
from geopandas import GeoDataFrame
from multiscale_spatial_image import MultiscaleSpatialImage
from numpy.random import default_rng
from pandas.api.types import is_categorical_dtype
from shapely.io import to_ragged_array
from spatial_image import SpatialImage, to_spatial_image
from spatialdata import SpatialData
from spatialdata._types import ArrayLike
from spatialdata.models import (
    Image2DModel,
    Image3DModel,
    Labels2DModel,
    Labels3DModel,
    PointsModel,
    ShapesModel,
    TableModel,
    get_model,
    points_dask_dataframe_to_geopandas,
    points_geopandas_to_dask_dataframe,
)
from spatialdata.models._utils import validate_axis_name
from spatialdata.models.models import RasterSchema
from spatialdata.transformations._utils import (
    _set_transformations,
    _set_transformations_xarray,
)
from spatialdata.transformations.operations import (
    get_transformation,
    set_transformation,
)
from spatialdata.transformations.transformations import Scale
from xarray import DataArray

from tests.conftest import (
    MULTIPOLYGON_PATH,
    POINT_PATH,
    POLYGON_PATH,
    _get_images,
    _get_labels,
    _get_points,
    _get_shapes,
    _get_table,
)

RNG = default_rng()


def test_validate_axis_name():
    for ax in ["c", "x", "y", "z"]:
        validate_axis_name(ax)
    with pytest.raises(TypeError):
        validate_axis_name("invalid")


@pytest.mark.ci_only
class TestModels:
    def _parse_transformation_from_multiple_places(self, model: Any, element: Any, **kwargs) -> None:
        # This function seems convoluted but the idea is simple: sometimes the parser creates a whole new object,
        # other times (SpatialImage, DataArray, AnnData, GeoDataFrame) the object is enriched in-place. In such
        # cases we check that if there was already a transformation in the object we consider it then we are not
        # passing it also explicitly in the parser.
        # This function does that for all the models (it's called by the various tests of the models) and it first
        # creates clean copies of the element, and then puts the transformation inside it with various methods
        if any(isinstance(element, t) for t in (SpatialImage, DataArray, AnnData, GeoDataFrame, DaskDataFrame)):
            element_erased = deepcopy(element)
            # we are not respecting the function signature (the transform should be not None); it's fine for testing
            if isinstance(element_erased, DataArray) and not isinstance(element_erased, SpatialImage):
                # this case is for xarray.DataArray where the user manually updates the transform in attrs,
                # or when a user takes an image from a MultiscaleSpatialImage
                _set_transformations_xarray(element_erased, {})
            else:
                _set_transformations(element_erased, {})
            element_copy0 = deepcopy(element_erased)
            parsed0 = model.parse(element_copy0, **kwargs)

            element_copy1 = deepcopy(element_erased)
            t = Scale([1.0, 1.0], axes=("x", "y"))
            parsed1 = model.parse(element_copy1, transformations={"global": t}, **kwargs)
            assert get_transformation(parsed0, "global") != get_transformation(parsed1, "global")

            element_copy2 = deepcopy(element_erased)
            if isinstance(element_copy2, DataArray) and not isinstance(element_copy2, SpatialImage):
                _set_transformations_xarray(element_copy2, {"global": t})
            else:
                set_transformation(element_copy2, t, "global")
            parsed2 = model.parse(element_copy2, **kwargs)
            assert get_transformation(parsed1, "global") == get_transformation(parsed2, "global")

            with pytest.raises(ValueError):
                element_copy3 = deepcopy(element_erased)
                if isinstance(element_copy3, DataArray) and not isinstance(element_copy3, SpatialImage):
                    _set_transformations_xarray(element_copy3, {"global": t})
                else:
                    set_transformation(element_copy3, t, "global")
                model.parse(element_copy3, transformations={"global": t}, **kwargs)
        elif any(
            isinstance(element, t)
            for t in (
                MultiscaleSpatialImage,
                str,
                np.ndarray,
                dask.array.core.Array,
                pathlib.PosixPath,
                pd.DataFrame,
            )
        ):
            # no need to apply this function since the parser always creates a new object and the transformation is not
            # part of the object passed as input
            pass
        else:
            raise ValueError(f"Unknown type {type(element)}")

    def _passes_validation_after_io(self, model: Any, element: Any, element_type: str) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.zarr")
            d = {"element": element}
            if element_type == "image":
                sdata = SpatialData(images=d)
            elif element_type == "labels":
                sdata = SpatialData(labels=d)
            elif element_type == "points":
                sdata = SpatialData(points=d)
            elif element_type == "shapes":
                sdata = SpatialData(shapes=d)
            else:
                raise ValueError(f"Unknown element type {element_type}")
            sdata.write(path)
            sdata_read = SpatialData.read(path)
            group_name = element_type if element_type != "image" else "images"
            element_read = sdata_read.__getattribute__(group_name)["element"]
            # TODO: raster models have validate as a method (for non-raster it's a class method),
            #  probably because they call the xarray schema validation in the superclass. Can we make it consistent?
            if element_type == "image" or element_type == "labels":
                model().validate(element_read)
            else:
                model.validate(element_read)

    @pytest.mark.parametrize("converter", [lambda _: _, from_array, DataArray, to_spatial_image])
    @pytest.mark.parametrize("model", [Image2DModel, Labels2DModel, Labels3DModel, Image3DModel])
    @pytest.mark.parametrize("permute", [True, False])
    @pytest.mark.parametrize("kwargs", [None, {"name": "test"}])
    def test_raster_schema(
        self, converter: Callable[..., Any], model: RasterSchema, permute: bool, kwargs: Optional[dict[str, str]]
    ) -> None:
        dims = np.array(model.dims.dims).tolist()
        if permute:
            RNG.shuffle(dims)
        n_dims = len(dims)

        if converter is DataArray:
            converter = partial(converter, dims=dims)
        elif converter is to_spatial_image:
            converter = partial(converter, dims=model.dims.dims)
        if n_dims == 2:
            image: ArrayLike = RNG.uniform(size=(10, 10))
        elif n_dims == 3:
            image: ArrayLike = RNG.uniform(size=(3, 10, 10))
        elif n_dims == 4:
            image: ArrayLike = RNG.uniform(size=(2, 3, 10, 10))
        image = converter(image)
        self._parse_transformation_from_multiple_places(model, image)
        spatial_image = model.parse(image)
        if model in [Image2DModel, Image3DModel]:
            element_type = "image"
        elif model in [Labels2DModel, Labels3DModel]:
            element_type = "labels"
        else:
            raise ValueError(f"Unknown model {model}")
        self._passes_validation_after_io(model, spatial_image, element_type)

        assert isinstance(spatial_image, SpatialImage)
        if not permute:
            assert spatial_image.shape == image.shape
            assert spatial_image.data.shape == image.shape
            np.testing.assert_array_equal(spatial_image.data, image)
        else:
            assert set(spatial_image.shape) == set(image.shape)
            assert set(spatial_image.data.shape) == set(image.shape)
        assert spatial_image.data.dtype == image.dtype
        if kwargs is not None:
            with pytest.raises(ValueError):
                model.parse(image, **kwargs)

    @pytest.mark.parametrize("model", [ShapesModel])
    @pytest.mark.parametrize("path", [POLYGON_PATH, MULTIPOLYGON_PATH, POINT_PATH])
    def test_shapes_model(self, model: ShapesModel, path: Path) -> None:
        radius = RNG.normal(size=(2,)) if path.name == "points.json" else None
        self._parse_transformation_from_multiple_places(model, path)
        poly = model.parse(path, radius=radius)
        self._passes_validation_after_io(model, poly, "shapes")
        assert ShapesModel.GEOMETRY_KEY in poly
        assert ShapesModel.TRANSFORM_KEY in poly.attrs
        geometry, data, offsets = to_ragged_array(poly.geometry.values)
        self._parse_transformation_from_multiple_places(model, data)
        other_poly = model.parse(data, geometry=geometry, offsets=offsets, radius=radius)
        self._passes_validation_after_io(model, other_poly, "shapes")
        assert poly.equals(other_poly)

        self._parse_transformation_from_multiple_places(model, poly)
        other_poly = model.parse(poly)
        self._passes_validation_after_io(model, other_poly, "shapes")
        assert poly.equals(other_poly)

    @pytest.mark.parametrize("model", [PointsModel])
    @pytest.mark.parametrize("instance_key", [None, "cell_id"])
    @pytest.mark.parametrize("feature_key", [None, "target"])
    @pytest.mark.parametrize("typ", [np.ndarray, pd.DataFrame, dd.DataFrame])
    @pytest.mark.parametrize("is_annotation", [True, False])
    @pytest.mark.parametrize("is_3d", [True, False])
    def test_points_model(
        self,
        model: PointsModel,
        typ: Any,
        is_3d: bool,
        is_annotation: bool,
        instance_key: Optional[str],
        feature_key: Optional[str],
    ) -> None:
        coords = ["A", "B", "C"]
        axes = ["x", "y", "z"]
        data = pd.DataFrame(RNG.integers(0, 101, size=(10, 3)), columns=coords)
        data["target"] = pd.Series(RNG.integers(0, 2, size=(10,))).astype(str)
        data["cell_id"] = pd.Series(RNG.integers(0, 5, size=(10,))).astype(np.int_)
        data["anno"] = pd.Series(RNG.integers(0, 1, size=(10,))).astype(np.int_)
        if not is_3d:
            coords = coords[:2]
            axes = axes[:2]
        if typ == np.ndarray:
            numpy_coords = data[coords].to_numpy()
            self._parse_transformation_from_multiple_places(model, numpy_coords)
            points = model.parse(
                numpy_coords,
                annotation=data,
                instance_key=instance_key,
                feature_key=feature_key,
            )
            self._passes_validation_after_io(model, points, "points")
        elif typ == pd.DataFrame:
            coordinates = dict(zip(axes, coords))
            self._parse_transformation_from_multiple_places(model, data)
            points = model.parse(
                data,
                coordinates=coordinates,
                instance_key=instance_key,
                feature_key=feature_key,
            )
            self._passes_validation_after_io(model, points, "points")
        elif typ == dd.DataFrame:
            coordinates = dict(zip(axes, coords))
            dd_data = dd.from_pandas(data, npartitions=2)
            self._parse_transformation_from_multiple_places(model, dd_data, coordinates=coordinates)
            points = model.parse(
                dd_data,
                coordinates=coordinates,
                instance_key=instance_key,
                feature_key=feature_key,
            )
            self._passes_validation_after_io(model, points, "points")
        assert "transform" in points.attrs
        if feature_key is not None and is_annotation:
            assert "spatialdata_attrs" in points.attrs
            assert "feature_key" in points.attrs["spatialdata_attrs"]
            assert "target" in points.attrs["spatialdata_attrs"]["feature_key"]
        if instance_key is not None and is_annotation:
            assert "spatialdata_attrs" in points.attrs
            assert "instance_key" in points.attrs["spatialdata_attrs"]
            assert "cell_id" in points.attrs["spatialdata_attrs"]["instance_key"]

    @pytest.mark.parametrize("model", [TableModel])
    @pytest.mark.parametrize("region", ["sample", RNG.choice([1, 2], size=10).tolist()])
    def test_table_model(
        self,
        model: TableModel,
        region: Union[str, np.ndarray],
    ) -> None:
        region_key = "reg"
        obs = pd.DataFrame(RNG.integers(0, 100, size=(10, 3)), columns=["A", "B", "C"])
        obs[region_key] = region
        adata = AnnData(RNG.normal(size=(10, 2)), obs=obs)
        table = model.parse(adata, region=region, region_key=region_key, instance_key="A")
        assert region_key in table.obs
        assert is_categorical_dtype(table.obs[region_key])
        assert table.obs[region_key].cat.categories.tolist() == np.unique(region).tolist()
        assert table.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY_KEY] == region_key
        assert table.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY] == region
        assert TableModel.ATTRS_KEY in table.uns
        assert TableModel.REGION_KEY in table.uns[TableModel.ATTRS_KEY]
        assert TableModel.REGION_KEY_KEY in table.uns[TableModel.ATTRS_KEY]
        assert table.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY] == region


def test_get_schema():
    images = _get_images()
    labels = _get_labels()
    points = _get_points()
    shapes = _get_shapes()
    table = _get_table()
    for k, v in images.items():
        schema = get_model(v)
        if "2d" in k:
            assert schema == Image2DModel
        elif "3d" in k:
            assert schema == Image3DModel
        else:
            raise ValueError(f"Unexpected key: {k}")
    for k, v in labels.items():
        schema = get_model(v)
        if "2d" in k:
            assert schema == Labels2DModel
        elif "3d" in k:
            assert schema == Labels3DModel
        else:
            raise ValueError(f"Unexpected key: {k}")
    for v in points.values():
        schema = get_model(v)
        assert schema == PointsModel
    for v in shapes.values():
        schema = get_model(v)
        assert schema == ShapesModel
    schema = get_model(table)
    assert schema == TableModel


def test_points_and_shapes_conversions(shapes, points):
    from spatialdata.transformations import get_transformation

    circles0 = shapes["circles"]
    circles1 = points_geopandas_to_dask_dataframe(circles0)
    circles2 = points_dask_dataframe_to_geopandas(circles1)
    circles0 = circles0[circles2.columns]
    assert np.all(circles0.values == circles2.values)

    t0 = get_transformation(circles0, get_all=True)
    t1 = get_transformation(circles1, get_all=True)
    t2 = get_transformation(circles2, get_all=True)
    assert t0 == t1
    assert t0 == t2

    points0 = points["points_0"]
    points1 = points_dask_dataframe_to_geopandas(points0)
    points2 = points_geopandas_to_dask_dataframe(points1)
    points0 = points0[points2.columns]
    assert np.all(points0.values == points2.values)

    t0 = get_transformation(points0, get_all=True)
    t1 = get_transformation(points1, get_all=True)
    t2 = get_transformation(points2, get_all=True)
    assert t0 == t1
    assert t0 == t2
