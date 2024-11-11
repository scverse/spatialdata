from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Callable
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import dask.array.core
import dask.dataframe as dd
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from dask.array.core import from_array
from dask.dataframe import DataFrame as DaskDataFrame
from geopandas import GeoDataFrame
from numpy.random import default_rng
from shapely.geometry import MultiPolygon, Point, Polygon
from shapely.io import to_ragged_array
from spatial_image import to_spatial_image
from xarray import DataArray, DataTree

from spatialdata._core.spatialdata import SpatialData
from spatialdata._types import ArrayLike
from spatialdata.models._utils import (
    force_2d,
    points_dask_dataframe_to_geopandas,
    points_geopandas_to_dask_dataframe,
    validate_axis_name,
)
from spatialdata.models.models import (
    Image2DModel,
    Image3DModel,
    Labels2DModel,
    Labels3DModel,
    PointsModel,
    RasterSchema,
    ShapesModel,
    TableModel,
    get_axes_names,
    get_model,
)
from spatialdata.testing import assert_elements_are_identical
from spatialdata.transformations._utils import (
    _set_transformations,
)
from spatialdata.transformations.operations import (
    get_transformation,
    set_transformation,
)
from spatialdata.transformations.transformations import Identity, Scale
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

RNG = default_rng(seed=0)


def test_validate_axis_name():
    for ax in ["c", "x", "y", "z"]:
        validate_axis_name(ax)
    with pytest.raises(TypeError):
        validate_axis_name("invalid")


@pytest.mark.ci_only
class TestModels:
    def _parse_transformation_from_multiple_places(self, model: Any, element: Any, **kwargs) -> None:
        # This function seems convoluted but the idea is simple: sometimes the parser creates a whole new object,
        # other times (DataArray, AnnData, GeoDataFrame) the object is enriched in-place. In such
        # cases we check that if there was already a transformation in the object then we are not
        # passing it also explicitly in the parser.
        # This function does that for all the models (it's called by the various tests of the models) and it first
        # creates clean copies of the element, and then puts the transformation inside it with various methods
        if any(isinstance(element, t) for t in (DataArray, GeoDataFrame, DaskDataFrame)):
            # no transformation in the element, nor passed to the parser (default transformation is added)

            _set_transformations(element, {})
            parsed0 = model.parse(element, **kwargs)
            assert get_transformation(parsed0, "global") == Identity()

            # no transformation in the element, but passed to the parser
            _set_transformations(element, {})
            t = Scale([1.0, 1.0], axes=("x", "y"))
            parsed1 = model.parse(element, transformations={"global": t}, **kwargs)
            assert get_transformation(parsed1, "global") == t

            # transformation in the element, but not passed to the parser
            _set_transformations(element, {})
            set_transformation(element, t, "global")
            parsed2 = model.parse(element, **kwargs)
            assert get_transformation(parsed2, "global") == t

            # transformation in the element, and passed to the parser
            with pytest.raises(ValueError):
                _set_transformations(element, {})
                set_transformation(element, t, "global")
                model.parse(element, transformations={"global": t}, **kwargs)
        elif any(
            isinstance(element, t)
            for t in (
                DataTree,
                str,
                np.ndarray,
                dask.array.core.Array,
                Path,
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
        self, converter: Callable[..., Any], model: RasterSchema, permute: bool, kwargs: dict[str, str] | None
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

        assert isinstance(spatial_image, DataArray)
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

    @pytest.mark.parametrize("model", [Labels2DModel, Labels3DModel])
    def test_labels_model_with_multiscales(self, model):
        # Passing "scale_factors" should generate multiscales with a "method" appropriate for labels
        dims = np.array(model.dims.dims).tolist()
        n_dims = len(dims)

        # A labels image with one label value 4, that partially covers 2×2 blocks.
        # Downsampling with interpolation would produce values 1, 2, 3, 4.
        image: ArrayLike = np.array([[0, 0, 0, 0], [0, 4, 4, 4], [4, 4, 4, 4], [0, 4, 4, 4]], dtype=np.uint16)
        if n_dims == 3:
            image = np.stack([image] * image.shape[0])
        actual = model.parse(image, scale_factors=(2,))
        assert isinstance(actual, DataTree)
        assert actual.children.keys() == {"scale0", "scale1"}
        assert actual.scale0.image.dtype == image.dtype
        assert actual.scale1.image.dtype == image.dtype
        assert set(np.unique(image)) == set(np.unique(actual.scale0.image)), "Scale0 should be preserved"
        assert set(np.unique(image)) >= set(
            np.unique(actual.scale1.image)
        ), "Subsequent scales should not have interpolation artifacts"

    @pytest.mark.parametrize("model", [ShapesModel])
    @pytest.mark.parametrize("path", [POLYGON_PATH, MULTIPOLYGON_PATH, POINT_PATH])
    def test_shapes_model(self, model: ShapesModel, path: Path) -> None:
        radius = np.abs(RNG.normal(size=(2,))) if path.name == "points.json" else None
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

        if ShapesModel.RADIUS_KEY in poly.columns:
            poly[ShapesModel.RADIUS_KEY].iloc[0] = -1
            with pytest.raises(ValueError, match="Radii of circles must be positive."):
                ShapesModel.validate(poly)
            poly[ShapesModel.RADIUS_KEY].iloc[0] = 0
            with pytest.raises(ValueError, match="Radii of circles must be positive."):
                ShapesModel.validate(poly)

            # tests to be restored when the validation is re-enabled (now it just raises a warning, that is tricky to
            # capture)
            # poly[ShapesModel.RADIUS_KEY].iloc[0] = np.nan
            # with pytest.raises(ValueError, match="Radii of circles must not be nan or inf."):
            #     ShapesModel.validate(poly)
            #
            # poly[ShapesModel.RADIUS_KEY].iloc[0] = np.inf
            # with pytest.raises(ValueError, match="Radii of circles must not be nan or inf."):
            #     ShapesModel.validate(poly)

    @pytest.mark.parametrize("model", [PointsModel])
    @pytest.mark.parametrize("instance_key", [None, "cell_id"])
    @pytest.mark.parametrize("feature_key", [None, "target"])
    @pytest.mark.parametrize("typ", [np.ndarray, pd.DataFrame, dd.DataFrame])
    @pytest.mark.parametrize("is_annotation", [True, False])
    @pytest.mark.parametrize("is_3d", [True, False])
    @pytest.mark.parametrize("coordinates", [None, {"x": "A", "y": "B", "z": "C"}])
    def test_points_model(
        self,
        model: PointsModel,
        typ: Any,
        is_3d: bool,
        is_annotation: bool,
        instance_key: str | None,
        feature_key: str | None,
        coordinates: dict[str, str] | None,
    ) -> None:
        if typ is np.ndarray and coordinates is not None:
            # the case np.ndarray ignores the coordinates argument
            return
        if coordinates is not None:
            coordinates = coordinates.copy()
        coords = ["A", "B", "C", "x", "y", "z"]
        n = 10
        data = pd.DataFrame(RNG.integers(0, 101, size=(n, 6)), columns=coords)
        data["target"] = pd.Series(RNG.integers(0, 2, size=(n,))).astype(str)
        data["cell_id"] = pd.Series(RNG.integers(0, 5, size=(n,))).astype(np.int_)
        data["anno"] = pd.Series(RNG.integers(0, 1, size=(n,))).astype(np.int_)
        # to test for non-contiguous indices
        data.drop(index=2, inplace=True)
        if not is_3d:
            if coordinates is not None:
                del coordinates["z"]
            else:
                del data["z"]
        if typ == np.ndarray:
            axes = ["x", "y"]
            if is_3d:
                axes += ["z"]
            numpy_coords = data[axes].to_numpy()
            self._parse_transformation_from_multiple_places(model, numpy_coords)
            points = model.parse(
                numpy_coords,
                annotation=data,
                instance_key=instance_key,
                feature_key=feature_key,
            )
            self._passes_validation_after_io(model, points, "points")
        elif typ == pd.DataFrame:
            self._parse_transformation_from_multiple_places(model, data)
            points = model.parse(
                data,
                coordinates=coordinates,
                instance_key=instance_key,
                feature_key=feature_key,
            )
            self._passes_validation_after_io(model, points, "points")
        elif typ == dd.DataFrame:
            dd_data = dd.from_pandas(data, npartitions=2)
            self._parse_transformation_from_multiple_places(model, dd_data, coordinates=coordinates)
            points = model.parse(
                dd_data,
                coordinates=coordinates,
                instance_key=instance_key,
                feature_key=feature_key,
            )
            if coordinates is not None:
                axes = get_axes_names(points)
                for axis in axes:
                    assert np.array_equal(points[axis], data[coordinates[axis]])
            self._passes_validation_after_io(model, points, "points")
        assert np.all(points.index.compute() == data.index)
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
        region: str | np.ndarray,
    ) -> None:
        region_key = "reg"
        obs = pd.DataFrame(
            RNG.choice(np.arange(0, 100, dtype=float), size=(10, 3), replace=False), columns=["A", "B", "C"]
        )
        obs[region_key] = region
        adata = AnnData(RNG.normal(size=(10, 2)), obs=obs)
        with pytest.raises(TypeError, match="Only int"):
            model.parse(adata, region=region, region_key=region_key, instance_key="A")

        obs = pd.DataFrame(RNG.choice(np.arange(0, 100), size=(10, 3), replace=False), columns=["A", "B", "C"])
        obs[region_key] = region
        adata = AnnData(RNG.normal(size=(10, 2)), obs=obs)
        table = model.parse(adata, region=region, region_key=region_key, instance_key="A")
        assert region_key in table.obs
        assert isinstance(table.obs[region_key].dtype, pd.CategoricalDtype)
        assert table.obs[region_key].cat.categories.tolist() == np.unique(region).tolist()
        assert table.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY_KEY] == region_key
        assert table.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY] == region
        assert TableModel.ATTRS_KEY in table.uns
        assert TableModel.REGION_KEY in table.uns[TableModel.ATTRS_KEY]
        assert TableModel.REGION_KEY_KEY in table.uns[TableModel.ATTRS_KEY]
        assert table.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY] == region

    @pytest.mark.parametrize("model", [TableModel])
    @pytest.mark.parametrize("region", [["sample_1"] * 5 + ["sample_2"] * 5])
    def test_table_instance_key_values_not_unique(self, model: TableModel, region: str | np.ndarray):
        region_key = "region"
        obs = pd.DataFrame(RNG.integers(0, 100, size=(10, 3)), columns=["A", "B", "C"])
        obs[region_key] = region
        obs["A"] = [1] * 5 + list(range(5))
        adata = AnnData(RNG.normal(size=(10, 2)), obs=obs)
        with pytest.raises(ValueError, match=re.escape("Instance key column for region(s) `sample_1`")):
            model.parse(adata, region=region, region_key=region_key, instance_key="A")

        adata.obs["A"] = [1] * 10
        with pytest.raises(ValueError, match=re.escape("Instance key column for region(s) `sample_1, sample_2`")):
            model.parse(adata, region=region, region_key=region_key, instance_key="A")


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


def test_model_polygon_z():
    import geopandas as gpd
    from shapely.geometry import Polygon

    polygon = Polygon([(0, 0, 0), (1, 1, 0), (2, 0, 0)])

    with pytest.warns(
        UserWarning,
        match="The geometry column of the GeoDataFrame has 3 dimensions, while 2 is expected. Please consider",
    ):
        _ = ShapesModel.parse(gpd.GeoDataFrame(geometry=[polygon]))


def test_force2d():
    # let's create a shapes object (circles) constructed from 3D points (let's mix 2D and 3D)
    circles_3d = ShapesModel.parse(GeoDataFrame({"geometry": (Point(1, 1, 1), Point(2, 2)), "radius": [2, 2]}))

    polygon1 = Polygon([(0, 0, 0), (1, 0, 0), (1, 1, 0)])
    polygon2 = Polygon([(0, 0), (1, 0), (1, 1)])

    # let's create a shapes object (polygons) constructed from 3D polygons
    polygons_3d = ShapesModel.parse(GeoDataFrame({"geometry": [polygon1, polygon2]}))

    # let's create a shapes object (multipolygons) constructed from 3D multipolygons
    multipolygons_3d = ShapesModel.parse(GeoDataFrame({"geometry": [MultiPolygon([polygon1, polygon2])]}))

    force_2d(circles_3d)
    force_2d(polygons_3d)
    force_2d(multipolygons_3d)

    expected_circles_2d = ShapesModel.parse(GeoDataFrame({"geometry": (Point(1, 1), Point(2, 2)), "radius": [2, 2]}))
    expected_polygons_2d = ShapesModel.parse(
        GeoDataFrame({"geometry": [Polygon([(0, 0), (1, 0), (1, 1)]), Polygon([(0, 0), (1, 0), (1, 1)])]})
    )
    expected_multipolygons_2d = ShapesModel.parse(
        GeoDataFrame(
            {"geometry": [MultiPolygon([Polygon([(0, 0), (1, 0), (1, 1)]), Polygon([(0, 0), (1, 0), (1, 1)])])]}
        )
    )

    assert_elements_are_identical(circles_3d, expected_circles_2d)
    assert_elements_are_identical(polygons_3d, expected_polygons_2d)
    assert_elements_are_identical(multipolygons_3d, expected_multipolygons_2d)


def test_dask_points_unsorted_index_with_warning(points):
    chunksize = 300
    element = points["points_0"]
    new_order = RNG.permutation(len(element))
    with pytest.warns(
        UserWarning,
        match=r"The index of the dataframe is not monotonic increasing\.",
    ):
        ordered = PointsModel.parse(element.compute().iloc[new_order, :], chunksize=chunksize)
        assert np.all(ordered.index.compute().to_numpy() == new_order)


@pytest.mark.xfail(reason="Not supporting multiple partitions when the index is not sorted.")
def test_dask_points_unsorted_index_with_xfail(points):
    chunksize = 150
    element = points["points_0"]
    new_order = RNG.permutation(len(element))
    with pytest.raises(
        ValueError,
        match=r"Not all divisions are known, can't align partitions. Please use `set_index` to set the index.",
    ):
        _ = PointsModel.parse(element.compute().iloc[new_order, :], chunksize=chunksize)
    raise ValueError("pytest.raises caught an exceptionG")


# helper function to create random points data and write to a Parquet file, used in the test below
def create_parquet_file(temp_dir, num_points=20, sorted_index=True):
    df = pd.DataFrame(
        {"x": RNG.uniform(size=num_points), "y": RNG.uniform(size=num_points), "z": RNG.uniform(size=num_points)}
    )
    if not sorted_index:
        new_order = RNG.permutation(len(df))
        df = df.iloc[new_order, :]
    file_path = f"{temp_dir}/points.parquet"
    df.to_parquet(file_path)
    return file_path


# this test was added because the xenium() reader (which reads a .parquet file into a dask-dataframe, was failing before
# https://github.com/scverse/spatialdata/pull/656.
# Luca: actually, this test is not able to reproduce the issue; anyway this PR fixes the issue and I'll still keep the
# test here as an explicit test for unsorted index in the case of dask dataframes.
@pytest.mark.parametrize("npartitions", [1, 2])
@pytest.mark.parametrize("sorted_index", [True, False])
def test_dask_points_from_parquet(points, npartitions: int, sorted_index: bool):
    with TemporaryDirectory() as temp_dir:
        file_path = create_parquet_file(temp_dir, sorted_index=sorted_index)
        points = dd.read_parquet(file_path)

        if sorted_index:
            _ = PointsModel.parse(points, npartitions=npartitions)
            assert np.all(points.index.compute().to_numpy() == np.arange(len(points)))
        else:
            with pytest.warns(
                UserWarning,
                match=r"The index of the dataframe is not monotonic increasing\.",
            ):
                _ = PointsModel.parse(points, npartitions=npartitions)
