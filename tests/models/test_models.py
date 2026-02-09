import os
import re
import tempfile
import warnings
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
from spatialdata._core.validation import ValidationError
from spatialdata._types import ArrayLike
from spatialdata.config import LARGE_CHUNK_THRESHOLD_BYTES
from spatialdata.models import get_table_keys
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
from spatialdata.transformations._utils import _set_transformations
from spatialdata.transformations.operations import get_transformation, set_transformation
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
            model.validate(element_read)

    @pytest.mark.parametrize("converter", [lambda _: _, from_array, DataArray, to_spatial_image])
    @pytest.mark.parametrize("model", [Image2DModel, Labels2DModel, Labels3DModel, Image3DModel])
    @pytest.mark.parametrize("permute", [True, False])
    @pytest.mark.parametrize("kwargs", [None, {"name": "test"}])
    def test_raster_schema(
        self,
        converter: Callable[..., Any],
        model: RasterSchema,
        permute: bool,
        kwargs: dict[str, str] | None,
    ) -> None:
        dims = np.array(model.dims).tolist()
        if permute:
            RNG.shuffle(dims)
        n_dims = len(dims)

        if converter is DataArray:
            converter = partial(converter, dims=dims)
        elif converter is to_spatial_image:
            converter = partial(converter, dims=model.dims)
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

    @pytest.mark.parametrize(
        "model,chunks,expected",
        [
            (Labels2DModel, None, (10, 10)),
            (Labels2DModel, 5, (5, 5)),
            (Labels2DModel, (5, 5), (5, 5)),
            (Labels2DModel, {"x": 5, "y": 5}, (5, 5)),
            (Labels3DModel, None, (2, 10, 10)),
            (Labels3DModel, 5, (2, 5, 5)),
            (Labels3DModel, (2, 5, 5), (2, 5, 5)),
            (Labels3DModel, {"z": 2, "x": 5, "y": 5}, (2, 5, 5)),
            (Image2DModel, None, (1, 10, 10)),  # Image2D Models always have a c dimension
            (Image2DModel, 5, (1, 5, 5)),
            (Image2DModel, (1, 5, 5), (1, 5, 5)),
            (Image2DModel, {"c": 1, "x": 5, "y": 5}, (1, 5, 5)),
            (Image3DModel, None, (1, 2, 10, 10)),  # Image3D models have z in addition, so 4 total dimensions
            (Image3DModel, 5, (1, 2, 5, 5)),
            (Image3DModel, (1, 2, 5, 5), (1, 2, 5, 5)),
            (
                Image3DModel,
                {"c": 1, "z": 2, "x": 5, "y": 5},
                (1, 2, 5, 5),
            ),
        ],
    )
    def test_raster_models_parse_with_chunks_parameter(self, model, chunks, expected):
        image: ArrayLike = np.arange(100).reshape((10, 10))
        if model in [Labels3DModel, Image3DModel]:
            image = np.stack([image] * 2)

        if model in [Image2DModel, Image3DModel]:
            image = np.expand_dims(image, axis=0)

        # parse as numpy array
        # single scale
        x_ss = model.parse(image, chunks=chunks)
        assert x_ss.data.chunksize == expected
        # multi scale
        x_ms = model.parse(image, chunks=chunks, scale_factors=(2,))
        assert x_ms["scale0"]["image"].data.chunksize == expected

        # parse as dask array
        dask_image = from_array(image)
        # single scale
        y_ss = model.parse(dask_image, chunks=chunks)
        assert y_ss.data.chunksize == expected
        # multi scale
        y_ms = model.parse(dask_image, chunks=chunks, scale_factors=(2,))
        assert y_ms["scale0"]["image"].data.chunksize == expected

        # parse as DataArray
        data_array = DataArray(image, dims=model.dims)
        # single scale
        z_ss = model.parse(data_array, chunks=chunks)
        assert z_ss.data.chunksize == expected
        # multi scale
        z_ms = model.parse(data_array, chunks=chunks, scale_factors=(2,))
        assert z_ms["scale0"]["image"].data.chunksize == expected

    @pytest.mark.parametrize("model", [Labels2DModel, Labels3DModel])
    def test_labels_model_with_multiscales(self, model):
        # Passing "scale_factors" should generate multiscales with a "method" appropriate for labels
        dims = np.array(model.dims).tolist()
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
        assert set(np.unique(image)) >= set(np.unique(actual.scale1.image)), (
            "Subsequent scales should not have interpolation artifacts"
        )

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
            poly.loc[0, ShapesModel.RADIUS_KEY] = -1
            with pytest.raises(ValueError, match="Radii of circles must be positive."):
                ShapesModel.validate(poly)
            poly.loc[0, ShapesModel.RADIUS_KEY] = 0
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
    @pytest.mark.parametrize("region", [["sample"] * 10, RNG.choice([1, 2], size=10).tolist()])
    def test_table_model(
        self,
        model: TableModel,
        region: str | np.ndarray,
    ) -> None:
        region_key = "reg"
        obs = pd.DataFrame(
            RNG.choice(np.arange(0, 100, dtype=float), size=(10, 3), replace=False),
            columns=["A", "B", "C"],
            index=list(map(str, range(10))),
        )
        obs[region_key] = pd.Categorical(region)
        adata = AnnData(RNG.normal(size=(10, 2)), obs=obs)
        with pytest.raises(TypeError, match="Only int"):
            model.parse(adata, region=region, region_key=region_key, instance_key="A")

        obs = pd.DataFrame(
            RNG.choice(np.arange(0, 100), size=(10, 3), replace=False),
            columns=["A", "B", "C"],
            index=list(map(str, range(10))),
        )
        obs[region_key] = pd.Categorical(region)
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

        # error when trying to parse a table by specifying region, region_key, instance_key, but these keys are
        # already set
        with pytest.raises(ValueError, match=" has already been set"):
            _ = TableModel.parse(adata, region=region, region_key=region_key, instance_key="A")

        # error when region is missing
        with pytest.raises(ValueError, match="`region` must be provided"):
            _ = TableModel.parse(adata, region_key=region_key, instance_key="A", overwrite_metadata=True)

        # error when region_key is missing
        with pytest.raises(ValueError, match="`region_key` must be provided"):
            _ = TableModel.parse(adata, region=region, instance_key="A", overwrite_metadata=True)

        # error when instance_key is missing
        with pytest.raises(ValueError, match="`instance_key` must be provided"):
            _ = TableModel.parse(adata, region=region, region_key=region_key, overwrite_metadata=True)

        # we try to overwrite, but the values in the `region_key` column do not match the expected `region` values
        with pytest.raises(ValueError, match="values do not match with `region` values"):
            _ = TableModel.parse(adata, region="element", region_key="B", instance_key="C", overwrite_metadata=True)

        # we correctly overwrite; here we check that the metadata is updated
        region_, region_key_, instance_key_ = get_table_keys(table)
        assert region_ == region
        assert region_key_ == region_key
        assert instance_key_ == "A"

        # let's fix the region_key column
        table.obs["B"] = pd.Categorical(["element"] * len(table))
        _ = TableModel.parse(adata, region="element", region_key="B", instance_key="C", overwrite_metadata=True)

        region_, region_key_, instance_key_ = get_table_keys(table)
        assert region_ == "element"
        assert region_key_ == "B"
        assert instance_key_ == "C"

        # we can parse a table when no metadata is present (i.e. the table does not annotate any element)
        del table.uns[TableModel.ATTRS_KEY]
        _ = TableModel.parse(table)

    @pytest.mark.parametrize(
        "name",
        [
            "",
            ".",
            "..",
            "__dunder",
            "has whitespace",
            "path/separator",
            "non-alnum_#$%&()*+,?@",
        ],
    )
    @pytest.mark.parametrize("element_type", ["images", "labels", "points", "shapes", "tables"])
    def test_model_invalid_names(self, full_sdata, element_type: str, name: str):
        element = next(iter(getattr(full_sdata, element_type).values()))
        with pytest.raises(ValueError, match="Name (must|cannot)"):
            SpatialData(**{element_type: {name: element}})

    @pytest.mark.parametrize(
        "names",
        [
            ["abc", "Abc"],
        ],
    )
    @pytest.mark.parametrize("element_type", ["images", "labels", "points", "shapes", "tables"])
    def test_model_not_unique_names(self, full_sdata, element_type: str, names: list[str]):
        element = next(iter(getattr(full_sdata, element_type).values()))
        with pytest.raises(ValidationError, match="Key `.*` is not unique"):
            SpatialData(**{element_type: dict.fromkeys(names, element)})

    @pytest.mark.parametrize("model", [TableModel])
    @pytest.mark.parametrize("region", [["sample_1"] * 5 + ["sample_2"] * 5])
    def test_table_instance_key_values_not_unique(self, model: TableModel, region: str | np.ndarray):
        region_key = "region"
        obs = pd.DataFrame(RNG.integers(0, 100, size=(10, 3)), columns=["A", "B", "C"], index=list(map(str, range(10))))
        obs[region_key] = region
        obs["A"] = [1] * 5 + list(range(5))
        adata = AnnData(RNG.normal(size=(10, 2)), obs=obs)
        with pytest.raises(ValueError, match=re.escape("Instance key column for region(s) `sample_1`")):
            model.parse(adata, region=region, region_key=region_key, instance_key="A")

        adata.obs["A"] = [1] * 10
        with pytest.raises(
            ValueError,
            match=re.escape("Instance key column for region(s) `sample_1, sample_2`"),
        ):
            model.parse(adata, region=region, region_key=region_key, instance_key="A", overwrite_metadata=True)

    @pytest.mark.parametrize(
        "key",
        [
            "",
            ".",
            "..",
            "__dunder",
            "_index",
            "has whitespace",
            "path/separator",
            "non-alnum_#$%&()*+,?@",
        ],
    )
    @pytest.mark.parametrize("attr", ["obs", "obsm", "obsp", "var", "varm", "varp", "uns", "layers"])
    @pytest.mark.parametrize("parse", [True, False])
    def test_table_model_invalid_names(self, key: str, attr: str, parse: bool):
        if attr in ("obs", "var"):
            df = pd.DataFrame([[None]], columns=[key], index=["1"])
            adata = AnnData(np.array([[0]]), **{attr: df})
            with pytest.raises(
                ValueError,
                match=f"Table contains invalid names(.|\n)*\n  {attr}/{re.escape(key)}",
            ):
                if parse:
                    TableModel.parse(adata)
                else:
                    TableModel.validate(adata)
        elif key != "_index":  # "_index" is only disallowed in obs/var
            if attr in ("obsm", "varm", "obsp", "varp", "layers"):
                array = np.array([[0]])
                adata = AnnData(np.array([[0]]), **{attr: {key: array}})
                with pytest.raises(
                    ValueError,
                    match=f"Table contains invalid names(.|\n)*\n  {attr}/{re.escape(key)}",
                ):
                    if parse:
                        TableModel.parse(adata)
                    else:
                        TableModel.validate(adata)
            elif attr == "uns":
                adata = AnnData(np.array([[0]]), **{attr: {key: {}}})
                with pytest.raises(
                    ValueError,
                    match=f"Table contains invalid names(.|\n)*\n  {attr}/{re.escape(key)}",
                ):
                    if parse:
                        TableModel.parse(adata)
                    else:
                        TableModel.validate(adata)

    @pytest.mark.parametrize(
        "keys",
        [
            ["abc", "abc"],
            ["abc", "Abc", "ABC"],
        ],
    )
    @pytest.mark.parametrize("attr", ["obs", "var"])
    @pytest.mark.parametrize("parse", [True, False])
    def test_table_model_not_unique_columns(self, keys: list[str], attr: str, parse: bool):
        invalid_key = keys[1]
        df = pd.DataFrame([[None] * len(keys)], columns=keys, index=["1"])
        adata = AnnData(np.array([[0]]), **{attr: df})
        with pytest.raises(ValueError, match=f"Table contains invalid names(.|\n)*\n  {attr}/{invalid_key}: "):
            if parse:
                TableModel.parse(adata)
            else:
                TableModel.validate(adata)


def test_validate_set_instance_key_missing_attrs():
    """Test _validate_set_instance_key behavior when ATTRS_KEY is missing from uns."""
    # When instance_key arg is provided and column exists, but attrs is missing, it should fail
    adata = AnnData(np.array([[0]]), obs=pd.DataFrame({"instance_id": [1]}, index=["1"]))
    with pytest.raises(ValueError, match="No 'spatialdata_attrs' found"):
        TableModel._validate_set_instance_key(adata, instance_key="instance_id")

    # When instance_key arg is provided but column doesn't exist, should raise about the column
    adata2 = AnnData(np.array([[0]]))
    adata2.uns[TableModel.ATTRS_KEY] = {}
    with pytest.raises(ValueError, match="Instance key column 'missing' not found"):
        TableModel._validate_set_instance_key(adata2, instance_key="missing")

    # When no instance_key arg and no attrs, should raise about missing attrs
    with pytest.raises(ValueError, match="No 'spatialdata_attrs' found"):
        TableModel._validate_set_instance_key(adata)


def test_validate_set_region_key_missing_attrs():
    """Test _validate_set_region_key behavior when ATTRS_KEY is missing from uns."""
    # When region_key arg is provided and column exists, but attrs is missing, it should fail
    adata = AnnData(np.array([[0]]), obs=pd.DataFrame({"region": ["r1"]}, index=["1"]))
    with pytest.raises(ValueError, match="No 'spatialdata_attrs' found"):
        TableModel._validate_set_region_key(adata, region_key="region")

    # When region_key arg is provided but column doesn't exist, should raise about the column
    adata2 = AnnData(np.array([[0]]))
    adata2.uns[TableModel.ATTRS_KEY] = {}
    with pytest.raises(ValueError, match="column not present in table.obs"):
        TableModel._validate_set_region_key(adata2, region_key="missing")

    # When no region_key arg and no attrs, should raise about missing attrs
    with pytest.raises(ValueError, match="No 'spatialdata_attrs' found"):
        TableModel._validate_set_region_key(adata)


def test_get_schema():
    images = _get_images()
    labels = _get_labels()
    points = _get_points()
    shapes = _get_shapes()
    table = _get_table(region="any", region_key="region", instance_key="instance_id")
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
        GeoDataFrame(
            {
                "geometry": [
                    Polygon([(0, 0), (1, 0), (1, 1)]),
                    Polygon([(0, 0), (1, 0), (1, 1)]),
                ]
            }
        )
    )
    expected_multipolygons_2d = ShapesModel.parse(
        GeoDataFrame(
            {
                "geometry": [
                    MultiPolygon(
                        [
                            Polygon([(0, 0), (1, 0), (1, 1)]),
                            Polygon([(0, 0), (1, 0), (1, 1)]),
                        ]
                    )
                ]
            }
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
        {
            "x": RNG.uniform(size=num_points),
            "y": RNG.uniform(size=num_points),
            "z": RNG.uniform(size=num_points),
        }
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


@pytest.mark.parametrize("scale_factors", [None, [2, 2]])
def test_c_coords_2d(scale_factors: list[int] | None):
    data = np.zeros((3, 30, 30))
    model = Image2DModel().parse(data, c_coords=["1st", "2nd", "3rd"], scale_factors=scale_factors)
    if scale_factors is None:
        assert model.coords["c"].data.tolist() == ["1st", "2nd", "3rd"]
    else:
        assert all(
            model[group]["image"].coords["c"].data.tolist() == ["1st", "2nd", "3rd"] for group in list(model.keys())
        )

    with pytest.raises(ValueError, match="The number of channel names"):
        Image2DModel().parse(
            data,
            c_coords=["1st", "2nd", "3rd", "too_much"],
            scale_factors=scale_factors,
        )


@pytest.mark.parametrize("model", [Labels2DModel, Labels3DModel])
def test_label_no_c_coords(model: Labels2DModel | Labels3DModel):
    with pytest.raises(ValueError, match="`c_coords` is not supported"):
        model().parse(np.zeros((30, 30)), c_coords=["1st", "2nd", "3rd"])


def test_warning_on_large_chunks():
    data_small = DataArray(dask.array.zeros((100, 100), chunks=(50, 50)), dims=["x", "y"])
    data_large = DataArray(dask.array.zeros((50000, 50000), chunks=(50000, 50000)), dims=["x", "y"])
    assert np.array(data_large.shape).prod().item() > LARGE_CHUNK_THRESHOLD_BYTES

    # single and multiscale, small chunk size
    with warnings.catch_warnings(record=True) as w:
        _ = Labels2DModel.parse(data_small)
        _ = Labels2DModel.parse(data_small, scale_factors=[2, 2])
        # method 'xarray_coarsen' is used to downsample the data lazily (otherwise the test would be too slow)
        _ = Labels2DModel.parse(data_large, scale_factors=[2, 2], method="xarray_coarsen")
        warnings.simplefilter("always")
        assert len(w) == 0, "Warning should not be raised for small chunk size"

    # single scale, large chunk size
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = Labels2DModel.parse(data_large)
        assert len(w) == 1, "Warning should be raised for large chunk size"
        assert issubclass(w[-1].category, UserWarning)
        assert "Detected chunks larger than:" in str(w[-1].message)

    # multiscale, large chunk size
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        multiscale = Labels2DModel.parse(data_large, scale_factors=[2, 2], method="xarray_coarsen")
        multiscale = multiscale.chunk({"x": 50000, "y": 50000})
        Labels2DModel.validate(multiscale)
        assert len(w) == 1, "Warning should be raised for large chunk size"
        assert issubclass(w[-1].category, UserWarning)
        assert "Detected chunks larger than:" in str(w[-1].message)


def test_categories_on_partitioned_dataframe(sdata_blobs: SpatialData):
    df = sdata_blobs["blobs_points"].compute()
    df["genes"] = RNG.choice([f"gene_{i}" for i in range(200)], len(df))
    N_PARTITIONS = 200
    ddf = dd.from_pandas(df, npartitions=N_PARTITIONS)
    ddf["genes"] = ddf["genes"].astype("category")

    df["genes"] = df["genes"].astype("category")
    df_parsed = PointsModel.parse(df, npartitions=N_PARTITIONS)
    ddf_parsed = PointsModel.parse(ddf, npartitions=N_PARTITIONS)

    assert df["genes"].equals(df_parsed["genes"].compute())
    assert df["genes"].cat.categories.equals(df_parsed["genes"].compute().cat.categories)

    assert np.array_equal(df["genes"].to_numpy(), ddf_parsed["genes"].compute().to_numpy())
    assert set(df["genes"].cat.categories.tolist()) == set(ddf_parsed["genes"].compute().cat.categories.tolist())

    # two behavior to investigate later/report to dask (they originate in dask)
    # TODO: df['genes'].cat.categories has dtype 'object', while ddf_parsed['genes'].compute().cat.categories has dtype
    #  'string'
    # this problem should disappear after pandas 3.0 is released
    assert df["genes"].cat.categories.dtype == "object"
    assert ddf_parsed["genes"].compute().cat.categories.dtype == "string"

    # TODO: the list of categories are not preserving the order
    assert df["genes"].cat.categories.tolist() != ddf_parsed["genes"].compute().cat.categories.tolist()
