# isort: off
import os

os.environ["USE_PYGEOS"] = "0"
# isort:on

from shapely import linearrings, polygons
from pathlib import Path
from typing import Union
from spatialdata._types import ArrayLike
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from dask.dataframe.core import DataFrame as DaskDataFrame
from geopandas import GeoDataFrame
from multiscale_spatial_image import MultiscaleSpatialImage
from numpy.random import default_rng
from shapely.geometry import MultiPolygon, Point, Polygon
from spatial_image import SpatialImage
from spatialdata import SpatialData
from spatialdata.models import (
    Image2DModel,
    Image3DModel,
    Labels2DModel,
    Labels3DModel,
    PointsModel,
    ShapesModel,
    TableModel,
)
from xarray import DataArray
from spatialdata.datasets import BlobsDataset
import geopandas as gpd
import dask.dataframe as dd
from spatialdata._utils import _deepcopy_geodataframe

RNG = default_rng(seed=0)

POLYGON_PATH = Path(__file__).parent / "data/polygon.json"
MULTIPOLYGON_PATH = Path(__file__).parent / "data/polygon.json"
POINT_PATH = Path(__file__).parent / "data/points.json"


@pytest.fixture()
def images() -> SpatialData:
    return SpatialData(images=_get_images())


@pytest.fixture()
def labels() -> SpatialData:
    return SpatialData(labels=_get_labels())


@pytest.fixture()
def shapes() -> SpatialData:
    return SpatialData(shapes=_get_shapes())


@pytest.fixture()
def points() -> SpatialData:
    return SpatialData(points=_get_points())


@pytest.fixture()
def table_single_annotation() -> SpatialData:
    return SpatialData(table=_get_table(region="sample1"))


@pytest.fixture()
def table_multiple_annotations() -> SpatialData:
    return SpatialData(table=_get_table(region=["sample1", "sample2"]))


@pytest.fixture()
def tables() -> list[AnnData]:
    _tables = []
    for region, region_key, instance_key in (
        [None, None, None],
        ["my_region0", None, "my_instance_key"],
        [["my_region0", "my_region1"], "my_region_key", "my_instance_key"],
    ):
        _tables.append(_get_table(region=region, region_key=region_key, instance_key=instance_key))
    return _tables


@pytest.fixture()
def full_sdata() -> SpatialData:
    return SpatialData(
        images=_get_images(),
        labels=_get_labels(),
        shapes=_get_shapes(),
        points=_get_points(),
        table=_get_table(region="sample1"),
    )


# @pytest.fixture()
# def empty_points() -> SpatialData:
#     geo_df = GeoDataFrame(
#         geometry=[],
#     )
#     from spatialdata import NgffIdentity
#     _set_transformations(geo_df, NgffIdentity())
#
#     return SpatialData(points={"empty": geo_df})


# @pytest.fixture()
# def empty_table() -> SpatialData:
#     adata = AnnData(shape=(0, 0), obs=pd.DataFrame(columns="region"), var=pd.DataFrame())
#     adata = TableModel.parse(adata=adata)
#     return SpatialData(table=adata)


@pytest.fixture(
    # params=["labels"]
    params=["full", "empty"]
    + ["images", "labels", "points", "table_single_annotation", "table_multiple_annotations"]
    # + ["empty_" + x for x in ["table"]] # TODO: empty table not supported yet
)
def sdata(request) -> SpatialData:
    if request.param == "full":
        return SpatialData(
            images=_get_images(),
            labels=_get_labels(),
            shapes=_get_shapes(),
            points=_get_points(),
            table=_get_table("sample1"),
        )
    if request.param == "empty":
        return SpatialData()
    return request.getfixturevalue(request.param)


def _get_images() -> dict[str, Union[SpatialImage, MultiscaleSpatialImage]]:
    out = {}
    dims_2d = ("c", "y", "x")
    dims_3d = ("z", "y", "x", "c")
    out["image2d"] = Image2DModel.parse(RNG.normal(size=(3, 64, 64)), dims=dims_2d, c_coords=["r", "g", "b"])
    out["image2d_multiscale"] = Image2DModel.parse(
        RNG.normal(size=(3, 64, 64)), scale_factors=[2, 2], dims=dims_2d, c_coords=["r", "g", "b"]
    )
    out["image2d_xarray"] = Image2DModel.parse(DataArray(RNG.normal(size=(3, 64, 64)), dims=dims_2d), dims=None)
    out["image2d_multiscale_xarray"] = Image2DModel.parse(
        DataArray(RNG.normal(size=(3, 64, 64)), dims=dims_2d),
        scale_factors=[2, 4],
        dims=None,
    )
    out["image3d_numpy"] = Image3DModel.parse(RNG.normal(size=(2, 64, 64, 3)), dims=dims_3d)
    out["image3d_multiscale_numpy"] = Image3DModel.parse(
        RNG.normal(size=(2, 64, 64, 3)), scale_factors=[2], dims=dims_3d
    )
    out["image3d_xarray"] = Image3DModel.parse(DataArray(RNG.normal(size=(2, 64, 64, 3)), dims=dims_3d), dims=None)
    out["image3d_multiscale_xarray"] = Image3DModel.parse(
        DataArray(RNG.normal(size=(2, 64, 64, 3)), dims=dims_3d),
        scale_factors=[2],
        dims=None,
    )
    return out


def _get_labels() -> dict[str, Union[SpatialImage, MultiscaleSpatialImage]]:
    out = {}
    dims_2d = ("y", "x")
    dims_3d = ("z", "y", "x")

    out["labels2d"] = Labels2DModel.parse(RNG.integers(0, 100, size=(64, 64)), dims=dims_2d)
    out["labels2d_multiscale"] = Labels2DModel.parse(
        RNG.integers(0, 100, size=(64, 64)), scale_factors=[2, 4], dims=dims_2d
    )
    out["labels2d_xarray"] = Labels2DModel.parse(
        DataArray(RNG.integers(0, 100, size=(64, 64)), dims=dims_2d), dims=None
    )
    out["labels2d_multiscale_xarray"] = Labels2DModel.parse(
        DataArray(RNG.integers(0, 100, size=(64, 64)), dims=dims_2d),
        scale_factors=[2, 4],
        dims=None,
    )
    out["labels3d_numpy"] = Labels3DModel.parse(RNG.integers(0, 100, size=(10, 64, 64)), dims=dims_3d)
    out["labels3d_multiscale_numpy"] = Labels3DModel.parse(
        RNG.integers(0, 100, size=(10, 64, 64)), scale_factors=[2, 4], dims=dims_3d
    )
    out["labels3d_xarray"] = Labels3DModel.parse(
        DataArray(RNG.integers(0, 100, size=(10, 64, 64)), dims=dims_3d), dims=None
    )
    out["labels3d_multiscale_xarray"] = Labels3DModel.parse(
        DataArray(RNG.integers(0, 100, size=(10, 64, 64)), dims=dims_3d),
        scale_factors=[2, 4],
        dims=None,
    )
    return out


def _get_shapes() -> dict[str, GeoDataFrame]:
    # TODO: add polygons from geojson and from ragged arrays since now only the GeoDataFrame initializer is tested.
    out = {}
    poly = GeoDataFrame(
        {
            "geometry": [
                Polygon(((0, 0), (0, 1), (1, 1), (1, 0))),
                Polygon(((0, 0), (0, -1), (-1, -1), (-1, 0))),
                Polygon(((0, 0), (0, 1), (1, 10))),
                Polygon(((10, 10), (10, 20), (20, 20))),
                Polygon(((0, 0), (0, 1), (1, 1), (1, 0), (1, 0))),
            ]
        }
    )

    multipoly = GeoDataFrame(
        {
            "geometry": [
                MultiPolygon(
                    [
                        Polygon(((0, 0), (0, 1), (1, 1), (1, 0))),
                        Polygon(((0, 0), (0, -1), (-1, -1), (-1, 0))),
                    ]
                ),
                MultiPolygon(
                    [
                        Polygon(((0, 0), (0, 1), (1, 10))),
                        Polygon(((0, 0), (0, 1), (1, 1))),
                        Polygon(((0, 0), (0, 1), (1, 1), (1, 0), (1, 0))),
                    ]
                ),
            ]
        }
    )

    points = GeoDataFrame(
        {
            "geometry": [
                Point((0, 1)),
                Point((1, 1)),
                Point((3, 4)),
                Point((4, 2)),
                Point((5, 6)),
            ]
        }
    )
    rng = np.random.default_rng(seed=0)
    points["radius"] = rng.normal(size=(len(points), 1))

    out["poly"] = ShapesModel.parse(poly)
    out["poly"].index = ["a", "b", "c", "d", "e"]
    out["multipoly"] = ShapesModel.parse(multipoly)
    out["circles"] = ShapesModel.parse(points)

    return out


def _get_points() -> dict[str, DaskDataFrame]:
    name = "points"
    out = {}
    for i in range(2):
        name = f"{name}_{i}"
        arr = RNG.normal(size=(300, 2))
        # randomly assign some values from v to the points
        points_assignment0 = RNG.integers(0, 10, size=arr.shape[0]).astype(np.int_)
        if i == 0:
            genes = RNG.choice(["a", "b"], size=arr.shape[0])
        else:
            # we need to test the case in which we have a categorical column with more than 127 categories, see full
            # explanation in write_points() (the parser will convert this column to a categorical since
            # feature_key="genes")
            genes = np.tile(np.array(list(map(str, range(280)))), 2)[:300]
        annotation = pd.DataFrame(
            {
                "genes": genes,
                "instance_id": points_assignment0,
            },
        )
        out[name] = PointsModel.parse(arr, annotation=annotation, feature_key="genes", instance_key="instance_id")
    return out


def _get_table(
    region: Union[str, list[str]] = "sample1",
    region_key: str = "region",
    instance_key: str = "instance_id",
) -> AnnData:
    adata = AnnData(RNG.normal(size=(100, 10)), obs=pd.DataFrame(RNG.normal(size=(100, 3)), columns=["a", "b", "c"]))
    adata.obs[instance_key] = np.arange(adata.n_obs)
    if isinstance(region, str):
        adata.obs[region_key] = region
    elif isinstance(region, list):
        adata.obs[region_key] = RNG.choice(region, size=adata.n_obs)
    return TableModel.parse(adata=adata, region=region, region_key=region_key, instance_key=instance_key)


@pytest.fixture()
def labels_blobs() -> ArrayLike:
    """Create a 2D labels."""
    return BlobsDataset()._labels_blobs()


@pytest.fixture()
def sdata_blobs() -> SpatialData:
    """Create a 2D labels."""
    from copy import deepcopy
    from spatialdata.datasets import blobs

    sdata = deepcopy(blobs(256, 300, 3))
    for k, v in sdata.shapes.items():
        sdata.shapes[k] = _deepcopy_geodataframe(v)
    from spatialdata._utils import multiscale_spatial_image_from_data_tree

    sdata.images["blobs_multiscale_image"] = multiscale_spatial_image_from_data_tree(
        sdata.images["blobs_multiscale_image"]
    )
    sdata.labels["blobs_multiscale_labels"] = multiscale_spatial_image_from_data_tree(
        sdata.labels["blobs_multiscale_labels"]
    )
    return sdata


def _make_points(coordinates: np.ndarray) -> DaskDataFrame:
    """Helper function to make a Points element."""
    k0 = int(len(coordinates) / 3)
    k1 = len(coordinates) - k0
    genes = np.hstack((np.repeat("a", k0), np.repeat("b", k1)))
    return PointsModel.parse(coordinates, annotation=pd.DataFrame({"genes": genes}), feature_key="genes")


def _make_squares(centroid_coordinates: np.ndarray, half_widths: list[float]) -> polygons:
    linear_rings = []
    for centroid, half_width in zip(centroid_coordinates, half_widths):
        min_coords = centroid - half_width
        max_coords = centroid + half_width

        linear_rings.append(
            linearrings(
                [
                    [min_coords[0], min_coords[1]],
                    [min_coords[0], max_coords[1]],
                    [max_coords[0], max_coords[1]],
                    [max_coords[0], min_coords[1]],
                ]
            )
        )
    s = polygons(linear_rings)
    polygon_series = gpd.GeoSeries(s)
    cell_polygon_table = gpd.GeoDataFrame(geometry=polygon_series)
    return ShapesModel.parse(cell_polygon_table)


def _make_circles(centroid_coordinates: np.ndarray, radius: list[float]) -> GeoDataFrame:
    return ShapesModel.parse(centroid_coordinates, geometry=0, radius=radius)


def _make_sdata_for_testing_querying_and_aggretation() -> SpatialData:
    """
    Creates a SpatialData object with many edge cases for testing querying and aggregation.

    Returns
    -------
    The SpatialData object.

    Notes
    -----
    Description of what is tested (for a quick visualization, please plot the returned SpatialData object):

        - values to query/aggregate: polygons, points, circles
        - values to query by: polygons, circles
        - the shapes are completely inside, outside, or intersecting the query region (with the centroid inside or
            outside the query region)

    Additional cases:

        - concave shape intersecting multiple times the same shape; used both as query and as value
        - shape intersecting multiple shapes; used both as query and as value
    """
    values_centroids_squares = np.array([[x * 18, 0] for x in range(8)] + [[8 * 18 + 7, 0]] + [[0, 90], [50, 90]])
    values_centroids_circles = np.array([[x * 18, 30] for x in range(8)] + [[8 * 18 + 7, 30]])
    by_centroids_squares = np.array([[119, 15], [100, 90], [150, 90], [210, 15]])
    by_centroids_circles = np.array([[24, 15], [290, 15]])
    values_points = _make_points(np.vstack((values_centroids_squares, values_centroids_circles)))

    values_squares = _make_squares(values_centroids_squares, half_widths=[6] * 9 + [15, 15])

    values_circles = _make_circles(values_centroids_circles, radius=[6] * 9)
    values_circles["categorical_in_gdf"] = pd.Categorical(["a"] * 9)
    values_circles["numerical_in_gdf"] = np.arange(9)

    by_squares = _make_squares(by_centroids_squares, half_widths=[30, 15, 15, 30])
    by_circles = _make_circles(by_centroids_circles, radius=[30, 30])

    from shapely.geometry import Polygon

    polygon = Polygon([(100, 90 - 10), (100 + 30, 90), (100, 90 + 10), (150, 90)])
    values_squares.loc[len(values_squares)] = [polygon]
    ShapesModel.validate(values_squares)

    values_squares["categorical_in_gdf"] = pd.Categorical(["a"] * 9 + ["b"] * 3)
    values_squares["numerical_in_gdf"] = np.arange(12)

    polygon = Polygon([(0, 90 - 10), (0 + 30, 90), (0, 90 + 10), (50, 90)])
    by_squares.loc[len(by_squares)] = [polygon]
    ShapesModel.validate(by_squares)

    s = pd.Series(pd.Categorical(["a"] * 9 + ["b"] * 9 + ["c"] * 2))
    values_points["categorical_in_ddf"] = dd.from_pandas(s, npartitions=1)
    s = pd.Series(RNG.random(20))
    values_points["numerical_in_ddf"] = dd.from_pandas(s, npartitions=1)

    sdata = SpatialData(
        points={"points": values_points},
        shapes={
            "values_polygons": values_squares,
            "values_circles": values_circles,
            "by_polygons": by_squares,
            "by_circles": by_circles,
        },
    )

    # generate table
    x = RNG.random((21, 1))
    region = np.array(["values_circles"] * 9 + ["values_polygons"] * 12)
    instance_id = np.array(list(range(9)) + list(range(12)))
    categorical_obs = pd.Series(pd.Categorical(["a"] * 9 + ["b"] * 9 + ["c"] * 3))
    numerical_obs = pd.Series(RNG.random(21))
    table = AnnData(
        x,
        obs=pd.DataFrame(
            {
                "region": region,
                "instance_id": instance_id,
                "categorical_in_obs": categorical_obs,
                "numerical_in_obs": numerical_obs,
            }
        ),
        var=pd.DataFrame(index=["numerical_in_var"]),
    )
    table = TableModel.parse(
        table, region=["values_circles", "values_polygons"], region_key="region", instance_key="instance_id"
    )
    sdata.table = table
    return sdata


@pytest.fixture()
def sdata_query_aggregation() -> SpatialData:
    return _make_sdata_for_testing_querying_and_aggretation()
