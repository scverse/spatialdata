from typing import Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from geopandas import GeoDataFrame
from numpy.random import default_rng
from shapely.geometry import MultiPolygon, Polygon

from spatialdata import SpatialData
from spatialdata._core.models import (
    Image2DModel,
    Label2DModel,
    Label3DModel,
    PointModel,
    PolygonModel,
    ShapeModel,
    TableModel,
)
from spatialdata._types import NDArray

RNG = default_rng()


@pytest.fixture()
def images() -> SpatialData:
    return SpatialData(images=_get_images())


@pytest.fixture()
def labels() -> SpatialData:
    return SpatialData(labels=_get_labels())


@pytest.fixture()
def polygons() -> SpatialData:
    return SpatialData(polygons=_get_polygons())


@pytest.fixture()
def shapes() -> SpatialData:
    return SpatialData(shapes=_get_shapes(2, name="shapes", shape_type=["Circle", "Square"], shape_size=[1, 2]))


@pytest.fixture()
def points() -> SpatialData:
    return SpatialData(points=_get_points(2, name="points", var_names=[np.arange(3), ["genex", "geney"]]))


@pytest.fixture()
def table() -> SpatialData:
    return SpatialData(table=_get_table(region="sample1"))


@pytest.fixture()
def empty_images() -> SpatialData:
    pytest.skip("empty images not supported")
    return SpatialData(images={"empty": np.zeros((0, 0, 0))})


@pytest.fixture()
def empty_labels() -> SpatialData:
    pytest.skip("empty labels not supported")
    return SpatialData(labels={"empty": np.zeros((0, 0), dtype=int)})


@pytest.fixture()
def empty_points() -> SpatialData:
    return SpatialData(points={"empty": AnnData(shape=(0, 0), obsm={"spatial": np.zeros((0, 2))})})


@pytest.fixture()
def empty_table() -> SpatialData:
    return SpatialData(table=AnnData(shape=(0, 0)))


@pytest.fixture(
    # params=["labels"]
    params=["full"]
    + ["images", "labels", "points", "table"]
    + ["empty_" + x for x in ["images", "labels", "points", "table"]]
)
def sdata(request) -> SpatialData:
    if request.param == "full":
        s = SpatialData(
            images=_get_images(),
            labels=_get_labels(),
            polygons=_get_polygons(3, name="polygons"),
            points=_get_points(2),
            table=_get_table(),
        )
    else:
        s = request.getfixturevalue(request.param)
    return s


def _get_images() -> Mapping[str, Sequence[NDArray]]:
    out = {}
    out["image2d"] = Image2DModel.parse(RNG.normal(size=(3, 64, 64)), name="image2d")
    out["image2d_multiscale"] = Image2DModel.parse(
        RNG.normal(size=(3, 64, 64)), name="image2d_multiscale", scale_factors=[2, 4]
    )
    # TODO: not supported atm.
    # out["image3d"] = Image3DModel.parse(RNG.normal(size=(2, 64, 64, 3)), name="image3d")
    # out["image3d_multiscale"] = Image3DModel.parse(
    #     RNG.normal(size=(2, 64, 64, 3)), name="image3d_multiscale", scale_factors=[2, 4]
    # )
    return out


def _get_labels() -> Mapping[str, Sequence[NDArray]]:
    out = {}
    out["labels2d"] = Label2DModel.parse(RNG.normal(size=(64, 64)), name="labels2d")
    out["labels2d_multiscale"] = Label2DModel.parse(
        RNG.normal(size=(64, 64)), name="labels2d_multiscale", scale_factors=[2, 4]
    )
    out["labels3d"] = Label3DModel.parse(RNG.normal(size=(10, 64, 64)), name="labels3d")
    out["labels3d_multiscale"] = Label3DModel.parse(
        RNG.normal(size=(10, 64, 64)), name="labels3d_multiscale", scale_factors=[2, 4]
    )
    return out


def _get_polygons() -> Mapping[str, Sequence[NDArray]]:

    out = {}
    poly = GeoDataFrame(
        {
            "geometry": [
                Polygon(((0, 0), (0, 1), (1, 1), (1, 0))),
                Polygon(((0, 0), (0, -1), (-1, -1), (-1, 0))),
                Polygon(((0, 0), (0, 1), (1, 10))),
                Polygon(((0, 0), (0, 1), (1, 1))),
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

    out["poly"] = PolygonModel.parse(poly, name="poly")
    out["multipoly"] = PolygonModel.parse(multipoly, name="multipoly")

    return out


def _get_shapes(
    n: int, name: str, shape_type: Sequence[str], shape_size: Sequence[str]
) -> Mapping[str, Sequence[NDArray]]:

    assert len(shape_type) == len(shape_size) == n

    out = {}
    for i, (typ, size) in enumerate(zip(shape_type, shape_size)):
        name_ = f"{name}_{i}"
        arr = RNG.normal(size=(100, 2))
        out[name_] = ShapeModel.parse(arr, shape_type=typ, shape_size=size)

    return out


def _get_points(n: int, name: str, var_names: Sequence[Sequence[str]]) -> Mapping[str, Sequence[NDArray]]:

    assert len(var_names) == n

    out = {}
    for i, v in enumerate(var_names):
        name = f"{name}_{i}"
        arr = RNG.normal(size=(100, 2))
        out[name] = PointModel.parse(arr, var_names=v)

    return out


def _get_table(
    region: Union[str, Sequence[str]], region_key: Optional[str] = None, instance_key: Optional[str] = None
) -> AnnData:
    adata = AnnData(RNG.normal(size=(100, 10)), obs=pd.DataFrame(RNG.normal(size=(100, 3)), columns=["a", "b", "c"]))
    if isinstance(region, str):
        return TableModel.parse(adata, region)
    elif isinstance(region, list):
        adata.obs[region_key] = region
        adata.obs[instance_key] = RNG.integers(0, 10, size=(100,))
        return TableModel.parse(adata, region, region_key, instance_key)
