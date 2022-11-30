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
    Labels2DModel,
    Labels3DModel,
    PointsModel,
    PolygonsModel,
    ShapesModel,
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
def empty_points() -> SpatialData:
    adata = AnnData(
        shape=(0, 3), obsm={PointsModel.COORDS_KEY: np.zeros((0, 2))}, var=pd.DataFrame(index=["a", "b", "c"])
    )
    from spatialdata import Identity

    adata.uns[PointsModel.TRANSFORM_KEY] = Identity()
    return SpatialData(points={"empty": adata})


@pytest.fixture()
def empty_table() -> SpatialData:
    adata = AnnData(shape=(0, 0))
    adata = TableModel.parse(data=adata)
    return SpatialData(table=adata)


@pytest.fixture(
    # params=["labels"]
    params=["full", "empty"]
    + ["images", "labels", "points", "table_single_annotation", "table_multiple_annotations"]
    + ["empty_" + x for x in ["points", "table"]]
)
def sdata(request) -> SpatialData:
    if request.param == "full":
        s = SpatialData(
            images=_get_images(),
            labels=_get_labels(),
            polygons=_get_polygons(),
            shapes=_get_shapes(),
            points=_get_points(),
            table=_get_table(),
        )
    elif request.param == "empty":
        s = SpatialData()
    else:
        s = request.getfixturevalue(request.param)
    print(f"request.param = {request.param}")
    print(s)
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
    out["labels2d"] = Labels2DModel.parse(RNG.normal(size=(64, 64)), name="labels2d")
    out["labels2d_multiscale"] = Labels2DModel.parse(
        RNG.normal(size=(64, 64)), name="labels2d_multiscale", scale_factors=[2, 4]
    )
    out["labels3d"] = Labels3DModel.parse(RNG.normal(size=(10, 64, 64)), name="labels3d")
    out["labels3d_multiscale"] = Labels3DModel.parse(
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

    out["poly"] = PolygonsModel.parse(poly, name="poly")
    out["multipoly"] = PolygonsModel.parse(multipoly, name="multipoly")

    return out


def _get_shapes() -> Mapping[str, Sequence[NDArray]]:
    name = "shapes"
    shape_type = ["Circle", "Square"]
    shape_size = [1.0, 2.0]

    assert len(shape_type) == len(shape_size)

    out = {}
    for i, (typ, size) in enumerate(zip(shape_type, shape_size)):
        name_ = f"{name}_{i}"
        arr = RNG.normal(size=(100, 2))
        out[name_] = ShapesModel.parse(arr, shape_type=typ, shape_size=size)

    return out


def _get_points() -> Mapping[str, Sequence[NDArray]]:
    name = "points"
    var_names = [np.arange(3), ["genex", "geney"]]

    out = {}
    for i, v in enumerate(var_names):
        name = f"{name}_{i}"
        arr = RNG.normal(size=(100, 2))
        # randomly assign some values from v to the points
        points_assignment = RNG.choice(v, size=arr.shape[0])
        out[name] = PointsModel.parse(coords=arr, var_names=v, points_assignment=points_assignment)
    return out


def _get_table(
    region: Optional[Union[str, Sequence[str]]] = None,
    region_key: Optional[str] = None,
    instance_key: Optional[str] = None,
) -> AnnData:
    region_key = region_key or "annotated_region"
    instance_key = instance_key or "instance_id"
    adata = AnnData(RNG.normal(size=(100, 10)), obs=pd.DataFrame(RNG.normal(size=(100, 3)), columns=["a", "b", "c"]))
    adata.obs[instance_key] = np.arange(adata.n_obs)
    if isinstance(region, str):
        return TableModel.parse(data=adata, region=region, instance_key=instance_key)
    elif isinstance(region, list):
        adata.obs[region_key] = RNG.choice(region, size=adata.n_obs)
        adata.obs[instance_key] = RNG.integers(0, 10, size=(100,))
        return TableModel.parse(data=adata, region=region, region_key=region_key, instance_key=instance_key)
