import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
from anndata import AnnData
from anndata.tests.helpers import assert_equal
from numpy.random import default_rng
from spatialdata import SpatialData
from spatialdata._core.operations.aggregate import aggregate
from spatialdata.models import Image2DModel, Labels2DModel, PointsModel, ShapesModel

RNG = default_rng(42)


def test_aggregate_points_by_polygons() -> None:
    points = PointsModel.parse(
        pd.DataFrame(
            {
                "x": [1.2, 2.3, 4.1, 6.0, 6.1, 8.0, 9.0],
                "y": [3.5, 4.8, 7.5, 4.0, 9.0, 5.5, 9.8],
                "gene": list("aaabbbb"),
            }
        ),
        coordinates={"x": "x", "y": "y"},
        feature_key="gene",
    )
    shapes = ShapesModel.parse(
        gpd.GeoDataFrame(
            geometry=[
                shapely.Polygon([(0.5, 7.0), (4.0, 2.0), (5.0, 8.0)]),
                shapely.Polygon([(3.0, 8.0), (7.0, 2.0), (10.0, 6.0), (7.0, 10.0)]),
            ],
            index=["shape_0", "shape_1"],
        )
    )

    result_adata = aggregate(points, shapes, "gene", agg_func="sum")
    assert result_adata.obs_names.to_list() == ["shape_0", "shape_1"]
    assert result_adata.var_names.to_list() == ["a", "b"]
    np.testing.assert_equal(result_adata.X.A, np.array([[2, 0], [1, 3]]))

    # id_key can be implicit for points
    result_adata_implicit = aggregate(points, shapes, agg_func="sum")
    assert_equal(result_adata, result_adata_implicit)


def test_aggregate_polygons_by_polygons() -> None:
    cellular = ShapesModel.parse(
        gpd.GeoDataFrame(
            geometry=[
                shapely.Polygon([(0.5, 7.0), (4.0, 2.0), (5.0, 8.0)]),
                shapely.Polygon([(3.0, 8.0), (7.0, 2.0), (10.0, 6.0), (7.0, 10.0)]),
            ],
            index=["shape_0", "shape_1"],
        )
    )
    subcellular = ShapesModel.parse(
        gpd.GeoDataFrame(
            {"structure": pd.Categorical.from_codes([0, 0, 0, 1, 1, 1, 1], ["nucleus", "mitochondria"])},
            index=[f"shape_{i}" for i in range(1, 8)],
        ).set_geometry(
            gpd.points_from_xy([1.2, 2.3, 4.1, 6.0, 6.1, 8.0, 9.0], [3.5, 4.8, 7.5, 4.0, 9.0, 5.5, 9.8]).buffer(0.1)
        )
    )

    result_adata = aggregate(subcellular, cellular, "structure", agg_func="sum")
    assert result_adata.obs_names.to_list() == ["shape_0", "shape_1"]
    assert result_adata.var_names.to_list() == ["nucleus", "mitochondria"]
    np.testing.assert_equal(result_adata.X.A, np.array([[2, 0], [1, 3]]))

    with pytest.raises(ValueError):
        aggregate(subcellular, cellular, agg_func="mean")


def test_aggregate_circles_by_polygons() -> None:
    # Basically the same as above, but not buffering explicitly
    cellular = ShapesModel.parse(
        gpd.GeoDataFrame(
            geometry=[
                shapely.Polygon([(0.5, 7.0), (4.0, 2.0), (5.0, 8.0)]),
                shapely.Polygon([(3.0, 8.0), (7.0, 2.0), (10.0, 6.0), (7.0, 10.0)]),
            ],
            index=["shape_0", "shape_1"],
        )
    )
    subcellular = ShapesModel.parse(
        gpd.GeoDataFrame(
            {
                "structure": pd.Categorical.from_codes([0, 0, 0, 1, 1, 1, 1], ["nucleus", "mitochondria"]),
                "radius": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            },
            index=[f"shape_{i}" for i in range(1, 8)],
        ).set_geometry(gpd.points_from_xy([1.2, 2.3, 4.1, 6.0, 6.1, 8.0, 9.0], [3.5, 4.8, 7.5, 4.0, 9.0, 5.5, 9.8]))
    )

    result_adata = aggregate(subcellular, cellular, "structure", agg_func="sum")
    assert result_adata.obs_names.to_list() == ["shape_0", "shape_1"]
    assert result_adata.var_names.to_list() == ["nucleus", "mitochondria"]
    np.testing.assert_equal(result_adata.X.A, np.array([[2, 0], [1, 3]]))


@pytest.mark.parametrize("image_schema", [Image2DModel])
@pytest.mark.parametrize("labels_schema", [Labels2DModel])
def test_aggregate_image_by_labels(labels_blobs, image_schema, labels_schema) -> None:
    image = RNG.normal(size=(3,) + labels_blobs.shape)

    image = image_schema.parse(image)
    labels = labels_schema.parse(labels_blobs)

    out = aggregate(image, labels)
    assert len(out) + 1 == len(np.unique(labels_blobs))
    assert isinstance(out, AnnData)
    np.testing.assert_array_equal(out.var_names, [f"channel_{i}_mean" for i in image.coords["c"].values])

    out = aggregate(image, labels, agg_func=["mean", "sum", "count"])
    assert len(out) + 1 == len(np.unique(labels_blobs))

    out = aggregate(image, labels, zone_ids=[1, 2, 3])
    assert len(out) == 3


def test_aggregate_spatialdata(sdata_blobs: SpatialData) -> None:
    sdata = sdata_blobs.aggregate(sdata_blobs.points["blobs_points"], by="blobs_polygons")
    assert isinstance(sdata, SpatialData)
    assert len(sdata.shapes["blobs_polygons"]) == 2
    assert sdata.table.shape == (2, 2)
    assert len(sdata.points["points"].compute()) == 300
