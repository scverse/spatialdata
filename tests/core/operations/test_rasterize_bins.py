from __future__ import annotations

import numpy as np
import pytest
from anndata import AnnData
from geopandas import GeoDataFrame
from numpy.random import default_rng
from pandas import DataFrame
from scipy.sparse import csr_matrix
from shapely.geometry import Polygon

from spatialdata._core.data_extent import are_extents_equal, get_extent
from spatialdata._core.operations.rasterize_bins import rasterize_bins
from spatialdata._core.spatialdata import SpatialData
from spatialdata._types import ArrayLike
from spatialdata.models.models import Labels2DModel, PointsModel, ShapesModel, TableModel
from spatialdata.transformations.transformations import Scale

RNG = default_rng(0)


def _get_bins_data(n: int) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    x = np.repeat(np.arange(n), n)
    y = np.tile(np.arange(n), n)
    data = np.stack([x, y], axis=1)
    theta = np.pi / 4
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return np.dot(data, rotation.T), x, y


@pytest.mark.parametrize("geometry", ["points", "circles", "squares"])
@pytest.mark.parametrize("value_key", [None, "instance_id", ["gene0", "gene1"]])
def test_rasterize_bins(geometry: str, value_key: str | list[str] | None):
    n = 10
    data, x, y = _get_bins_data(n)
    scale = Scale([2.0], axes=("x",))

    if geometry == "points":
        points = PointsModel.parse(data, transformations={"global": scale})
    elif geometry == "circles":
        points = ShapesModel.parse(data, geometry=0, radius=1, transformations={"global": scale})
    else:
        assert geometry == "squares"

        gdf = GeoDataFrame(
            data={"geometry": [Polygon([(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1), (x, y)]) for x, y in data]}
        )

        points = ShapesModel.parse(gdf, transformations={"global": scale})

    obs = DataFrame(
        data={"region": ["points"] * n * n, "instance_id": np.arange(n * n), "col_index": x, "row_index": y}
    )
    X = RNG.normal(size=(n * n, 2))
    var = DataFrame(index=["gene0", "gene1"])
    table = TableModel.parse(
        AnnData(X=X, var=var, obs=obs), region="points", region_key="region", instance_key="instance_id"
    )
    sdata = SpatialData.init_from_elements({"points": points}, tables={"table": table})
    rasterized = rasterize_bins(
        sdata=sdata,
        bins="points",
        table_name="table",
        col_key="col_index",
        row_key="row_index",
        value_key=value_key,
    )
    points_extent = get_extent(points)
    raster_extent = get_extent(rasterized)
    # atol can be set tighter when https://github.com/scverse/spatialdata/issues/165 is addressed
    assert are_extents_equal(points_extent, raster_extent, atol=2)


def test_rasterize_bins_invalid():
    n = 2
    data, x, y = _get_bins_data(n)
    points = PointsModel.parse(data)
    obs = DataFrame(
        data={"region": ["points"] * n * n, "instance_id": np.arange(n * n), "col_index": x, "row_index": y}
    )
    table = TableModel.parse(
        AnnData(X=RNG.normal(size=(n * n, 2)), obs=obs),
        region="points",
        region_key="region",
        instance_key="instance_id",
    )
    sdata = SpatialData.init_from_elements({"points": points}, tables={"table": table})

    # not enough points
    with pytest.raises(ValueError, match="At least 6 bins are needed to estimate the transformation."):
        _ = rasterize_bins(
            sdata=sdata,
            bins="points",
            table_name="table",
            col_key="col_index",
            row_key="row_index",
            value_key="instance_id",
        )

    # the matrix should be a csc_matrix or a full matrix; in particular not a csr_matrix
    table.X = csr_matrix(table.X)
    with pytest.raises(
        ValueError,
        match="To speed up bins rasterization, the X matrix in the table, when sparse, should be a csc_matrix matrix.",
    ):
        _ = rasterize_bins(
            sdata=sdata,
            bins="points",
            table_name="table",
            col_key="col_index",
            row_key="row_index",
            # note that value_key is None here since the csc matrix is needed when rasterizing from table.X
        )

    # table annotating multiple elements
    regions = table.obs["region"].copy()
    regions = regions.cat.add_categories(["shapes"])
    regions[0] = "shapes"
    sdata["shapes"] = sdata["points"]
    table.obs["region"] = regions
    with pytest.raises(
        ValueError,
        match="The table should be associated with the specified bins. Found multiple regions annotated by the table: "
        "points, shapes.",
    ):
        _ = rasterize_bins(
            sdata=sdata,
            bins="points",
            table_name="table",
            col_key="col_index",
            row_key="row_index",
            value_key="instance_id",
        )

    # region_key should be categorical
    table.obs["region"] = table.obs["region"].astype(str)
    with pytest.raises(ValueError, match="Please convert `table.obs.*` to a category series to improve performances"):
        _ = rasterize_bins(
            sdata=sdata,
            bins="points",
            table_name="table",
            col_key="col_index",
            row_key="row_index",
            value_key="instance_id",
        )

    # the element to rasterize should be a GeoDataFrame or a DaskDataFrame
    image = Labels2DModel.parse(RNG.normal(size=(n, n)))
    del sdata["points"]
    sdata["points"] = image
    with pytest.raises(ValueError, match="The bins should be a GeoDataFrame or a DaskDataFrame."):
        _ = rasterize_bins(
            sdata=sdata,
            bins="points",
            table_name="table",
            col_key="col_index",
            row_key="row_index",
            value_key="instance_id",
        )
