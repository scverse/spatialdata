import logging
import re

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from geopandas import GeoDataFrame
from numpy.random import default_rng
from pandas import DataFrame
from scipy.sparse import csr_matrix
from shapely.geometry import Polygon

from spatialdata._core.data_extent import are_extents_equal, get_extent
from spatialdata._core.operations.rasterize_bins import (
    _relabel_labels,
    rasterize_bins,
    rasterize_bins_link_table_to_labels,
)
from spatialdata._core.spatialdata import SpatialData
from spatialdata._logging import logger
from spatialdata._types import ArrayLike
from spatialdata.models.models import (
    Image2DModel,
    Labels2DModel,
    PointsModel,
    ShapesModel,
    TableModel,
)
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
@pytest.mark.parametrize("return_region_as_labels", [True, False])
def test_rasterize_bins(geometry: str, value_key: str | list[str] | None, return_region_as_labels: bool):
    n = 10
    data, x, y = _get_bins_data(n)
    scale = Scale([2.0], axes=("x",))
    index = np.arange(1, len(data) + 1)

    if geometry == "points":
        points = PointsModel.parse(
            data,
            transformations={"global": scale},
            annotation=pd.DataFrame(index=index),
        )
    elif geometry == "circles":
        points = ShapesModel.parse(data, geometry=0, radius=1, transformations={"global": scale}, index=index)
    else:
        assert geometry == "squares"

        gdf = GeoDataFrame(
            index=index,
            data={"geometry": [Polygon([(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1), (x, y)]) for x, y in data]},
        )
        points = ShapesModel.parse(gdf, transformations={"global": scale})

    obs = DataFrame(
        data={
            "region": pd.Categorical(["points"] * n * n),
            "instance_id": index,
            "col_index": x,
            "row_index": y,
        },
        index=[f"{i}" for i in range(n * n)],
    )
    X = RNG.normal(size=(n * n, 2))
    var = DataFrame(index=["gene0", "gene1"])
    table = TableModel.parse(
        AnnData(X=X, var=var, obs=obs),
        region="points",
        region_key="region",
        instance_key="instance_id",
    )
    sdata = SpatialData.init_from_elements({"points": points, "table": table})
    rasterized = rasterize_bins(
        sdata=sdata,
        bins="points",
        table_name="table",
        col_key="col_index",
        row_key="row_index",
        value_key=value_key,
        return_region_as_labels=return_region_as_labels,
    )
    points_extent = get_extent(points)
    raster_extent = get_extent(rasterized)
    # atol can be set tighter when https://github.com/scverse/spatialdata/issues/165 is addressed
    assert are_extents_equal(points_extent, raster_extent, atol=2)

    # if regions are returned as labels, we can annotate the table with 'rasterized',
    # which is a labels layer containing the bins, and then run rasterize_bins again
    # but now with return_region_as_labels set to False to get a lazy image.
    if return_region_as_labels:
        labels_name = "labels"
        sdata[labels_name] = rasterized

        rasterize_bins_link_table_to_labels(sdata=sdata, table_name="table", rasterized_labels_name=labels_name)

        # this fails because table already annotated by labels layer
        with pytest.raises(
            ValueError,
            match="bins is already a labels layer that annotates the table 'table'. "
            "Consider setting 'return_region_as_labels' to 'False' to create a lazy spatial image.",
        ):
            _ = rasterize_bins(
                sdata,
                bins="labels",
                table_name="table",
                col_key="col_index",
                row_key="row_index",
                value_key=value_key,
                return_region_as_labels=True,
            )

        # but we want to be able to create the lazy raster even if the table is already annotated by a labels layer
        rasterized = rasterize_bins(
            sdata,
            bins="labels",
            table_name="table",
            col_key="col_index",
            row_key="row_index",
            value_key=value_key,
            return_region_as_labels=False,
        )
        raster_extent = get_extent(rasterized)
        assert are_extents_equal(points_extent, raster_extent, atol=2)


def test_rasterize_bins_invalid():
    def _get_sdata(n: int):
        data, x, y = _get_bins_data(n)
        points = PointsModel.parse(data)
        obs = DataFrame(
            data={
                "region": pd.Categorical(["points"] * n * n),
                "instance_id": np.arange(n * n),
                "col_index": x,
                "row_index": y,
            },
            index=[f"{i}" for i in range(n * n)],
        )
        table = TableModel.parse(
            AnnData(X=RNG.normal(size=(n * n, 2)), obs=obs),
            region="points",
            region_key="region",
            instance_key="instance_id",
        )
        return SpatialData.init_from_elements({"points": points, "table": table})

    # sdata with not enough bins (2*2) to estimate transformation
    sdata = _get_sdata(n=2)
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
    sdata = _get_sdata(n=3)
    table = sdata.tables["table"]
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
    regions.iloc[0] = "shapes"
    sdata["shapes"] = sdata["points"]
    table.obs["region"] = regions
    with pytest.raises(
        ValueError,
        match="Found multiple regions annotated by the table: points, shapes.",
    ):
        _ = rasterize_bins(
            sdata=sdata,
            bins="points",
            table_name="table",
            col_key="col_index",
            row_key="row_index",
            value_key="instance_id",
        )
    # table annotating wrong element
    sdata = _get_sdata(n=3)
    table = sdata.tables["table"]
    table.obs["region"] = "shapes"
    table.obs["region"] = table.obs["region"].astype("category")
    with pytest.raises(
        ValueError,
        match="The table should be associated with the specified bins.",
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
    sdata = _get_sdata(n=3)
    table = sdata.tables["table"]
    table.obs["region"] = table.obs["region"].astype(str)
    with pytest.raises(
        ValueError,
        match="Please convert `table.obs.*` to a category series to improve performances",
    ):
        _ = rasterize_bins(
            sdata=sdata,
            bins="points",
            table_name="table",
            col_key="col_index",
            row_key="row_index",
            value_key="instance_id",
        )

    # the element to rasterize should be a GeoDataFrame, a DaskDataFrame or a DataArray holding labels
    sdata = _get_sdata(n=3)
    with pytest.raises(
        ValueError,
        match="The bins should be a GeoDataFrame, a DaskDataFrame or a DataArray.",
    ):
        _ = rasterize_bins(
            sdata=sdata,
            bins="table",
            table_name="table",
            col_key="col_index",
            row_key="row_index",
            value_key="instance_id",
        )

    # if bins is a DataArray it should contain labels
    image = Image2DModel.parse(RNG.integers(low=0, high=10, size=(1, 3, 3)), dims=("c", "y", "x"))
    del sdata["points"]
    sdata["points"] = image
    with pytest.raises(
        ValueError,
        match=re.escape("If bins is a DataArray, it should hold labels; found a image element instead, with"),
    ):
        _ = rasterize_bins(
            sdata=sdata,
            bins="points",
            table_name="table",
            col_key="col_index",
            row_key="row_index",
            value_key="instance_id",
        )

    # if bins is a DataArray, it should hold integers
    image = Labels2DModel.parse(RNG.normal(size=(3, 3)), dims=("y", "x"))
    del sdata["points"]
    sdata["points"] = image
    with pytest.raises(
        ValueError,
        match=f"If bins is a DataArray, it should hold integers. Found dtype {image.dtype}.",
    ):
        _ = rasterize_bins(
            sdata=sdata,
            bins="points",
            table_name="table",
            col_key="col_index",
            row_key="row_index",
            value_key="instance_id",
        )


def test_relabel_labels(caplog):
    obs = DataFrame(
        data={
            "instance_key0": np.arange(1, 11),
            "instance_key1": np.arange(10),
            "instance_key2": [1, 2] + list(range(4, 12)),
            "instance_key3": [str(i) for i in range(1, 11)],
        },
        index=[f"{i}" for i in range(10)],
    )
    adata = AnnData(X=RNG.normal(size=(10, 2)), obs=obs)
    _relabel_labels(table=adata, instance_key="instance_key0")
    # check logger info message
    expected_log_message = (
        "will be relabeled to ensure a numeric data type, with a continuous range and without including the value 0 ("
        "which is reserved for the background). The new labels will be stored in a new column named"
    )
    logger.propagate = True
    with caplog.at_level(logging.INFO):
        _relabel_labels(table=adata, instance_key="instance_key1")
        assert expected_log_message in caplog.text

    with caplog.at_level(logging.INFO):
        _relabel_labels(table=adata, instance_key="instance_key2")
        assert expected_log_message in caplog.text

    with caplog.at_level(logging.INFO):
        _relabel_labels(table=adata, instance_key="instance_key3")
        assert expected_log_message in caplog.text
    logger.propagate = False


if __name__ == "__main__":
    test_relabel_labels()
