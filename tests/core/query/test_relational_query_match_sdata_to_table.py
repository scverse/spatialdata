from __future__ import annotations

import contextlib

import annsel as an
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from xarray import DataArray

from spatialdata import SpatialData, concatenate, match_sdata_to_table
from spatialdata._core.query.relational_query import filter_by_table_query
from spatialdata.datasets import blobs_annotating_element
from spatialdata.models import Labels3DModel, TableModel


def _make_test_data() -> SpatialData:
    sdata1 = blobs_annotating_element("blobs_polygons")
    sdata2 = blobs_annotating_element("blobs_polygons")
    sdata = concatenate({"sdata1": sdata1, "sdata2": sdata2}, concatenate_tables=True)
    sdata["table"].obs["value"] = list(range(sdata["table"].obs.shape[0]))
    return sdata


# constructing the example data; reuse the same object on most tests without having to recreate it
@pytest.fixture(scope="module")
def sdata():
    return _make_test_data()


def test_match_sdata_to_table_filter_specific_instances(sdata):
    """
    Filter to keep only specific instances. Note that it works even when the table annotates multiple elements.
    """
    matched = match_sdata_to_table(
        sdata,
        table=sdata["table"][sdata["table"].obs.instance_id.isin([1, 2])],
        table_name="table",
    )
    assert len(matched["table"]) == 4
    assert "blobs_polygons-sdata1" in matched
    assert "blobs_polygons-sdata2" in matched


def test_match_sdata_to_table_filter_specific_instances_element(sdata):
    """
    Filter to keep only specific instances, in a specific element.
    """
    matched = match_sdata_to_table(
        sdata,
        table=sdata["table"][
            sdata["table"].obs.instance_id.isin([1, 2]) & (sdata["table"].obs.region == "blobs_polygons-sdata1")
        ],
        table_name="table",
    )
    assert len(matched["table"]) == 2
    assert "blobs_polygons-sdata1" in matched
    assert "blobs_polygons-sdata2" not in matched


def test_match_sdata_to_table_filter_by_threshold(sdata):
    """
    Filter by a threshold on a value column, in a specific element.
    """
    matched = match_sdata_to_table(
        sdata,
        table=sdata["table"][sdata["table"].obs.query('value < 5 and region == "blobs_polygons-sdata1"').index],
        table_name="table",
    )
    assert len(matched["table"]) == 5
    assert "blobs_polygons-sdata1" in matched
    assert "blobs_polygons-sdata2" not in matched


def test_match_sdata_to_table_subset_certain_obs(sdata):
    """
    Subset to certain obs (we could also subset to certain var or layer).
    """
    matched = match_sdata_to_table(
        sdata,
        table=sdata["table"][[0, 1, 2, 3]],
        table_name="table",
    )
    assert len(matched["table"]) == 4
    assert "blobs_polygons-sdata1" in matched
    assert "blobs_polygons-sdata2" not in matched


def test_match_sdata_to_table_shapes_and_points():
    """
    The function works both for shapes (examples above) and points.
    Changes the target of the table to labels.
    """
    sdata = _make_test_data()
    sdata["table"].obs["region"] = sdata["table"].obs["region"].apply(lambda x: x.replace("polygons", "points"))
    sdata["table"].obs["region"] = sdata["table"].obs["region"].astype("category")
    sdata.set_table_annotates_spatialelement(
        table_name="table",
        region=["blobs_points-sdata1", "blobs_points-sdata2"],
        region_key="region",
        instance_key="instance_id",
    )

    matched = match_sdata_to_table(
        sdata,
        table=sdata["table"],
        table_name="table",
    )

    assert len(matched["table"]) == 10
    assert "blobs_points-sdata1" in matched
    assert "blobs_points-sdata2" in matched
    assert "blobs_polygons-sdata1" not in matched


@pytest.mark.parametrize("subset_func_name", ["match_sdata_to_table", "filter_by_table_query"])
@pytest.mark.parametrize("element_name", ["blobs_labels", "blobs_multiscale_labels"])
def test_filter_out_instances(subset_func_name: str, element_name: str) -> None:
    """
    By default a warning is issued when labels are encountered in a right join and pixels are not filtered.
    Passing filter_label_pixels=True filters the label pixels to match the table.
    """
    sdata = blobs_annotating_element(element_name)
    keep_id = 3
    table = sdata.tables["table"]
    subset_table = table[table.obs["instance_id"] == keep_id]

    # None → warning issued; False → silenced, pixels still unfiltered
    for flp, ctx in [
        (None, pytest.warns(UserWarning, match="pixels are not filtered")),
        (False, contextlib.nullcontext()),
    ]:
        with ctx:
            if subset_func_name == "match_sdata_to_table":
                match_sdata_to_table(sdata, "table", table=subset_table, filter_label_pixels=flp)
            else:
                filter_by_table_query(
                    sdata, "table", obs_expr=an.col("instance_id") == keep_id, filter_label_pixels=flp
                )

    # filter_label_pixels=True: pixels are zeroed for removed instances
    if subset_func_name == "match_sdata_to_table":
        subset_sdata = match_sdata_to_table(sdata, "table", table=subset_table, filter_label_pixels=True)
    else:
        subset_sdata = filter_by_table_query(
            sdata, "table", obs_expr=an.col("instance_id") == keep_id, filter_label_pixels=True
        )

    elem = subset_sdata[element_name]
    if isinstance(elem, DataArray):
        remaining_ids = set(np.unique(elem.data.compute())) - {0}
        assert remaining_ids == {keep_id}
    else:  # DataTree (multiscale)
        for scale in elem:
            # at coarser scales an instance may vanish due to downsampling, but no other instance should appear
            remaining_ids = set(np.unique(elem[scale].image.compute())) - {0}
            assert remaining_ids <= {keep_id}


def test_match_sdata_to_table_no_table_argument(sdata):
    """
    If no table argument is passed, the table_name argument will be used to match the table.
    """
    matched = match_sdata_to_table(sdata=sdata, table_name="table")

    assert len(matched["table"]) == 10
    assert "blobs_polygons-sdata1" in matched
    assert "blobs_polygons-sdata2" in matched


@pytest.mark.parametrize("subset_func_name", ["match_sdata_to_table", "filter_by_table_query"])
def test_subset_sdata_by_table_mask(subset_func_name: str) -> None:
    """
    Subsetting an sdata with mixed element types (labels, shapes, points) keeps only the requested instances.
    Labels are pixel-filtered when filter_label_pixels=True.
    """
    sdata = concatenate(
        {
            "labels": blobs_annotating_element("blobs_labels"),
            "shapes": blobs_annotating_element("blobs_circles"),
            "points": blobs_annotating_element("blobs_points"),
            "multiscale_labels": blobs_annotating_element("blobs_multiscale_labels"),
        },
        concatenate_tables=True,
    )
    table = sdata.tables["table"]
    subset_table = table[table.obs["instance_id"] == 3]

    if subset_func_name == "match_sdata_to_table":
        subset_sdata = match_sdata_to_table(sdata, "table", table=subset_table, filter_label_pixels=True)
    else:
        subset_sdata = filter_by_table_query(
            sdata, "table", obs_expr=an.col("instance_id") == 3, filter_label_pixels=True
        )

    assert set(subset_sdata.labels.keys()) == {"blobs_labels-labels", "blobs_multiscale_labels-multiscale_labels"}
    assert set(subset_sdata.points.keys()) == {"blobs_points-points"}
    assert set(subset_sdata.shapes.keys()) == {"blobs_circles-shapes"}

    labels_remaining_ids = set(np.unique(subset_sdata.labels["blobs_labels-labels"].data.compute())) - {0}
    assert labels_remaining_ids == {3}

    for scale in subset_sdata.labels["blobs_multiscale_labels-multiscale_labels"]:
        ms_labels_remaining_ids = set(
            np.unique(subset_sdata.labels["blobs_multiscale_labels-multiscale_labels"][scale].image.compute())
        ) - {0}
        assert ms_labels_remaining_ids == {3}

    points_remaining_ids = set(np.unique(subset_sdata.points["blobs_points-points"].index)) - {0}
    assert points_remaining_ids == {3}

    shapes_remaining_ids = set(np.unique(subset_sdata.shapes["blobs_circles-shapes"].index)) - {0}
    assert shapes_remaining_ids == {3}


@pytest.mark.parametrize("subset_func_name", ["match_sdata_to_table", "filter_by_table_query"])
def test_filter_out_instances_3d_labels_not_supported(subset_func_name: str) -> None:
    """Pixel-level filtering of 3D labels raises NotImplementedError."""
    data = np.zeros((5, 5, 5), dtype=np.int32)
    data[1:3, 1:3, 1:3] = 1
    data[3:5, 3:5, 3:5] = 2
    labels_3d = Labels3DModel.parse(data, dims=["z", "y", "x"])

    obs_df = pd.DataFrame({"region": pd.Categorical(["labels_3d"]), "instance_id": [1]}, index=["0"])
    table = TableModel.parse(
        AnnData(shape=(1, 0), obs=obs_df), region="labels_3d", region_key="region", instance_key="instance_id"
    )
    sdata = SpatialData(labels={"labels_3d": labels_3d}, tables={"table": table})

    with pytest.raises(NotImplementedError, match="3D labels"):
        if subset_func_name == "match_sdata_to_table":
            match_sdata_to_table(sdata, "table", filter_label_pixels=True)
        else:
            filter_by_table_query(sdata, "table", obs_expr=an.col("instance_id") == 1, filter_label_pixels=True)
