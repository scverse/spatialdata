from __future__ import annotations

import annsel as an
import numpy as np
import pytest
from xarray import DataArray

from spatialdata import concatenate, match_sdata_to_table
from spatialdata._core.query.relational_query import filter_by_table_query
from spatialdata.datasets import blobs_annotating_element


@pytest.mark.parametrize("subset_func_name", ["match_sdata_to_table", "filter_by_table_query"])
def test_subset_sdata_by_table_mask(subset_func_name: str) -> None:
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
    third_elems = table.obs["instance_id"] == 3
    subset_table = table[third_elems]

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
@pytest.mark.parametrize("element_name", ["blobs_labels", "blobs_multiscale_labels"])
def test_filter_out_instances(subset_func_name: str, element_name: str) -> None:
    sdata = blobs_annotating_element(element_name)
    table = sdata.tables["table"]
    keep_id = 3
    subset_table = table[table.obs["instance_id"] == keep_id]

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
