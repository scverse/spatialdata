from __future__ import annotations

import warnings

import annsel as an
import numpy as np
import pytest
from xarray import DataArray

from spatialdata import concatenate, match_sdata_to_table
from spatialdata._core.query.relational_query import (
    _set_instance_ids_in_labels_to_zero,
    filter_by_table_query,
)
from spatialdata.datasets import blobs_annotating_element
from spatialdata.models import Labels2DModel


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
    ids_to_remove = list(np.unique(table.obs["instance_id"][~third_elems]))
    subset_table = table[third_elems]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="labels not supported for right join", category=UserWarning)
        if subset_func_name == "match_sdata_to_table":
            subset_sdata = match_sdata_to_table(sdata, "table", table=subset_table)
        else:
            subset_sdata = filter_by_table_query(sdata, "table", obs_expr=an.col("instance_id") == 3)

    for label_name in list(subset_sdata.labels.keys()):
        elem = subset_sdata[label_name]
        del subset_sdata[label_name]
        if isinstance(elem, DataArray):
            filtered = Labels2DModel.parse(_set_instance_ids_in_labels_to_zero(elem, ids_to_remove))
        else:  # DataTree (multiscale)
            scales = list(elem.keys())
            scale_factors = [
                round(elem[scales[i]].image.shape[0] / elem[scales[i + 1]].image.shape[0])
                for i in range(len(scales) - 1)
            ]
            filtered = Labels2DModel.parse(
                _set_instance_ids_in_labels_to_zero(elem[scales[0]].image, ids_to_remove),
                scale_factors=scale_factors,
            )
        subset_sdata[label_name] = filtered

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
    ids_to_remove = [i for i in table.obs["instance_id"].unique() if i != keep_id]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="labels not supported for right join", category=UserWarning)
        if subset_func_name == "match_sdata_to_table":
            subset_table = table[table.obs["instance_id"] == keep_id]
            subset_sdata = match_sdata_to_table(sdata, "table", table=subset_table)
        else:
            subset_sdata = filter_by_table_query(sdata, "table", obs_expr=an.col("instance_id") == keep_id)

    elem = subset_sdata[element_name]
    if isinstance(elem, DataArray):
        filtered = Labels2DModel.parse(_set_instance_ids_in_labels_to_zero(elem, ids_to_remove))
        remaining_ids = set(np.unique(filtered.data.compute())) - {0}
        assert remaining_ids == {keep_id}
    else:  # DataTree (multiscale)
        scales = list(elem.keys())
        scale_factors = [
            round(elem[scales[i]].image.shape[0] / elem[scales[i + 1]].image.shape[0]) for i in range(len(scales) - 1)
        ]
        filtered = Labels2DModel.parse(
            _set_instance_ids_in_labels_to_zero(elem[scales[0]].image, ids_to_remove),
            scale_factors=scale_factors,
        )
        for scale in filtered:
            # at coarser scales an instance may vanish due to downsampling, but no other instance should appear
            remaining_ids = set(np.unique(filtered[scale].image.compute())) - {0}
            assert remaining_ids <= {keep_id}
