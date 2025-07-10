import numpy as np
import pytest

from spatialdata import concatenate, subset_sdata_by_table_mask
from spatialdata._core.query.relational_query import _filter_by_instance_ids
from spatialdata.datasets import blobs_annotating_element


def test_filter_labels2dmodel_by_instance_ids():
    sdata = blobs_annotating_element("blobs_labels")
    labels_element = sdata["blobs_labels"]
    all_instance_ids = sdata.tables["table"].obs["instance_id"].unique()
    filtered_labels_element = _filter_by_instance_ids(labels_element, [2, 3], "instance_id")

    # because 0 is the background, we expect the filtered ids to be the instance ids that are not 0
    filtered_ids = set(np.unique(filtered_labels_element.data.compute())) - {
        0,
    }
    preserved_ids = np.unique(labels_element.data.compute())
    assert filtered_ids == (set(all_instance_ids) - {2, 3})
    # check if there is modification of the original labels
    assert set(preserved_ids) == set(all_instance_ids) | {0}

    sdata.tables["table"].uns["spatialdata_attrs"]["region"] = "blobs_multiscale_labels"
    sdata.tables["table"].obs.region = "blobs_multiscale_labels"
    labels_element = sdata["blobs_multiscale_labels"]
    filtered_labels_element = _filter_by_instance_ids(labels_element, [2, 3], "instance_id")

    for scale in labels_element:
        filtered_ids = set(np.unique(filtered_labels_element[scale].image.compute())) - {
            0,
        }
        preserved_ids = np.unique(labels_element[scale].image.compute())
        assert filtered_ids == (set(all_instance_ids) - {2, 3})
        # check if there is modification of the original labels
        assert set(preserved_ids) == set(all_instance_ids) | {0}


def test_subset_sdata_by_table_mask():
    sdata = concatenate(
        {
            "labels": blobs_annotating_element("blobs_labels"),
            "shapes": blobs_annotating_element("blobs_circles"),
            "points": blobs_annotating_element("blobs_points"),
            "multiscale_labels": blobs_annotating_element("blobs_multiscale_labels"),
        },
        concatenate_tables=True,
    )
    third_elems = sdata.tables["table"].obs["instance_id"] == 3
    subset_sdata = subset_sdata_by_table_mask(sdata, "table", third_elems)

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

    points_remaining_ids = set(np.unique(subset_sdata.points["blobs_points-points"]["instance_id"].compute())) - {0}
    assert points_remaining_ids == {3}

    shapes_remaining_ids = set(np.unique(subset_sdata.shapes["blobs_circles-shapes"].index)) - {0}
    assert shapes_remaining_ids == {3}


def test_subset_sdata_by_table_mask_with_no_annotated_elements():
    with pytest.raises(ValueError, match="Table table_not_found not found in SpatialData object."):
        sdata = blobs_annotating_element("blobs_labels")
        _ = subset_sdata_by_table_mask(sdata, "table_not_found", sdata.tables["table"].obs["instance_id"] == 3)


def test_filter_by_instance_ids_fails_for_unsupported_element_models():
    with pytest.raises(NotImplementedError, match="Filtering by instance ids is not implemented for"):
        _filter_by_instance_ids([1, 1, 1, 2], [1], "instance_id")
