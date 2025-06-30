import numpy as np

from spatialdata._core.query.masking import filter_labels2dmodel_by_instance_ids, filter_shapesmodel_by_instance_ids
from spatialdata.datasets import blobs_annotating_element


def test_filter_labels2dmodel_by_instance_ids():
    sdata = blobs_annotating_element("blobs_labels")
    labels_element = sdata["blobs_labels"]
    all_instance_ids = sdata.tables["table"].obs["instance_id"].unique()
    filtered_labels_element = filter_labels2dmodel_by_instance_ids(labels_element, [2, 3])

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
    filtered_labels_element = filter_labels2dmodel_by_instance_ids(labels_element, [2, 3])

    for scale in labels_element:
        filtered_ids = set(np.unique(filtered_labels_element[scale].image.compute())) - {
            0,
        }
        preserved_ids = np.unique(labels_element[scale].image.compute())
        assert filtered_ids == (set(all_instance_ids) - {2, 3})
        # check if there is modification of the original labels
        assert set(preserved_ids) == set(all_instance_ids) | {0}


def test_filter_shapesmodel_by_instance_ids():
    sdata = blobs_annotating_element("blobs_circles")
    shapes_element = sdata["blobs_circles"]
    filtered_shapes_element = filter_shapesmodel_by_instance_ids(shapes_element, [2, 3])

    assert set(filtered_shapes_element.index.tolist()) == {0, 1, 4}
