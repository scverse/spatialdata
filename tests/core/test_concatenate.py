import pytest

import spatialdata as sd
from spatialdata.datasets import blobs


@pytest.mark.parametrize("merge_coordinate_systems_on_name", [True, False])
def test_concatenate_merge_coordinate_systems_on_name(merge_coordinate_systems_on_name):
    blob1 = blobs()
    blob2 = blobs()

    sdata_keys = ["blob1", "blob2"]
    sdata = sd.concatenate(
        dict(zip(sdata_keys, [blob1, blob2], strict=True)),
        merge_coordinate_systems_on_name=merge_coordinate_systems_on_name,
    )

    expected_images = ["blobs_image", "blobs_multiscale_image"]
    expected_labels = ["blobs_labels", "blobs_multiscale_labels"]
    expected_points = ["blobs_points"]
    expected_shapes = ["blobs_circles", "blobs_polygons", "blobs_multipolygons"]

    expected_suffixed_images = [f"{name}-{key}" for key in sdata_keys for name in expected_images]
    expected_suffixed_labels = [f"{name}-{key}" for key in sdata_keys for name in expected_labels]
    expected_suffixed_points = [f"{name}-{key}" for key in sdata_keys for name in expected_points]
    expected_suffixed_shapes = [f"{name}-{key}" for key in sdata_keys for name in expected_shapes]

    assert set(sdata.images.keys()) == set(expected_suffixed_images)
    assert set(sdata.labels.keys()) == set(expected_suffixed_labels)
    assert set(sdata.points.keys()) == set(expected_suffixed_points)
    assert set(sdata.shapes.keys()) == set(expected_suffixed_shapes)

    if merge_coordinate_systems_on_name:
        assert set(sdata.coordinate_systems) == {"global"}
    else:
        assert set(sdata.coordinate_systems) == {"global-blob1", "global-blob2"}
