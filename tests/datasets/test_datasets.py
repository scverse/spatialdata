from spatialdata.datasets import blobs, raccoon


def test_datasets() -> None:
    extra_cs = "test"
    sdata_blobs = blobs(extra_coord_system=extra_cs)

    assert len(sdata_blobs["table"]) == 26
    assert len(sdata_blobs.shapes["blobs_circles"]) == 5
    assert len(sdata_blobs.shapes["blobs_polygons"]) == 5
    assert len(sdata_blobs.shapes["blobs_multipolygons"]) == 2
    assert len(sdata_blobs.points["blobs_points"].compute()) == 200
    assert sdata_blobs.images["blobs_image"].shape == (3, 512, 512)
    assert len(sdata_blobs.images["blobs_multiscale_image"]) == 3
    assert sdata_blobs.labels["blobs_labels"].shape == (512, 512)
    assert len(sdata_blobs.labels["blobs_multiscale_labels"]) == 3
    assert extra_cs in sdata_blobs.coordinate_systems
    # this catches this bug: https://github.com/scverse/spatialdata/issues/269
    _ = str(sdata_blobs)

    sdata_raccoon = raccoon()
    assert "table" not in sdata_raccoon.tables
    assert len(sdata_raccoon.shapes["circles"]) == 4
    assert sdata_raccoon.images["raccoon"].shape == (3, 768, 1024)
    assert sdata_raccoon.labels["segmentation"].shape == (768, 1024)
    _ = str(sdata_raccoon)
