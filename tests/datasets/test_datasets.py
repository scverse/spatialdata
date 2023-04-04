from spatialdata.datasets import blobs, raccoon


def test_datasets() -> None:
    sdata_blobs = blobs()

    assert len(sdata_blobs.table) == 26
    assert len(sdata_blobs.shapes["blobs_shapes"]) == 5
    assert len(sdata_blobs.shapes["blobs_shapes"]) == 5
    assert len(sdata_blobs.points["blobs_points"].compute()) == 200

    sdata_raccoon = raccoon()
    assert sdata_raccoon.table is None
    assert len(sdata_raccoon.shapes["circles"]) == 4
