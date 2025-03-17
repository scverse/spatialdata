import pytest

from spatialdata import SpatialData, concatenate, match_sdata_to_table
from spatialdata.datasets import blobs_annotating_element


def _make_test_data() -> SpatialData:
    sdata1 = blobs_annotating_element("blobs_polygons")
    sdata2 = blobs_annotating_element("blobs_polygons")
    sdata = concatenate({"sdata1": sdata1, "sdata2": sdata2}, concatenate_tables=True)
    sdata["table"].obs["value"] = list(range(sdata["table"].obs.shape[0]))
    return sdata


# constructing the example data; let's use a global variable as we can reuse the same object on most tests
# without having to recreate it
sdata = _make_test_data()


def test_match_sdata_to_table_filter_specific_instances():
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


def test_match_sdata_to_table_filter_specific_instances_element():
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


def test_match_sdata_to_table_filter_by_threshold():
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


def test_match_sdata_to_table_subset_certain_obs():
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


def test_match_sdata_to_table_match_labels_error():
    """
    match_sdata_to_table() uses the join operations; so when trying to match labels, the error will be raised by the
    join.
    """
    sdata = _make_test_data()
    sdata["table"].obs["region"] = sdata["table"].obs["region"].apply(lambda x: x.replace("polygons", "labels"))
    sdata["table"].obs["region"] = sdata["table"].obs["region"].astype("category")
    sdata.set_table_annotates_spatialelement(
        table_name="table",
        region=["blobs_labels-sdata1", "blobs_labels-sdata2"],
        region_key="region",
        instance_key="instance_id",
    )

    with pytest.warns(
        UserWarning,
        match="Element type `labels` not supported for 'right' join. Skipping ",
    ):
        matched = match_sdata_to_table(
            sdata,
            table=sdata["table"],
            table_name="table",
        )

    assert len(matched["table"]) == 10
    assert "blobs_labels-sdata1" in matched
    assert "blobs_labels-sdata2" in matched
    assert "blobs_points-sdata1" not in matched


def test_match_sdata_to_table_no_table_argument():
    """
    If no table argument is passed, the table_name argument will be used to match the table.
    """
    matched = match_sdata_to_table(sdata=sdata, table_name="table")

    assert len(matched["table"]) == 10
    assert "blobs_polygons-sdata1" in matched
    assert "blobs_polygons-sdata2" in matched
