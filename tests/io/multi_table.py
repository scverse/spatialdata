from spatialdata.models import TableModel, ShapesModel
from anndata import AnnData
import numpy as np
from spatialdata import SpatialData
from pathlib import Path

# notes on paths: https://github.com/orgs/scverse/projects/17/views/1?pane=issue&itemId=44066734
# notes for the people (to prettify) https://hackmd.io/wd7K4Eg1SlykKVN-nOP44w

# shapes
visium_locations0 = ShapesModel.parse(np.random.rand(10, 2), geometry=0, radius=1)
visium_locations1 = ShapesModel.parse(np.random.rand(10, 2), geometry=0, radius=1)
visium_locations2 = ShapesModel.parse(np.random.rand(10, 2), geometry=0, radius=1)

# shuffle the indicesof hte 3 dataframes
np.random.shuffle(visium_locations0.index)
np.random.shuffle(visium_locations1.index)
np.random.shuffle(visium_locations2.index)


def get_annotation_target_of_table(table: AnnData) -> str:
    return table.obs["__spatial_element__"]


def set_annotation_target_of_table(table: AnnData, spatial_element: str) -> None:
    table.obs["__spatial_element__"] = spatial_element

@pytest.mark.paramterreize("instance_id", [None, np.array([str(i) for i in range(20000)])])
def test_single_table(tmp_path: str, instance_id: Any):
    # shared table
    tmpdir = Path(tmp_path) / "tmp.zarr"
    adata0 = AnnData(np.random.rand(10, 20000))
    table = TableModel.parse(adata0, spatial_element: None | str = "<spatial_element_path>", instance_id=instance_id)

    visium = SpatialData(
        shapes={
            "visium0": visium_locations0,
        },
        tables={
            "segmentation": table
        },
    )
    visium.write(tmpdir)
    sdata = SpatialData.read(tmpdir)
    assert sdata.get("segmentation")
    assert isinstance(sdata["segmentation"], AnnData)
    from anndata.tests.helpers import assert_equal
    assert assert_equal(visium["segmentation"], sdata["segmentation"])

    # use case example 1
    # sorting the shapes to match the order of the table
    sdata["visium0"][sdata.table.obs["__instance_id__"]]
    assert ...
    # use case example 2
    # sorting the table to match the order of the shapes
    sdata.table.obs.set_index(keys=["__instance_id__"])
    sdata.table.obs[sdata["visium0"]]
    assert ...

def test_elements_transfer_annotation():
    # shared table
    tmpdir = Path(tmp_path) / "tmp.zarr"
    adata0 = AnnData(np.random.rand(10, 20000))
    table = TableModel.parse(adata0, spatial_element: None | str | Sequence[str] = "<spatial_element_path>", instance_id: None | Sequence[Any] = instance_id)

    visium = SpatialData(
        labels={"visium_label": "blobs_dataset_label_same_number_shapes"}
        shapes={
            "visium0": visium_locations0,
        },
        tables={
            "segmentation": table
        },
    )
    set_annotation_target_of_table(visium.table, "visium1")
    assert get_annotation_target_of_table(visium.table, "visium1")

def test_single_table_multiple_elements(tmp_path: str):
    tmpdir = Path(tmp_path) / "tmp.zarr"
    adata0 = AnnData(np.random.rand(10, 20000))
    table = TableModel.parse(adata0, spatial_element: None | str | Sequence[str] = ["<spatial_element_path>",...], instance_id: None | Sequence[Any] = instance_id)

    visium = SpatialData(
        shapes={
            "visium0": visium_locations0,
            "visium1": visium_locations1,
        },
        tables={
            "segmentation": table
        },
    )
    visium.write(visium)
    sdata = SpatialData.read(tmpdir)

    # use case example 1
    # sorting the shapes visium0 to match the order of the table
    sdata["visium0"][sdata.table.obs["__instance_id__"][sdata.table.obs["__spatial_element__"] == "visium0"]]
    assert ...
    # use case example 2
    # subsetting and sorting the table to match the order of the shapes visium0
    sub_table = sdata.table[sdata.table.obs["__spatial_element"] == "visium0"]
    sub_table.set_index(keys=["__instance_id__"])
    sub_table.obs[sdata["visium0"]]
    assert ...

def test_concatenate(tmp_path: str):
    table0 = TableModel.parse(adata0, spatial_element: None | str | Sequence[str] = ["<spatial_element_path>", ...], instance_id: None | Sequence[
    table1 = TableModel.parse(adata0, spatial_element: None | str | Sequence[str] = ["<spatial_element_path>",
                                                                                         ...], instance_id: None |
                                                                                                            Sequence[
                                                                                                                Any] = instance_i

   concatenated_table = ad.concat([table0, table1])
visium = SpatialData(
    shapes={
        "visium0": visium_locations0,
        "visium1": visium_locations1,
    },
    tables={
        "segmentation": concatenated_table
    },
)
    # use case tests as above (we test only visium0)

def test_multiple_table_without_element():
    table0 = TableModel.parse(adata0)
    table1 = TableModel.parse(adata0)

    visium = SpatialData(
        tables={
            "segmentation0": table0,
            "segmentation1": table1
        },
    )

    # nothing to test? probably nothing

def test_multiple_tables_same_element():
    tmpdir = Path(tmp_path) / "tmp.zarr"
    adata0 = AnnData(np.random.rand(10, 20000))
    table0 = TableModel.parse(adata0, spatial_element: None | str | Sequence[str] = ["<spatial_element_path>",...], instance_id: None | Sequence[Any] = instance_id)
    table1 = TableModel.parse(adata0, spatial_element: None | str | Sequence[str] = ["<spatial_element_path>",...], instance_id: None | Sequence[Any] = instance_id)

    visium = SpatialData(
        shapes={
            "visium0": visium_locations0,
        },
        tables={
            "segmentation0": table0,
            "segmentation1": table1
        },
    )

    # do we reallyneed to do the test below? maybe we can write something smarter
    # use cases
    # use case example 1
    # sorting the shapes to match the order of the table
    sdata["visium0"][sdata.table.obs["__instance_id__"]]
    assert ...
    # use case example 2
    # sorting the table to match the order of the shapes
    sdata.table.obs.set_index(keys=["__instance_id__"])
    sdata.table.obs[sdata["visium0"]]
    assert ...

def test_partial_match():
    # the function spatialdata._core.query.relational_query.match_table_to_element(no s) needs to be modified (will be
    # simpler), we need also a function match_element_to_table. Maybe we can have just one function doing both the things,
    # called match_table_and_elements test that tables and elements do not need to have the same indices
    pass
    # the test would check that we cna call SpatiaLData() on such combinations of mismatching elements and that the
    # match_table_to_element-like functions return the correct subset of the data
