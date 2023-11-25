from pathlib import Path

import anndata as ad
import numpy as np
from anndata import AnnData
from spatialdata import SpatialData

from tests.conftest import _get_new_table, _get_shapes

# notes on paths: https://github.com/orgs/scverse/projects/17/views/1?pane=issue&itemId=44066734
# notes for the people (to prettify) https://hackmd.io/wd7K4Eg1SlykKVN-nOP44w

# shapes
test_shapes = _get_shapes()
instance_id = np.array([str(i) for i in range(5)])
table = _get_new_table(spatial_element="test_shapes", instance_id=instance_id)


# shuffle the indices of the dataframe
np.random.default_rng().shuffle(test_shapes["poly"].index)


def get_annotation_target_of_table(table: AnnData) -> str:
    return table.obs["__spatial_element__"]


def set_annotation_target_of_table(table: AnnData, spatial_element: str) -> None:
    table.obs["__spatial_element__"] = spatial_element


class TestMultiTable:
    def test_single_table(self, tmp_path: str):
        # shared table
        tmpdir = Path(tmp_path) / "tmp.zarr"

        test_sdata = SpatialData(
            shapes={
                "test_shapes": test_shapes["poly"],
            },
            tables={"shape_annotate": table},
        )
        test_sdata.write(tmpdir)
        sdata = SpatialData.read(tmpdir)
        assert sdata.get("segmentation")
        assert isinstance(sdata["segmentation"], AnnData)
        from anndata.tests.helpers import assert_equal

        assert assert_equal(test_sdata["segmentation"], sdata["segmentation"])

        # # use case example 1
        # # sorting the shapes to match the order of the table
        # sdata["visium0"][sdata.table.obs["__instance_id__"]]
        # assert ...
        # # use case example 2
        # # sorting the table to match the order of the shapes
        # sdata.table.obs.set_index(keys=["__instance_id__"])
        # sdata.table.obs[sdata["visium0"]]
        # assert ...

    def test_elements_transfer_annotation(self, tmp_path: str):
        test_sdata = SpatialData(
            shapes={"test_shapes": test_shapes["poly"], "test_multipoly": test_shapes["multipoly"]},
            tables={"segmentation": table},
        )
        set_annotation_target_of_table(test_sdata["segmentation"], "test_multipoly")
        assert get_annotation_target_of_table(test_sdata["segmentation"]) == "test_multipoly"

    def test_single_table_multiple_elements(self, tmp_path: str):
        tmpdir = Path(tmp_path) / "tmp.zarr"

        test_sdata = SpatialData(
            shapes={
                "test_shapes": test_shapes["poly"],
                "test_multipoly": test_shapes["multi_poly"],
            },
            tables={"segmentation": table},
        )
        test_sdata.write(tmpdir)
        # sdata = SpatialData.read(tmpdir)

        # # use case example 1
        # # sorting the shapes visium0 to match the order of the table
        # sdata["visium0"][sdata.table.obs["__instance_id__"][sdata.table.obs["__spatial_element__"] == "visium0"]]
        # assert ...
        # # use case example 2
        # # subsetting and sorting the table to match the order of the shapes visium0
        # sub_table = sdata.table[sdata.table.obs["__spatial_element"] == "visium0"]
        # sub_table.set_index(keys=["__instance_id__"])
        # sub_table.obs[sdata["visium0"]]
        # assert ...

    def test_concatenate_tables(self):
        table_two = _get_new_table(spatial_element="test_multipoly", instance_id=np.array([str(i) for i in range(2)]))
        concatenated_table = ad.concat([table, table_two])
        test_sdata = SpatialData(
            shapes={
                "test_shapes": test_shapes["poly"],
                "test_multipoly": test_shapes["multi_poly"],
            },
            tables={"segmentation": concatenated_table},
        )
        # use case tests as above (we test only visium0)

    def test_multiple_table_without_element(self):
        table = _get_new_table()
        table_two = _get_new_table()

        test_sdata = SpatialData(
            tables={"table": table, "table_two": table_two},
        )

    def test_multiple_tables_same_element(self, tmp_path: str):
        tmpdir = Path(tmp_path) / "tmp.zarr"
        table_two = _get_new_table(spatial_element="test_shapes", instance_id=instance_id)

        test_sdata = SpatialData(
            shapes={
                "test_shapes": test_shapes["poly"],
            },
            tables={"segmentation": table, "segmentation_two": table_two},
        )
        test_sdata.write(tmpdir)


#
#     # do we reallyneed to do the test below? maybe we can write something smarter
#     # use cases
#     # use case example 1
#     # sorting the shapes to match the order of the table
#     sdata["visium0"][sdata.table.obs["__instance_id__"]]
#     assert ...
#     # use case example 2
#     # sorting the table to match the order of the shapes
#     sdata.table.obs.set_index(keys=["__instance_id__"])
#     sdata.table.obs[sdata["visium0"]]
#     assert ...
#
# def test_partial_match():
#     # the function spatialdata._core.query.relational_query.match_table_to_element(no s) needs to be modified (will be
#     # simpler), we need also a function match_element_to_table. Maybe we can have just one function doing both the
#     things,
#     # called match_table_and_elements test that tables and elements do not need to have the same indices
#     pass
#     # the test would check that we cna call SpatiaLData() on such combinations of mismatching elements and that the
#     # match_table_to_element-like functions return the correct subset of the data
