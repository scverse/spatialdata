from pathlib import Path

import anndata as ad
import numpy as np
import pytest
from anndata import AnnData
from anndata.tests.helpers import assert_equal
from spatialdata import SpatialData
from spatialdata.models import TableModel

from tests.conftest import _get_new_table, _get_table

# notes on paths: https://github.com/orgs/scverse/projects/17/views/1?pane=issue&itemId=44066734
# notes for the people (to prettify) https://hackmd.io/wd7K4Eg1SlykKVN-nOP44w


# shuffle the indices of the dataframe
# np.random.default_rng().shuffle(test_shapes["poly"].index)


class TestMultiTable:
    def test_set_get_tables_from_spatialdata(self, full_sdata: SpatialData, tmp_path: str):
        tmpdir = Path(tmp_path) / "tmp.zarr"
        adata0 = _get_table(region="polygon")
        adata1 = _get_table(region="multipolygon")
        full_sdata["adata0"] = adata0
        full_sdata["adata1"] = adata1
        assert len(full_sdata.tables) == 3
        assert "adata0" in full_sdata.tables.keys() and "adata1" in full_sdata.tables.keys()
        full_sdata.write(tmpdir)

        full_sdata = SpatialData.read(tmpdir)
        assert_equal(adata0, full_sdata["adata0"])
        assert_equal(adata1, full_sdata["adata1"])
        assert "adata0" in full_sdata.tables.keys() and "adata1" in full_sdata.tables.keys()

    @pytest.mark.parametrize(
        "region_key, instance_key, error_msg",
        [
            (None, None, "Instance key and/or instance"),
            ("region", None, "Instance key and/or instance"),
            ("region", "instance_id", "Instance key instance_id not"),
            (None, "instance_id", "Instance key instance_id not"),
        ],
    )
    def test_change_annotation_target(self, full_sdata, region_key, instance_key, error_msg):
        n_obs = full_sdata["table"].n_obs
        with pytest.raises(ValueError, match="Mismatch found between regions"):
            full_sdata.set_table_annotation_target("table", "poly")

        del full_sdata["table"].obs["region"]
        with pytest.raises(ValueError, match="Region key and/or region key"):
            full_sdata.set_table_annotation_target("table", "poly")

        del full_sdata["table"].obs["instance_id"]
        full_sdata["table"].obs["region"] = ["poly"] * n_obs
        with pytest.raises(ValueError, match=error_msg):
            full_sdata.set_table_annotation_target("table", "poly", region_key=region_key, instance_key=instance_key)

        full_sdata["table"].obs["instance_id"] = [i for i in range(n_obs)]
        full_sdata.set_table_annotation_target("table", "poly", instance_key="instance_id", region_key=region_key)

        with pytest.raises(ValueError, match="not_existing not present"):
            full_sdata.set_table_annotation_target("table", "circles", region_key="not_existing")

    def test_set_table_nonexisting_target(self, full_sdata):
        with pytest.raises(ValueError, match="Annotation target"):
            full_sdata.set_table_annotation_target("table", "non_existing")

    def test_set_table_annotation_target(self, full_sdata):
        del full_sdata["table"].uns[TableModel.ATTRS_KEY]
        with pytest.raises(ValueError, match="Specified region_key"):
            full_sdata.set_table_annotation_target("table", "labels2d", region_key="non_existent")
        with pytest.raises(ValueError, match="Specified instance_key"):
            full_sdata.set_table_annotation_target(
                "table", "labels2d", region_key="region", instance_key="non_existent"
            )
        full_sdata.set_table_annotation_target("table", "labels2d", region_key="region", instance_key="instance_id")

    def test_old_accessor_deprecation(self, full_sdata, tmp_path):
        # To test self._backed
        tmpdir = Path(tmp_path) / "tmp.zarr"
        full_sdata.write(tmpdir)
        adata0 = _get_table(region="polygon")

        with pytest.warns(DeprecationWarning):
            _ = full_sdata.table
        with pytest.raises(ValueError):
            full_sdata.table = adata0
        with pytest.warns(DeprecationWarning):
            del full_sdata.table
        with pytest.raises(KeyError):
            del full_sdata.table
        with pytest.warns(DeprecationWarning):
            full_sdata.table = adata0  # this gets placed in sdata['table']

        assert_equal(adata0, full_sdata.table)

        del full_sdata.table

        full_sdata.tables["my_new_table0"] = adata0
        assert full_sdata.table is None

    def test_single_table(self, tmp_path: str):
        # shared table
        tmpdir = Path(tmp_path) / "tmp.zarr"

        test_sdata = SpatialData(
            shapes={
                "test_shapes": test_shapes["poly"],
            },
            table={"shape_annotate": table},
        )
        test_sdata.write(tmpdir)
        sdata = SpatialData.read(tmpdir)

        assert isinstance(sdata["shape_annotate"], AnnData)

        assert_equal(test_sdata["shape_annotate"], sdata["shape_annotate"])

        # note (to keep in the code): these tests here should silmulate the interactions from teh users; if the syntax
        # here we are matching the table to the shapes and viceversa (= subset + reordeing)
        # there is already a function to do one of these two join operations which is match_table_to_element()
        # is too verbose/complex we need to adjust the internals to make it smoother
        # # use case example 1
        # # sorting the shapes to match the order of the table
        # alternatively, we can have a helper function (join, and simpler ones "match_table_to_element()"
        # "match_element_to_table()", "match_annotations_order(...)", "mathc_reference_eleemnt_order??(...)")
        # sdata["visium0"][SpatialData.get_instance_key_column(sdata.table['visium0'])]
        # assert ...
        # # use case example 2
        # # sorting the table to match the order of the shapes
        # sdata.table.obs.set_index(keys=["__instance_id__"])
        # sdata.table.obs[sdata["visium0"]]
        # assert ...

    def test_paired_elements_tables(self, tmp_path: str):
        tmpdir = Path(tmp_path) / "tmp.zarr"

        test_sdata = SpatialData(
            shapes={
                "test_shapes": test_shapes["poly"],
            },
            table={
                "shape_annotate": table,
            },
        )

    def test_single_table_multiple_elements(self, tmp_path: str):
        tmpdir = Path(tmp_path) / "tmp.zarr"

        test_sdata = SpatialData(
            shapes={
                "test_shapes": test_shapes["poly"],
                "test_multipoly": test_shapes["multipoly"],
            },
            tables={"segmentation": table},
        )
        test_sdata.write(tmpdir)
        SpatialData.read(tmpdir)

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
                "test_multipoly": test_shapes["multipoly"],
            },
            tables={"segmentation": concatenated_table},
        )
        # use case tests as above (we test only visium0)

    def test_multiple_table_without_element(self):
        table = _get_new_table()
        table_two = _get_new_table()

        SpatialData(
            tables={"table": table, "table_two": table_two},
        )

    def test_multiple_tables_same_element(self, tmp_path: str):
        tmpdir = Path(tmp_path) / "tmp.zarr"
        table_two = _get_new_table(spatial_element="test_shapes", instance_id=instance_id)

        test_sdata = SpatialData(
            shapes={
                "test_shapes": test_shapes["poly"],
            },
            tables={"region_props": table, "codebook": table_two},
        )
        test_sdata.write(tmpdir)


#
#     # these use cases could be the preferred one for the users; we need to choose one/two preferred ones (either this, either helper function, ...)
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
