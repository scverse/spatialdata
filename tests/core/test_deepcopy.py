from pandas.testing import assert_frame_equal
from spatialdata._core._deepcopy import deepcopy as _deepcopy
from spatialdata.testing import assert_spatial_data_objects_are_identical


def test_deepcopy(full_sdata):
    to_delete = []
    for element_type, element_name in to_delete:
        del getattr(full_sdata, element_type)[element_name]

    copied = _deepcopy(full_sdata)
    # we first compute() the data in-place, then deepcopy and then we make the data lazy again; if the last step is
    # missing, calling _deepcopy() again on the original data would fail. Here we check for that.
    copied_again = _deepcopy(full_sdata)

    # workaround for https://github.com/scverse/spatialdata/issues/486
    for _, element_name, _ in full_sdata.gen_elements():
        assert full_sdata[element_name] is not copied[element_name]
        assert full_sdata[element_name] is not copied_again[element_name]
        assert copied[element_name] is not copied_again[element_name]

    p0_0 = full_sdata["points_0"].compute()
    columns = list(p0_0.columns)
    p0_1 = full_sdata["points_0_1"].compute()[columns]

    p1_0 = copied["points_0"].compute()[columns]
    p1_1 = copied["points_0_1"].compute()[columns]

    p2_0 = copied_again["points_0"].compute()[columns]
    p2_1 = copied_again["points_0_1"].compute()[columns]

    assert_frame_equal(p0_0, p1_0)
    assert_frame_equal(p0_1, p1_1)
    assert_frame_equal(p0_0, p2_0)
    assert_frame_equal(p0_1, p2_1)

    del full_sdata.points["points_0"]
    del full_sdata.points["points_0_1"]
    del copied.points["points_0"]
    del copied.points["points_0_1"]
    del copied_again.points["points_0"]
    del copied_again.points["points_0_1"]
    # end workaround

    assert_spatial_data_objects_are_identical(full_sdata, copied)
    assert_spatial_data_objects_are_identical(full_sdata, copied_again)
