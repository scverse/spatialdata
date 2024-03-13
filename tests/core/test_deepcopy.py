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

    assert_spatial_data_objects_are_identical(full_sdata, copied)
    assert_spatial_data_objects_are_identical(full_sdata, copied_again)

    for _, element_name, _ in full_sdata.gen_elements():
        assert full_sdata[element_name] is not copied[element_name]
        assert full_sdata[element_name] is not copied_again[element_name]
        assert copied[element_name] is not copied_again[element_name]
