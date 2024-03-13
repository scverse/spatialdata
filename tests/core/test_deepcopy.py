from spatialdata._core._deepcopy import deepcopy as _deepcopy

# TODO: replace with spatialdata.testing.assert_equal when https://github.com/scverse/spatialdata/pull/473 is merged
from spatialdata._utils import _assert_spatialdata_objects_seem_identical


def test_deepcopy(full_sdata):
    to_delete = []
    for element_type, element_name in to_delete:
        del getattr(full_sdata, element_type)[element_name]

    copied = _deepcopy(full_sdata)
    # we first compute() the data in-place, then deepcopy and then we make the data lazy again; if the last step is
    # missing, calling _deepcopy() again on the original data would fail. Here we check for that.
    copied_again = _deepcopy(full_sdata)

    _assert_spatialdata_objects_seem_identical(full_sdata, copied)
    _assert_spatialdata_objects_seem_identical(full_sdata, copied_again)

    for _, element_name, _ in full_sdata.gen_elements():
        assert full_sdata[element_name] is not copied[element_name]
        assert full_sdata[element_name] is not copied_again[element_name]
        assert copied[element_name] is not copied_again[element_name]
