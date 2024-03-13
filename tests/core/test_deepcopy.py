from spatialdata._core._deepcopy import deepcopy as _deepcopy

# TODO: replace with spatialdata.testing.assert_equal when https://github.com/scverse/spatialdata/pull/473 is merged
# TODO: when multiscale_spatial_image 1.0.0 is supported, re-enable the deepcopy and test for DataTree
from spatialdata._utils import _assert_spatialdata_objects_seem_identical


def test_deepcopy(full_sdata):
    to_delete = []
    for element_type, element_name, _ in full_sdata.gen_elements():
        if "multiscale" in element_name:
            to_delete.append((element_type, element_name))
    for element_type, element_name in to_delete:
        del getattr(full_sdata, element_type)[element_name]

    copied = _deepcopy(full_sdata)

    _assert_spatialdata_objects_seem_identical(full_sdata, copied)

    for _, element_name, _ in full_sdata.gen_elements():
        assert full_sdata[element_name] is not copied[element_name]
