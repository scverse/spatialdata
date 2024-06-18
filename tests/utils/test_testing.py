import copy

from datatree import DataTree
from spatialdata import SpatialData, deepcopy
from spatialdata.models import (
    Image2DModel,
    Image3DModel,
    Labels2DModel,
    Labels3DModel,
    PointsModel,
    ShapesModel,
    TableModel,
    get_model,
)
from spatialdata.testing import assert_elements_are_identical, assert_spatial_data_objects_are_identical
from spatialdata.transformations import Scale, set_transformation

scale = Scale([1.0], axes=("x",))


def _change_metadata_points(sdata: SpatialData, element_name: str, attrs: bool, transformations: bool) -> None:
    element = sdata[element_name]
    if attrs:
        # incorrect new values, just for them to be different from the original ones
        element.attrs[PointsModel.ATTRS_KEY][PointsModel.FEATURE_KEY] = "a"
        element.attrs[PointsModel.ATTRS_KEY][PointsModel.INSTANCE_KEY] = "b"
    if transformations:
        set_transformation(element, copy.deepcopy(scale))


def _change_metadata_shapes(sdata: SpatialData, element_name: str) -> None:
    set_transformation(sdata[element_name], copy.deepcopy(scale))


def _change_metadata_tables(sdata: SpatialData, element_name: str) -> None:
    element = sdata[element_name]
    # incorrect new values, just for them to be different from the original ones
    element.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY] = "circles"
    element.uns[TableModel.ATTRS_KEY][TableModel.REGION_KEY_KEY] = "a"
    element.uns[TableModel.ATTRS_KEY][TableModel.INSTANCE_KEY] = "b"


def test_assert_elements_are_identical_metadata(full_sdata):
    assert_spatial_data_objects_are_identical(full_sdata, full_sdata)

    copied = deepcopy(full_sdata)
    assert_spatial_data_objects_are_identical(full_sdata, copied)

    to_iter = list(copied.gen_elements())
    for _, element_name, element in to_iter:
        if get_model(element) in (Image2DModel, Image3DModel) or get_model(element) in (Labels2DModel, Labels3DModel):
            if not isinstance(copied[element_name], DataTree):
                assert_elements_are_identical(full_sdata[element_name], copied[element_name])
        elif get_model(element) == PointsModel or get_model(element) == ShapesModel:
            assert_elements_are_identical(full_sdata[element_name], copied[element_name])
        else:
            assert get_model(element) == TableModel
            assert_elements_are_identical(full_sdata[element_name], copied[element_name])

    assert_spatial_data_objects_are_identical(full_sdata, copied)
