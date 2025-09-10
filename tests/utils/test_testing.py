import copy

import numpy as np
import pytest
from xarray import DataArray, DataTree

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


def _change_metadata_image(sdata: SpatialData, element_name: str, coords: bool, transformations: bool) -> None:
    if coords:
        if isinstance(sdata[element_name], DataArray):
            sdata[element_name] = sdata[element_name].assign_coords({"c": np.array(["m", "l", "b"])})
        else:
            assert isinstance(sdata[element_name], DataTree)

            dt = sdata[element_name].assign_coords({"c": np.array(["m", "l", "b"])})
            sdata[element_name] = dt
    if transformations:
        set_transformation(sdata[element_name], copy.deepcopy(scale))


def _change_metadata_labels(sdata: SpatialData, element_name: str) -> None:
    set_transformation(sdata[element_name], copy.deepcopy(scale))


def test_assert_elements_are_identical_metadata(full_sdata):
    assert_spatial_data_objects_are_identical(full_sdata, full_sdata)

    copied = deepcopy(full_sdata)
    assert_spatial_data_objects_are_identical(full_sdata, copied)

    to_iter = list(copied.gen_elements())
    for _, element_name, element in to_iter:
        if get_model(element) in (Image2DModel, Image3DModel):
            if not isinstance(copied[element_name], DataTree):
                assert_elements_are_identical(full_sdata[element_name], copied[element_name])
                _change_metadata_image(copied, element_name, coords=True, transformations=False)
                with pytest.raises(AssertionError):
                    assert_elements_are_identical(full_sdata[element_name], copied[element_name])
        elif get_model(element) in (Labels2DModel, Labels3DModel):
            if not isinstance(copied[element_name], DataTree):
                assert_elements_are_identical(full_sdata[element_name], copied[element_name])
                _change_metadata_labels(copied, element_name)
                with pytest.raises(AssertionError):
                    assert_elements_are_identical(full_sdata[element_name], copied[element_name])
        elif get_model(element) == PointsModel:
            assert_elements_are_identical(full_sdata[element_name], copied[element_name])
            _change_metadata_points(copied, element_name, attrs=True, transformations=False)
            with pytest.raises(AssertionError):
                assert_elements_are_identical(full_sdata[element_name], copied[element_name])
        elif get_model(element) == ShapesModel:
            assert_elements_are_identical(full_sdata[element_name], copied[element_name])
            _change_metadata_shapes(copied, element_name)
            with pytest.raises(AssertionError):
                assert_elements_are_identical(full_sdata[element_name], copied[element_name])
        else:
            assert get_model(element) == TableModel
            assert_elements_are_identical(full_sdata[element_name], copied[element_name])
            _change_metadata_tables(copied, element_name)
            with pytest.raises(AssertionError):
                assert_elements_are_identical(full_sdata[element_name], copied[element_name])

    with pytest.raises(AssertionError):
        assert_spatial_data_objects_are_identical(full_sdata, copied)
