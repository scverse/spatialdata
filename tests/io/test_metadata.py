import logging
import os
import tempfile

import pytest

from spatialdata import SpatialData, read_zarr
from spatialdata._io._utils import _is_element_self_contained
from spatialdata._logging import logger
from spatialdata.transformations import Scale, get_transformation, set_transformation


def test_save_transformations(full_sdata):
    """test io for transformations"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        f = os.path.join(tmp_dir, "data.zarr")
        scale = Scale([2, 2], axes=("x", "y"))

        element_names = ["image2d", "labels2d", "points_0", "circles"]
        for element_name in element_names:
            set_transformation(full_sdata[element_name], scale)

        full_sdata.write(f)

        sdata = read_zarr(f)
        for element_name in element_names:
            scale0 = get_transformation(sdata[element_name])
            assert isinstance(scale0, Scale)


@pytest.mark.parametrize("element_name", ["image2d", "labels2d", "points_0", "circles"])
def test_validate_can_write_metadata_on_element(full_sdata, element_name):
    with tempfile.TemporaryDirectory() as tmp_dir:
        # element not present in the SpatialData object
        with pytest.raises(
            ValueError,
            match="Cannot save the metadata to the element as it has not been found in the SpatialData object.",
        ):
            full_sdata._validate_can_write_metadata_on_element("invalid")

        # trying to save metadata before writing the data
        with pytest.warns(
            UserWarning,
            match="The SpatialData object appears not to be backed by a Zarr storage, so metadata cannot be "
            "written.",
        ):
            full_sdata._validate_can_write_metadata_on_element(element_name)

        f0 = os.path.join(tmp_dir, f"{element_name}0.zarr")
        full_sdata.write(f0)
        full_sdata[f"{element_name}_again"] = full_sdata[element_name]

        # the new element is not saved yet
        with pytest.warns(UserWarning, match="Not saving the metadata to element"):
            full_sdata._validate_can_write_metadata_on_element(f"{element_name}_again")


@pytest.mark.parametrize("element_name", ["image2d", "labels2d", "points_0", "circles"])
def test_save_transformations_incremental(element_name, full_sdata, caplog):
    """test io for transformations"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        f0 = os.path.join(tmp_dir, f"{element_name}0.zarr")
        full_sdata.write(f0)
        full_sdata[f"{element_name}_again"] = full_sdata[element_name]

        # save the element and then save the transformation
        scale = Scale([2, 2], axes=("x", "y"))
        full_sdata.write_element(f"{element_name}_again")
        set_transformation(element=full_sdata[f"{element_name}_again"], transformation=scale)
        full_sdata.write_transformations(f"{element_name}_again")

        # other way to save the metadata, but again, the element is not saved yet
        full_sdata[f"{element_name}_again_again"] = full_sdata[f"{element_name}_again"]
        with pytest.warns(UserWarning, match="Not saving the metadata to element"):
            set_transformation(
                element=full_sdata[f"{element_name}_again_again"], transformation=scale, write_to_sdata=full_sdata
            )

        # save the element and then save the transformation
        full_sdata.write_element(f"{element_name}_again_again")
        set_transformation(
            element=full_sdata[f"{element_name}_again_again"], transformation=scale, write_to_sdata=full_sdata
        )

        # check that the transformation is saved
        sdata2 = read_zarr(f0)
        assert isinstance(get_transformation(sdata2[f"{element_name}_again"]), Scale)
        assert isinstance(get_transformation(sdata2[f"{element_name}_again_again"]), Scale)

        f1 = os.path.join(tmp_dir, f"{element_name}1.zarr")
        sdata2.write(f1)
        assert not sdata2.is_self_contained()

        # check that the user gets a logger.info() when the transformation is saved to a non-self-contained element
        # (points, images, labels)
        element_type = sdata2._element_type_from_element_name(element_name)
        element_path = sdata2.path / element_type / element_name
        element_self_contained = _is_element_self_contained(sdata2[element_name], element_path=element_path)
        if element_name == "circles":
            assert element_self_contained
        else:
            assert element_name in ["image2d", "labels2d", "points_0"]

            assert not element_self_contained

            logger.propagate = True
            with caplog.at_level(logging.INFO):
                sdata2.write_transformations(element_name)
                assert f"Element {element_type}/{element_name} is not self-contained." in caplog.text
            logger.propagate = False


# test io for channel names
@pytest.mark.skip(reason="Not implemented yet")
def test_save_channel_names_incremental(images: SpatialData) -> None:
    # note: the non-incremental IO for channel names is already covered in TestReadWrite.test_images(), so here we
    # only test the incremental IO
    pass


# test io for consolidated metadata
def test_consolidated_metadata(full_sdata: SpatialData) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        f0 = os.path.join(tmp_dir, "data0.zarr")
        full_sdata.write(f0)
        assert full_sdata.has_consolidated_metadata()

        f1 = os.path.join(tmp_dir, "data1.zarr")
        full_sdata.write(f1, consolidate_metadata=False)
        assert not full_sdata.has_consolidated_metadata()

        full_sdata.write_metadata(consolidate_metadata=True)
        assert full_sdata.has_consolidated_metadata()


def test_save_all_metadata(full_sdata: SpatialData) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        # write
        f = os.path.join(tmp_dir, "data.zarr")
        full_sdata.write(f, consolidate_metadata=False)

        # set transformations in memory
        scale = Scale([2, 2], axes=("x", "y"))
        for _, _, element in full_sdata.gen_spatial_elements():
            set_transformation(element, scale)

        # write transformations, read, check that transformations are correct
        full_sdata.write_transformations()
        sdata0 = read_zarr(f)
        assert not sdata0.has_consolidated_metadata()
        for _, _, element in sdata0.gen_spatial_elements():
            assert isinstance(get_transformation(element), Scale)

        # write metadata, check that consolidated metadata is correct
        full_sdata.write_metadata(consolidate_metadata=True)
        assert full_sdata.has_consolidated_metadata()
