import re

import numpy as np
import pytest
from spatialdata._core.operations.map import map_raster
from spatialdata.transformations import Translation, get_transformation, set_transformation
from xarray import DataArray


def _multiply(arr, parameter=10):
    return arr * parameter


def _multiply_alter_c(arr, parameter=10):
    arr = arr * parameter
    arr = arr[0]
    return arr[None, ...]


def _multiply_squeeze_z(arr, parameter=10):
    arr = arr * parameter
    return arr[:, 0, ...]


def _multiply_to_labels(arr, parameter=10):
    arr = arr * parameter
    return arr[0].astype(np.int32)


@pytest.mark.parametrize(
    "depth",
    [
        None,
        (0, 60, 60),
    ],
)
@pytest.mark.parametrize("element_name", ["blobs_image", "blobs_labels"])
def test_map_raster(sdata_blobs, depth, element_name):
    if element_name == "blobs_labels" and depth is not None:
        depth = (60, 60)

    func_kwargs = {"parameter": 20}
    se = map_raster(
        sdata_blobs[element_name],
        func=_multiply,
        func_kwargs=func_kwargs,
        c_coords=None,
        depth=depth,
    )

    assert isinstance(se, DataArray)
    data = sdata_blobs[element_name].data.compute()
    res = se.data.compute()
    assert np.array_equal(data * func_kwargs["parameter"], res)


@pytest.mark.parametrize(
    "depth",
    [
        None,
        (0, 60, 60),
    ],
)
def test_map_raster_multiscale(sdata_blobs, depth):
    img_layer = "blobs_multiscale_image"
    func_kwargs = {"parameter": 20}
    se = map_raster(
        sdata_blobs[img_layer],
        func=_multiply,
        func_kwargs=func_kwargs,
        c_coords=None,
        depth=depth,
    )

    data = sdata_blobs[img_layer]["scale0"]["image"].data.compute()
    res = se.data.compute()
    assert np.array_equal(data * func_kwargs["parameter"], res)


def test_map_raster_no_blockwise(sdata_blobs):
    img_layer = "blobs_image"
    func_kwargs = {"parameter": 20}
    se = map_raster(
        sdata_blobs[img_layer],
        func=_multiply,
        func_kwargs=func_kwargs,
        blockwise=False,
        c_coords=None,
        depth=None,
    )

    assert isinstance(se, DataArray)
    data = sdata_blobs[img_layer].data.compute()
    res = se.data.compute()
    assert np.array_equal(data * func_kwargs["parameter"], res)


def test_map_raster_output_chunks(sdata_blobs):
    depth = 60
    func_kwargs = {"parameter": 20}
    output_channels = ["test"]
    se = map_raster(
        sdata_blobs["blobs_image"].chunk((3, 100, 100)),
        func=_multiply_alter_c,
        func_kwargs=func_kwargs,
        chunks=(
            (1,),
            (100 + 2 * depth, 96 + 2 * depth, 60 + 2 * depth),
            (100 + 2 * depth, 96 + 2 * depth, 60 + 2 * depth),
        ),  # account for rechunking done by map_overlap to ensure minimum chunksize
        c_coords=["test"],
        depth=(0, depth, depth),
    )

    assert isinstance(se, DataArray)
    assert np.array_equal(np.array(output_channels), se.c.data)
    data = sdata_blobs["blobs_image"].data.compute()
    res = se.data.compute()
    assert np.array_equal(data[0] * func_kwargs["parameter"], res[0])


@pytest.mark.parametrize("img_layer", ["blobs_image", "blobs_multiscale_image"])
def test_map_transformation(sdata_blobs, img_layer):
    func_kwargs = {"parameter": 20}
    target_coordinate_system = "my_other_space0"
    transformation = Translation(translation=[10, 12], axes=["y", "x"])

    se = sdata_blobs[img_layer]

    set_transformation(se, transformation=transformation, to_coordinate_system=target_coordinate_system)
    se = map_raster(
        se,
        func=_multiply,
        func_kwargs=func_kwargs,
        blockwise=False,
        c_coords=None,
        depth=None,
    )
    assert transformation == get_transformation(se, to_coordinate_system=target_coordinate_system)


@pytest.mark.parametrize(
    "blockwise, chunks, drop_axis",
    [
        (False, None, None),
        (True, ((256,), (256,)), 0),
    ],
)
def test_map_to_labels_(sdata_blobs, blockwise, chunks, drop_axis):
    img_layer = "blobs_image"
    func_kwargs = {"parameter": 20}

    se = sdata_blobs[img_layer]

    se = map_raster(
        se.chunk((3, 256, 256)),
        func=_multiply_to_labels,
        func_kwargs=func_kwargs,
        c_coords=None,
        blockwise=blockwise,
        chunks=chunks,
        drop_axis=drop_axis,
        dims=("y", "x"),
    )

    data = sdata_blobs[img_layer].data.compute()
    res = se.data.compute()
    assert np.array_equal((data[0] * func_kwargs["parameter"]).astype(np.int32), res)


def test_map_squeeze_z(full_sdata):
    img_layer = "image3d_numpy"
    func_kwargs = {"parameter": 20}

    se = map_raster(
        full_sdata[img_layer].chunk((3, 2, 64, 64)),
        func=_multiply_squeeze_z,
        func_kwargs=func_kwargs,
        chunks=((3,), (64,), (64,)),
        drop_axis=1,
        c_coords=None,
        dims=("c", "y", "x"),
        depth=None,
    )

    assert isinstance(se, DataArray)
    data = full_sdata[img_layer].data.compute()
    res = se.data.compute()
    assert np.array_equal(data[:, 0, ...] * func_kwargs["parameter"], res)


def test_map_squeeze_z_fails(full_sdata):
    img_layer = "image3d_numpy"
    func_kwargs = {"parameter": 20}

    with pytest.raises(
        ValueError,
        match=re.escape(
            "The number of dimensions of the output data (3) "
            "differs from the number of dimensions in 'dims' (('c', 'z', 'y', 'x')). "
            "Please provide correct output dimension via the 'dims' parameter.",
        ),
    ):
        map_raster(
            full_sdata[img_layer].chunk((3, 2, 64, 64)),
            func=_multiply_squeeze_z,
            func_kwargs=func_kwargs,
            chunks=((3,), (64,), (64,)),
            drop_axis=1,
            c_coords=None,
            depth=None,
        )


def test_invalid_map_raster(sdata_blobs):
    with pytest.raises(ValueError, match="Only 'DataArray' and 'DataTree' are supported."):
        map_raster(
            sdata_blobs["blobs_points"],
            func=_multiply,
            func_kwargs={"parameter": 20},
            c_coords=None,
            depth=(0, 60),
        )

    with pytest.raises(
        ValueError,
        match=re.escape("Depth (0, 60) is provided for 2 dimensions. Please provide depth for 3 dimensions."),
    ):
        map_raster(
            sdata_blobs["blobs_image"],
            func=_multiply,
            func_kwargs={"parameter": 20},
            c_coords=None,
            depth=(0, 60),
        )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Channel coordinates `c_coords` can not be provided if output data consists of labels "
            "('c' channel missing)."
        ),
    ):
        map_raster(
            sdata_blobs["blobs_labels"],
            func=_multiply,
            func_kwargs={"parameter": 20},
            c_coords=["c"],
            depth=(0, 60, 60),
        )
