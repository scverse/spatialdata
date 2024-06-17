import numpy as np
import pytest
from multiscale_spatial_image import MultiscaleSpatialImage
from spatial_image import SpatialImage
from spatialdata._core.operations.map import map_raster
from spatialdata.transformations import Translation, get_transformation, set_transformation


def _multiply(arr, parameter=10):
    return arr * parameter


def _multiply_alter_c(arr, parameter=10):
    arr = arr * parameter
    arr = arr[0]
    return arr[None, ...]


def _multiply_squeeze_z(arr, parameter=10):
    arr = arr * parameter
    return arr[:, 0, ...]


@pytest.mark.parametrize(
    "depth",
    [
        None,
        (0, 60, 60),
    ],
)
def test_map_raster(sdata_blobs, depth):
    img_layer = "blobs_image"
    fn_kwargs = {"parameter": 20}
    se = map_raster(
        sdata_blobs[img_layer],
        func=_multiply,
        fn_kwargs=fn_kwargs,
        chunks=(3, 100, 100),
        c_coords=None,
        scale_factors=None,
        depth=depth,
    )

    assert isinstance(se, SpatialImage)
    data = sdata_blobs[img_layer].data.compute()
    res = se.data.compute()
    assert np.array_equal(data * fn_kwargs["parameter"], res)


@pytest.mark.parametrize(
    "depth",
    [
        None,
        (0, 60, 60),
    ],
)
def test_map_raster_multiscale(sdata_blobs, depth):
    img_layer = "blobs_multiscale_image"
    fn_kwargs = {"parameter": 20}
    se = map_raster(
        sdata_blobs[img_layer],
        func=_multiply,
        fn_kwargs=fn_kwargs,
        chunks=(3, 100, 100),
        c_coords=None,
        scale_factors=[2, 2, 2, 2],
        depth=depth,
    )

    assert isinstance(se, MultiscaleSpatialImage)
    data = sdata_blobs[img_layer]["scale0"]["image"].data.compute()
    res = se["scale0"]["image"].data.compute()
    assert np.array_equal(data * fn_kwargs["parameter"], res)


def test_map_raster_chunks_none(sdata_blobs):
    img_layer = "blobs_image"
    fn_kwargs = {"parameter": 20}
    se = map_raster(
        sdata_blobs[img_layer],
        func=_multiply,
        fn_kwargs=fn_kwargs,
        chunks=None,
        c_coords=None,
        scale_factors=None,
        depth=None,
    )

    assert isinstance(se, SpatialImage)
    data = sdata_blobs[img_layer].data.compute()
    res = se.data.compute()
    assert np.array_equal(data * fn_kwargs["parameter"], res)


def test_map_raster_output_chunks(sdata_blobs):
    depth = 60
    fn_kwargs = {"parameter": 20}
    output_channels = ["test"]
    se = map_raster(
        sdata_blobs["blobs_image"],
        func=_multiply_alter_c,
        fn_kwargs=fn_kwargs,
        chunks=(3, 100, 100),
        output_chunks=(
            (1,),
            (100 + 2 * depth, 96 + 2 * depth, 60 + 2 * depth),
            (100 + 2 * depth, 96 + 2 * depth, 60 + 2 * depth),
        ),  # account for rechunking done by map_overlap to ensure minimum chunksize
        c_coords=["test"],
        scale_factors=None,
        depth=(0, depth, depth),
    )

    assert isinstance(se, SpatialImage)
    assert np.array_equal(np.array(output_channels), se.c.data)
    data = sdata_blobs["blobs_image"].data.compute()
    res = se.data.compute()
    assert np.array_equal(data[0] * fn_kwargs["parameter"], res[0])


@pytest.mark.parametrize(
    "img_layer, expected_type, scale_factors",
    [
        ("blobs_image", SpatialImage, None),
        ("blobs_multiscale_image", MultiscaleSpatialImage, [2, 2, 2, 2]),
    ],
)
def test_map_transformation(sdata_blobs, img_layer, expected_type, scale_factors):
    fn_kwargs = {"parameter": 20}
    target_coordinate_system = "my_other_space0"
    transformation = Translation(translation=[10, 12], axes=["y", "x"])

    se = sdata_blobs[img_layer]

    set_transformation(se, transformation=transformation, to_coordinate_system=target_coordinate_system)
    se = map_raster(
        sdata_blobs[img_layer],
        func=_multiply,
        fn_kwargs=fn_kwargs,
        chunks=None,
        c_coords=None,
        scale_factors=scale_factors,
        depth=None,
    )
    assert isinstance(se, expected_type)
    assert transformation == get_transformation(se, to_coordinate_system=target_coordinate_system)


def test_map_remove_z_fails(full_sdata):
    fn_kwargs = {"parameter": 20}

    # currently can not alter dims, e.g. ("c","z","y","x") -> ("c","y","x") fails
    # could be supported by adding dims (and possibly transformations) to parameters of map_raster
    with pytest.raises(IndexError):
        map_raster(
            full_sdata["image3d_numpy"],
            func=_multiply_squeeze_z,
            fn_kwargs=fn_kwargs,
            chunks=100,
            output_chunks=((3,), (64,), (64,)),
            drop_axis=1,
            c_coords=None,
            scale_factors=None,
            depth=None,
        )
