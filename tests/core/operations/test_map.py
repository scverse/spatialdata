import math
import re

import dask.array as da
import numpy as np
import pytest
from xarray import DataArray

from spatialdata._core.operations.map import map_raster, relabel_sequential
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


def _multiply_to_labels(arr, parameter=10):
    arr = arr * parameter
    return arr[0].astype(np.int32)


def _to_constant(arr, constant):
    arr[arr > 0] = constant
    return arr


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
        relabel=False,
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
        sdata_blobs["blobs_image"].chunk({"c": 3, "y": 100, "x": 100}),
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
        se.chunk({"c": 3, "y": 256, "x": 256}),
        func=_multiply_to_labels,
        func_kwargs=func_kwargs,
        c_coords=None,
        blockwise=blockwise,
        chunks=chunks,
        drop_axis=drop_axis,
        dims=("y", "x"),
        relabel=False,
    )

    data = sdata_blobs[img_layer].data.compute()
    res = se.data.compute()
    assert np.array_equal((data[0] * func_kwargs["parameter"]).astype(np.int32), res)


def test_map_squeeze_z(full_sdata):
    img_layer = "image3d_numpy"
    func_kwargs = {"parameter": 20}

    se = map_raster(
        full_sdata[img_layer].chunk({"c": 3, "z": 2, "y": 64, "x": 64}),
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
            full_sdata[img_layer].chunk({"c": 3, "z": 2, "y": 64, "x": 64}),
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


def test_map_raster_relabel(sdata_blobs):
    constant = 2047
    func_kwargs = {"constant": constant}

    element_name = "blobs_labels"
    se = map_raster(
        sdata_blobs[element_name].chunk({"y": 100, "x": 100}),
        func=_to_constant,
        func_kwargs=func_kwargs,
        c_coords=None,
        depth=None,
        relabel=True,
    )

    # check if labels in different blocks are all mapped to a different value
    assert isinstance(se, DataArray)
    se.data.compute()
    a = set()
    for chunk in se.data.to_delayed().flatten():
        chunk = chunk.compute()
        b = set(np.unique(chunk))
        b.remove(0)
        assert not b.intersection(a)
        a.update(b)
    # 9 blocks, each block contains 'constant' left shifted by (9-1).bit_length() + block_num.
    shift = (math.prod(se.data.numblocks) - 1).bit_length()
    assert a == set(range(constant << shift, (constant << shift) + math.prod(se.data.numblocks)))


def test_map_raster_relabel_fail(sdata_blobs):
    constant = 2048
    func_kwargs = {"constant": constant}

    element_name = "blobs_labels"

    # Testing the case of having insufficient number of bits.
    with pytest.raises(
        ValueError,
        match=re.escape("Relabel was set to True, but"),
    ):
        se = map_raster(
            sdata_blobs[element_name].chunk({"y": 100, "x": 100}),
            func=_to_constant,
            func_kwargs=func_kwargs,
            c_coords=None,
            depth=None,
            relabel=True,
        )

        se.data.compute()

    constant = 2047
    func_kwargs = {"constant": constant}

    element_name = "blobs_labels"
    with pytest.raises(
        ValueError,
        match=re.escape(f"Relabeling is only supported for arrays of type {np.integer}."),
    ):
        map_raster(
            sdata_blobs[element_name].astype(float).chunk({"y": 100, "x": 100}),
            func=_to_constant,
            func_kwargs=func_kwargs,
            c_coords=None,
            depth=None,
            relabel=True,
        )


def test_relabel_sequential(sdata_blobs):
    def _is_sequential(arr):
        if arr.ndim != 1:
            raise ValueError("Input array must be one-dimensional")
        sorted_arr = np.sort(arr)
        expected_sequence = np.arange(sorted_arr[0], sorted_arr[0] + len(sorted_arr))
        return np.array_equal(sorted_arr, expected_sequence)

    arr = sdata_blobs["blobs_labels"].data.rechunk(100)

    arr_relabeled = relabel_sequential(arr)

    labels_relabeled = da.unique(arr_relabeled).compute()
    labels_original = da.unique(arr).compute()

    assert labels_relabeled.shape == labels_original.shape
    assert _is_sequential(labels_relabeled)

    # test some edge cases
    arr = da.asarray(np.array([0]))
    assert np.array_equal(relabel_sequential(arr).compute(), np.array([0]))

    arr = da.asarray(np.array([1]))
    assert np.array_equal(relabel_sequential(arr).compute(), np.array([1]))

    arr = da.asarray(np.array([2]))
    assert np.array_equal(relabel_sequential(arr).compute(), np.array([1]))

    arr = da.asarray(np.array([2, 0]))
    assert np.array_equal(relabel_sequential(arr).compute(), np.array([1, 0]))

    arr = da.asarray(np.array([0, 9, 5]))
    assert np.array_equal(relabel_sequential(arr).compute(), np.array([0, 2, 1]))

    arr = da.asarray(np.array([4, 1, 3]))
    assert np.array_equal(relabel_sequential(arr).compute(), np.array([3, 1, 2]))


def test_relabel_sequential_fails(sdata_blobs):
    with pytest.raises(
        ValueError, match=re.escape(f"Sequential relabeling is only supported for arrays of type {np.integer}.")
    ):
        relabel_sequential(sdata_blobs["blobs_labels"].data.astype(float))
