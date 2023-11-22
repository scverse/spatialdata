import numpy as np
import pytest

from spatialdata import SpatialData
from spatialdata._core.operations.apply import _precondition


def _multiply(image, parameter=2):
    return image * parameter


def _add(image, parameter_add=1):
    return image + parameter_add


def test_apply(sdata_blobs: SpatialData):
    fn_kwargs = {"parameter": 4}

    # combine_c == True -> process channels all at once
    sdata_blobs = sdata_blobs.apply(
        func=_multiply,
        fn_kwargs=fn_kwargs,
        img_layer="blobs_image",
        output_layer="blobs_apply",
        combine_c=True,
        combine_z=True,
        overwrite=True,
        chunks=212,
        scale_factors=None,
    )

    res = sdata_blobs.images["blobs_image"].compute()
    res2 = sdata_blobs.images["blobs_apply"].compute()
    assert np.array_equal(res * fn_kwargs["parameter"], res2)

    # combine_c==False -> process channels one by one
    sdata_blobs = sdata_blobs.apply(
        func=_multiply,
        fn_kwargs=fn_kwargs,
        img_layer="blobs_image",
        output_layer="blobs_apply",
        combine_c=False,
        combine_z=True,
        overwrite=True,
        chunks=212,
        scale_factors=None,
    )

    res = sdata_blobs.images["blobs_image"].compute()
    res2 = sdata_blobs.images["blobs_apply"].compute()
    assert np.array_equal(res * fn_kwargs["parameter"], res2)

    # only process channels 0 and 2, combine_c==True
    channel = [0, 2]
    sdata_blobs = sdata_blobs.apply(
        func=_multiply,
        fn_kwargs=fn_kwargs,
        img_layer="blobs_image",
        output_layer="blobs_apply",
        channel=channel,
        combine_c=True,
        combine_z=True,
        overwrite=True,
        chunks=212,
        scale_factors=None,
    )

    for c in channel:
        res = sdata_blobs.images["blobs_image"].sel(c=c).compute()
        res2 = sdata_blobs.images["blobs_apply"].sel(c=c).compute()
        assert np.array_equal(res * fn_kwargs["parameter"], res2)

    with pytest.raises(KeyError):
        sdata_blobs.images["blobs_apply"].sel(c=1)

    # only process channels 0 and 2, combine_c==False
    channel = [0, 2]
    sdata_blobs = sdata_blobs.apply(
        func=_multiply,
        fn_kwargs=fn_kwargs,
        img_layer="blobs_image",
        output_layer="blobs_apply",
        channel=channel,
        combine_c=False,
        combine_z=True,
        overwrite=True,
        chunks=212,
        scale_factors=None,
    )

    for c in channel:
        res = sdata_blobs.images["blobs_image"].sel(c=c).compute()
        res2 = sdata_blobs.images["blobs_apply"].sel(c=c).compute()
        assert np.array_equal(res * fn_kwargs["parameter"], res2)

    with pytest.raises(KeyError):
        sdata_blobs.images["blobs_apply"].sel(c=1)


def test_apply_multiple_parameters(sdata_blobs: SpatialData):
    fn_kwargs = {0: {"parameter": 4}, 2: {"parameter": 8}}

    # if combine c = True, but there is a channel dimension specified in kwargs, then
    # raise a ValueError
    with pytest.raises(ValueError):
        _ = sdata_blobs.apply(
            func=_multiply,
            fn_kwargs=fn_kwargs,
            img_layer="blobs_image",
            output_layer="blobs_apply",
            combine_c=True,
            combine_z=True,
            overwrite=True,
            chunks=212,
            scale_factors=None,
        )

    # setting combine_c to False, now channels 0 and 2 are processed independently
    sdata_blobs = sdata_blobs.apply(
        func=_multiply,
        fn_kwargs=fn_kwargs,
        img_layer="blobs_image",
        output_layer="blobs_apply",
        combine_c=False,
        combine_z=True,
        overwrite=True,
        chunks=212,
        scale_factors=None,
    )

    for c in fn_kwargs.keys():
        res = sdata_blobs.images["blobs_image"].sel(c=c).compute()
        res2 = sdata_blobs.images["blobs_apply"].sel(c=c).compute()
        assert np.array_equal(res * fn_kwargs[c]["parameter"], res2)

    with pytest.raises(KeyError):
        sdata_blobs.images["blobs_apply"].sel(c=1)


def test_apply_multiple_functions(sdata_blobs: SpatialData):
    func = {0: _multiply, 2: _add}
    fn_kwargs = {}

    with pytest.raises(ValueError):
        _ = sdata_blobs.apply(
            func=func,
            fn_kwargs=fn_kwargs,
            img_layer="blobs_image",
            output_layer="blobs_apply",
            combine_c=True,
            combine_z=True,
            overwrite=True,
            chunks=212,
            scale_factors=None,
        )

    sdata_blobs = sdata_blobs.apply(
        func=func,
        fn_kwargs=fn_kwargs,
        img_layer="blobs_image",
        output_layer="blobs_apply",
        combine_c=False,
        combine_z=True,
        overwrite=True,
        chunks=212,
        scale_factors=None,
    )

    for c in func.keys():
        res = sdata_blobs.images["blobs_image"].sel(c=c).compute()
        res2 = sdata_blobs.images["blobs_apply"].sel(c=c).compute()
        assert np.array_equal(func[c](res), res2)

    with pytest.raises(KeyError):
        sdata_blobs.images["blobs_apply"].sel(c=1)


def test_apply_multiple_functions_multiple_fn_kwargs(sdata_blobs: SpatialData):
    func = {0: _multiply, 2: _add}
    fn_kwargs = {0: {"parameter": 4}, 2: {"parameter_add": 4}}

    with pytest.raises(ValueError):
        _ = sdata_blobs.apply(
            func=func,
            fn_kwargs=fn_kwargs,
            img_layer="blobs_image",
            output_layer="blobs_apply",
            combine_c=True,
            combine_z=True,
            overwrite=True,
            chunks=212,
            scale_factors=None,
        )

    sdata_blobs = sdata_blobs.apply(
        func=func,
        fn_kwargs=fn_kwargs,
        img_layer="blobs_image",
        output_layer="blobs_apply",
        combine_c=False,
        combine_z=True,
        overwrite=True,
        chunks=212,
        scale_factors=None,
    )

    for c in func.keys():
        res = sdata_blobs.images["blobs_image"].sel(c=c).compute()
        res2 = sdata_blobs.images["blobs_apply"].sel(c=c).compute()
        assert np.array_equal(func[c](res, **fn_kwargs[c]), res2)

    with pytest.raises(KeyError):
        sdata_blobs.images["blobs_apply"].sel(c=1)

    # AssertionErrro if keys in fn_kwargs do not match keys in func
    fn_kwargs = {0: {"parameter": 4}, 1: {"parameter_add": 4}}
    with pytest.raises(AssertionError):
        _ = sdata_blobs.apply(
            func=func,
            fn_kwargs=fn_kwargs,
            img_layer="blobs_image",
            output_layer="blobs_apply",
            combine_c=False,
            combine_z=True,
            overwrite=True,
            chunks=212,
            scale_factors=None,
        )


# test multiscale + test transformations


def test_precondition():
    fn_kwargs = {"parameter": 4}
    func = _multiply

    fn_kwargs_post, func_post = _precondition(
        fn_kwargs=fn_kwargs,
        func=func,
        combine_c=True,
        combine_z=True,
        channels=[0, 1],
        z_slices=[0.5, 1.5],
    )

    assert fn_kwargs_post == fn_kwargs
    assert func_post == func

    fn_kwargs_post, func_post = _precondition(
        fn_kwargs=fn_kwargs,
        func=func,
        combine_c=False,
        combine_z=True,
        channels=[0, 1],
        z_slices=[0.5, 1.5],
    )

    assert fn_kwargs_post == {0: {"parameter": 4}, 1: {"parameter": 4}}
    assert func_post == {0: _multiply, 1: _multiply}

    fn_kwargs_post, func_post = _precondition(
        fn_kwargs=fn_kwargs,
        func=func,
        combine_c=True,
        combine_z=False,
        channels=[0, 1],
        z_slices=[0.5, 1.5],
    )

    assert fn_kwargs_post == {0.5: {"parameter": 4}, 1.5: {"parameter": 4}}
    assert func_post == {0.5: _multiply, 1.5: _multiply}

    fn_kwargs_post, func_post = _precondition(
        fn_kwargs=fn_kwargs,
        func=func,
        combine_c=False,
        combine_z=False,
        channels=[0, 1],
        z_slices=[0.5, 1.5],
    )

    assert fn_kwargs_post == {
        0: {0.5: {"parameter": 4}, 1.5: {"parameter": 4}},
        1: {0.5: {"parameter": 4}, 1.5: {"parameter": 4}},
    }
    assert func_post == {
        0: {0.5: _multiply, 1.5: _multiply},
        1: {0.5: _multiply, 1.5: _multiply},
    }

    fn_kwargs_post, func_post = _precondition(
        fn_kwargs=fn_kwargs,
        func=func,
        combine_c=False,
        combine_z=False,
        channels=[0],
        z_slices=[0.5, 1.5],
    )

    assert fn_kwargs_post == {
        0: {0.5: {"parameter": 4}, 1.5: {"parameter": 4}},
    }
    assert func_post == {
        0: {0.5: _multiply, 1.5: _multiply},
    }

    fn_kwargs_post, func_post = _precondition(
        fn_kwargs=fn_kwargs,
        func=func,
        combine_c=False,
        combine_z=False,
        channels=[0, 1],
        z_slices=[0.5],
    )

    assert fn_kwargs_post == {
        0: {
            0.5: {"parameter": 4},
        },
        1: {
            0.5: {"parameter": 4},
        },
    }
    assert func_post == {
        0: {
            0.5: _multiply,
        },
        1: {
            0.5: _multiply,
        },
    }


def test_precondition_multiple_func():
    fn_kwargs = {0: {"parameter": 4}, 1: {"parameter_add": 10}}
    func = {0: _multiply, 1: _add}

    with pytest.raises(ValueError):
        _, _ = _precondition(
            fn_kwargs=fn_kwargs,
            func=func,
            combine_c=True,
            combine_z=True,
            channels=[0, 1],
            z_slices=[0.5, 1.5],
        )

    fn_kwargs_post, func_post = _precondition(
        fn_kwargs=fn_kwargs,
        func=func,
        combine_c=False,
        combine_z=True,
        channels=[0, 1],
        z_slices=[0.5, 1.5],
    )

    assert fn_kwargs_post == {0: {"parameter": 4}, 1: {"parameter_add": 10}}
    assert func_post == {0: _multiply, 1: _add}

    fn_kwargs_post, func_post = _precondition(
        fn_kwargs=fn_kwargs,
        func=func,
        combine_c=False,
        combine_z=False,
        channels=[0, 1],
        z_slices=[0.5, 1.5],
    )

    assert fn_kwargs_post == {
        0: {0.5: {"parameter": 4}, 1.5: {"parameter": 4}},
        1: {0.5: {"parameter_add": 10}, 1.5: {"parameter_add": 10}},
    }
    assert func_post == {0: {0.5: _multiply, 1.5: _multiply}, 1: {0.5: _add, 1.5: _add}}

    fn_kwargs = {0.5: {"parameter": 4}, 1.5: {"parameter_add": 10}}
    func = {0.5: _multiply, 1.5: _add}

    fn_kwargs_post, func_post = _precondition(
        fn_kwargs=fn_kwargs,
        func=func,
        combine_c=False,
        combine_z=False,
        channels=[0, 1],
        z_slices=[0.5, 1.5],
    )

    assert fn_kwargs_post == {
        0: {0.5: {"parameter": 4}, 1.5: {"parameter_add": 10}},
        1: {0.5: {"parameter": 4}, 1.5: {"parameter_add": 10}},
    }
    assert func_post == {0: {0.5: _multiply, 1.5: _add}, 1: {0.5: _multiply, 1.5: _add}}

    # if conflicts in parameters specified in fn_kwargs and channels,
    # then let fn_kwargs decide.
    fn_kwargs = {0.5: {"parameter": 4}, 1.5: {"parameter_add": 10}}
    func = {0.5: _multiply, 1.5: _add}

    fn_kwargs_post, func_post = _precondition(
        fn_kwargs=fn_kwargs,
        func=func,
        combine_c=False,
        combine_z=False,
        channels=[0, 1],
        z_slices=[0.5],
    )

    assert fn_kwargs_post == {
        0: {0.5: {"parameter": 4}, 1.5: {"parameter_add": 10}},
        1: {0.5: {"parameter": 4}, 1.5: {"parameter_add": 10}},
    }
    assert func_post == {0: {0.5: _multiply, 1.5: _add}, 1: {0.5: _multiply, 1.5: _add}}

    # fn_kwargs and func should match with keys, if func is a mapping
    fn_kwargs = {0.5: {"parameter": 4}, 1.5: {"parameter_add": 10}}
    func = {0.5: _multiply}

    with pytest.raises(AssertionError):
        fn_kwargs_post, func_post = _precondition(
            fn_kwargs=fn_kwargs,
            func=func,
            combine_c=False,
            combine_z=False,
            channels=[0, 1],
            z_slices=[0.5],
        )

        fn_kwargs_post, func_post


def test_precondition_empty_fn_kwargs():
    fn_kwargs = {}
    func = _multiply

    fn_kwargs_post, func_post = _precondition(
        fn_kwargs=fn_kwargs,
        func=func,
        combine_c=False,
        combine_z=False,
        channels=[0, 1],
        z_slices=[0.5],
    )

    assert fn_kwargs_post == {0: {0.5: {}}, 1: {0.5: {}}}
    assert func_post == {0: {0.5: _multiply}, 1: {0.5: _multiply}}

    fn_kwargs = {}
    func = {0.5: _multiply, 1.5: _add}

    fn_kwargs_post, func_post = _precondition(
        fn_kwargs=fn_kwargs,
        func=func,
        combine_c=False,
        combine_z=False,
        channels=[0, 1],
        z_slices=[0.5],
    )

    assert fn_kwargs_post == {0: {0.5: {}, 1.5: {}}, 1: {0.5: {}, 1.5: {}}}
    assert func_post == {0: {0.5: _multiply, 1.5: _add}, 1: {0.5: _multiply, 1.5: _add}}
