import numpy as np
import pytest

from spatialdata import SpatialData
from spatialdata._core.operations.apply import _precondition
from spatialdata.transformations import Translation, get_transformation, set_transformation


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

    # AssertionError if keys in fn_kwargs do not match keys in func
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


def test_apply_transformation(sdata_blobs: SpatialData):
    fn_kwargs = {"parameter": 4}

    # set a dummy translation
    translation = Translation([100, 100], axes=("x", "y"))
    set_transformation(sdata_blobs.images["blobs_image"], translation)

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

    tr1 = get_transformation(sdata_blobs.images["blobs_image"])
    tr2 = get_transformation(sdata_blobs.images["blobs_apply"])
    assert tr1 == tr2

    res = sdata_blobs.images["blobs_image"].compute()
    res2 = sdata_blobs.images["blobs_apply"].compute()
    assert np.array_equal(res * fn_kwargs["parameter"], res2)


def test_apply_fail(sdata_blobs: SpatialData):
    fn_kwargs = {0: {"parameter": 4}, 2: {"parameter": 8}, 3: {"parameter": 16}}

    # this should fail because some keys in fn_kwargs contain channels that
    # are not in sdata[ "blobs_image" ].c.data
    with pytest.raises(ValueError):
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


def test_apply_multiscale(sdata_blobs: SpatialData):
    fn_kwargs = {"parameter": 4}

    # TODO
    # apply triggers recomputation for every scale.
    # potential fix could be to write to intermediate zarr with results for scale0,
    # and for backed sdata, arr.persist() would prevent recomputation.
    sdata_blobs = sdata_blobs.apply(
        func=_multiply,
        fn_kwargs=fn_kwargs,
        img_layer="blobs_multiscale_image",
        output_layer="blobs_apply",
        combine_c=False,
        combine_z=True,
        overwrite=True,
        chunks=212,
        scale_factors=[2, 2],
    )

    for scale in sdata_blobs.images["blobs_apply"]:
        res = sdata_blobs.images["blobs_multiscale_image"][scale][
            sdata_blobs.images["blobs_multiscale_image"][scale].__iter__().__next__()
        ]
        res2 = sdata_blobs.images["blobs_apply"][scale][sdata_blobs.images["blobs_apply"][scale].__iter__().__next__()]

        assert np.array_equal(res.compute() * fn_kwargs["parameter"], res2.compute())


def test_apply_alter_c_dim(sdata_blobs: SpatialData):
    fn_kwargs = {}

    def _alter_c_dimension(image):
        return np.stack([image[0], image[0]])

    # TODO Do we want to support altering of c dimension when combine_c is True?
    with pytest.raises(ValueError):
        sdata_blobs = sdata_blobs.apply(
            func=_alter_c_dimension,
            fn_kwargs=fn_kwargs,
            img_layer="blobs_image",
            output_layer="blobs_apply",
            combine_c=True,
            combine_z=True,
            overwrite=True,
            chunks=256,
            scale_factors=None,
            output_chunks=((2,), (256,), (256,)),
        )

        res = sdata_blobs.images["blobs_apply"].compute()


def test_apply_alter_x_y_dim(sdata_blobs: SpatialData):
    fn_kwargs = {}

    def _alter_x_dimension(
        image,
    ):
        padding = ((0, 0), (5, 5), (5, 5))
        padded_array = np.pad(image, pad_width=padding, mode="constant", constant_values=0)
        return padded_array

    sdata_blobs = sdata_blobs.apply(
        func=_alter_x_dimension,
        fn_kwargs=fn_kwargs,
        img_layer="blobs_image",
        output_layer="blobs_apply",
        combine_c=True,
        combine_z=True,
        overwrite=True,
        chunks=212,
        scale_factors=None,
        output_chunks=((3,), (212 + 10, 44 + 10), (212 + 10, 44 + 10)),
    )

    res = sdata_blobs.images["blobs_apply"].compute()
    res


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

    # if conflicts in channels/z_slices specified in fn_kwargs/func and channels/z_slices,
    # then raise a ValueError to prevent unwanted behaviour.
    fn_kwargs = {0.5: {"parameter": 4}, 1.5: {"parameter_add": 10}}
    func = {0.5: _multiply, 1.5: _add}

    with pytest.raises(ValueError):
        fn_kwargs_post, func_post = _precondition(
            fn_kwargs=fn_kwargs,
            func=func,
            combine_c=False,
            combine_z=False,
            channels=[0, 1],
            z_slices=[0.5],
        )

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
        z_slices=[0.5, 1.5],
    )

    assert fn_kwargs_post == {0: {0.5: {}, 1.5: {}}, 1: {0.5: {}, 1.5: {}}}
    assert func_post == {0: {0.5: _multiply, 1.5: _add}, 1: {0.5: _multiply, 1.5: _add}}

    # if keys specified they should be in channels or z_slices, otherwise raise ValueError.
    with pytest.raises(ValueError):
        fn_kwargs_post, func_post = _precondition(
            fn_kwargs=fn_kwargs,
            func=func,
            combine_c=False,
            combine_z=False,
            channels=[0, 1],
            z_slices=[0.5],
        )
