import itertools

import dask_image.ndinterp
import pytest
import xarray
from xarray import DataArray, DataTree

from spatialdata._utils import skip_non_dimension_nodes, unpad_raster
from spatialdata.models import get_model
from spatialdata.transformations import Affine


def _pad_raster(data: DataArray, axes: tuple[str, ...]) -> DataArray:
    new_shape = tuple([data.shape[i] * (2 if axes[i] != "c" else 1) for i in range(len(data.shape))])
    x = data.shape[axes.index("x")]
    y = data.shape[axes.index("y")]
    affine = Affine(
        [
            [1, 0, -x / 2.0],
            [0, 1, -y / 2.0],
            [0, 0, 1],
        ],
        input_axes=("x", "y"),
        output_axes=("x", "y"),
    )
    matrix = affine.to_affine_matrix(input_axes=axes, output_axes=axes)
    return dask_image.ndinterp.affine_transform(data, matrix, output_shape=new_shape)


@pytest.mark.ci_only
def test_unpad_raster(images, labels) -> None:
    for raster in itertools.chain(images.images.values(), labels.labels.values()):
        schema = get_model(raster)
        if isinstance(raster, DataArray):
            data = raster
        elif isinstance(raster, DataTree):
            d = dict(raster["scale0"])
            assert len(d) == 1
            data = d.values().__iter__().__next__()
        else:
            raise ValueError(f"Unknown type: {type(raster)}")
        padded = _pad_raster(data.data, data.dims)
        if isinstance(raster, DataArray):
            padded = schema.parse(padded, dims=data.dims, c_coords=data.coords.get("c", None))
        elif isinstance(raster, DataTree):
            # some arbitrary scaling factors
            padded = schema.parse(padded, dims=data.dims, scale_factors=[2, 2], c_coords=data.coords.get("c", None))
        else:
            raise ValueError(f"Unknown type: {type(raster)}")
        unpadded = unpad_raster(padded)
        if isinstance(raster, DataArray):
            try:
                xarray.testing.assert_equal(raster, unpadded)
            except AssertionError as e:
                raise e
        elif isinstance(raster, DataTree):
            d0 = dict(raster["scale0"])
            assert len(d0) == 1
            d1 = dict(unpadded["scale0"])
            assert len(d1) == 1
            try:
                xarray.testing.assert_equal(d0.values().__iter__().__next__(), d1.values().__iter__().__next__())
            except AssertionError as e:
                raise e
        else:
            raise ValueError(f"Unknown type: {type(raster)}")


def test_skip_nodes(images):
    multiscale_img = images["image2d_multiscale"]

    @skip_non_dimension_nodes
    def transpose(ds, *args, **kwargs):
        return ds.transpose(*args, **kwargs)

    for scale in list(multiscale_img.keys()):
        assert multiscale_img[scale]["image"].dims == ("c", "y", "x")

    # applying this function without skipping the root node would fail as the root node does not have dimensions.
    result = images["image2d_multiscale"].map_over_datasets(transpose, "y", "x", "c")
    for scale in list(result.keys()):
        assert result[scale]["image"].dims == ("y", "x", "c")
