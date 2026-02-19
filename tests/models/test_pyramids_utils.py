from __future__ import annotations

import numpy as np
import pytest
from multiscale_spatial_image.to_multiscale.to_multiscale import Methods

from spatialdata.datasets import BlobsDataset
from spatialdata.models import Image2DModel, Image3DModel, Labels2DModel, Labels3DModel

CHUNK_SIZE = 32


@pytest.mark.parametrize(
    ("model", "length", "ndim", "n_channels", "scale_factors", "method"),
    [
        (Image2DModel, 128, 2, 3, [2, 3], Methods.XARRAY_COARSEN),
        (Image3DModel, 32, 3, 3, [{"x": 3, "y": 2, "z": 1}, {"x": 1, "y": 2, "z": 2}], Methods.XARRAY_COARSEN),
        (Labels2DModel, 128, 2, 0, [2, 2], Methods.DASK_IMAGE_NEAREST),
        (Labels3DModel, 32, 3, 0, [{"x": 2, "y": 2, "z": 3}, 2], Methods.DASK_IMAGE_NEAREST),
    ],
)
def test_to_multiscale_via_ome_zarr_scaler(model, length, ndim, n_channels, scale_factors, method):
    blob_gen = BlobsDataset()

    if model in [Image2DModel, Image3DModel]:
        array = blob_gen._image_blobs(length=length, n_channels=n_channels, ndim=ndim).data
    else:
        array = blob_gen._labels_blobs(length=length, ndim=ndim).data

    dims = model.dims
    dask_data = array.rechunk(CHUNK_SIZE)

    # multiscale-spatial-image (method is not None)
    result_msi = model.parse(dask_data, dims=dims, scale_factors=scale_factors, chunks=CHUNK_SIZE, method=method)

    # ome-zarr-py scaler (method=None triggers the ome-zarr-py scaler)
    result_ozp = model.parse(dask_data, dims=dims, scale_factors=scale_factors, chunks=CHUNK_SIZE)

    for i, scale_name in enumerate(result_msi.children):
        msi_arr = result_msi[scale_name].ds["image"]
        ozp_arr = result_ozp[scale_name].ds["image"]
        assert msi_arr.sizes == ozp_arr.sizes

        if i == 0:
            # scale0 is the original data, must be identical
            np.testing.assert_array_equal(msi_arr.values, ozp_arr.values)
        else:
            if model in (Labels2DModel, Labels3DModel):
                # labels use different nearest-like methods; expect <50% non-identical entries
                fraction_non_equal = np.sum(msi_arr.values != ozp_arr.values) / np.prod(msi_arr.values.shape)
                assert fraction_non_equal < 0.5, (
                    f"{scale_name}: {fraction_non_equal:.1%} non-identical entries (expected <50%)"
                )
            else:
                # images use fundamentally different algorithms (coarsen vs spline interpolation);
                # just check that the value ranges are similar
                msi_vals, ozp_vals = msi_arr.values, ozp_arr.values
                np.testing.assert_allclose(msi_vals.mean(), ozp_vals.mean(), rtol=0.5)
                np.testing.assert_allclose(msi_vals.std(), ozp_vals.std(), rtol=0.5)
