import dask
import numpy as np
import pytest
from multiscale_spatial_image.to_multiscale.to_multiscale import Methods

from spatialdata.datasets import BlobsDataset
from spatialdata.models import Image2DModel, Image3DModel, Labels2DModel, Labels3DModel

CHUNK_SIZE = 32


@pytest.mark.parametrize(
    ("model", "length", "ndim", "n_channels", "scale_factors", "method"),
    [
        (Image2DModel, 128, 2, 3, (2, 2), Methods.XARRAY_COARSEN),
        (Image3DModel, 32, 3, 3, (2, 2), Methods.XARRAY_COARSEN),
        (Labels2DModel, 128, 2, 0, (2, 2), Methods.DASK_IMAGE_NEAREST),
        (Labels3DModel, 32, 3, 0, (2, 2), Methods.DASK_IMAGE_NEAREST),
    ],
)
def test_to_multiscale_via_ome_zarr_scaler(model, length, ndim, n_channels, scale_factors, method):
    blob_gen = BlobsDataset()

    if n_channels > 0:
        # Image: stack multiple blob channels
        masks = []
        for i in range(n_channels):
            mask = blob_gen._generate_blobs(length=length, seed=i, ndim=ndim)
            mask = (mask - mask.min()) / np.ptp(mask)
            masks.append(mask)
        array = np.stack(masks, axis=0)
    else:
        # Labels: threshold blob pattern to get integer labels
        mask = blob_gen._generate_blobs(length=length, ndim=ndim)
        threshold = np.percentile(mask, 70)
        array = (mask >= threshold).astype(np.int64)

    dims = model.dims
    dask_data = dask.array.from_array(array).rechunk(CHUNK_SIZE)

    # multiscale-spatial-image path (explicit method)
    result_msi = model.parse(dask_data, dims=dims, scale_factors=scale_factors, chunks=CHUNK_SIZE, method=method)

    # ome-zarr-py scaler path (method=None triggers the ome-zarr-py scaler)
    result_ozp = model.parse(dask_data, dims=dims, scale_factors=scale_factors, chunks=CHUNK_SIZE)

    # Compare data values at each scale level
    for i, scale_name in enumerate(result_msi.children):
        msi_arr = result_msi[scale_name].ds["image"]
        ozp_arr = result_ozp[scale_name].ds["image"]
        assert msi_arr.sizes == ozp_arr.sizes
        if model in [Image2DModel, Image3DModel]:
            # exact comparison for images
            np.testing.assert_allclose(msi_arr.values, ozp_arr.values)
        else:
            if i == 0:
                # no downscaling is performed, so they must be equal
                np.testing.assert_array_equal(msi_arr.values, ozp_arr.values)
            else:
                # we expect differences: ngff-zarr uses "nearest", ozp uses "resize"
                # TODO: when https://github.com/ome/ome-zarr-py/pull/516 is merged we can use nearest for labels and
                # expect a much stricter adherence
                fraction_non_equal = np.sum(msi_arr.values != ozp_arr.values) / np.prod(msi_arr.values.shape)
                assert fraction_non_equal < 0.5
