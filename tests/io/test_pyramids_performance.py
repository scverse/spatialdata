from pathlib import Path
from typing import TYPE_CHECKING

import dask
import dask.array
import numpy as np
import pytest
import xarray as xr
import zarr

from spatialdata import SpatialData
from spatialdata._io import write_image
from spatialdata._io.format import CurrentRasterFormat
from spatialdata.models import Image2DModel

if TYPE_CHECKING:
    import _pytest.fixtures


@pytest.fixture
def sdata_with_image(request: "_pytest.fixtures.SubRequest", tmp_path: Path) -> SpatialData:
    params = request.param if request.param is not None else {}
    width = params.get("width", 2048)
    chunksize = params.get("chunk_size", 1024)
    scale_factors = params.get("scale_factors", (2,))
    # Create a disk-backed Dask array for scale 0.
    npg = np.random.default_rng(0)
    array = npg.integers(low=0, high=2**16, size=(1, width, width))
    array_path = tmp_path / "image.zarr"
    dask.array.from_array(array).rechunk(chunksize).to_zarr(array_path)
    array_backed = dask.array.from_zarr(array_path)
    # Create an in-memory SpatialData with disk-backed scale 0.
    image = Image2DModel.parse(array_backed, dims=("c", "y", "x"), scale_factors=scale_factors, chunks=chunksize)
    return SpatialData(images={"image": image})


def count_chunks(array: xr.DataArray | xr.Dataset | xr.DataTree) -> int:
    if isinstance(array, xr.DataTree):
        array = array.ds
    # From `chunksizes`, we get only the number of chunks per axis.
    # By multiplying them, we get the total number of chunks in 2D/3D.
    return np.prod([len(chunk_sizes) for chunk_sizes in array.chunksizes.values()])


@pytest.mark.parametrize(
    ("sdata_with_image",),
    [
        ({"width": 32, "chunk_size": 16, "scale_factors": (2,)},),
        ({"width": 64, "chunk_size": 16, "scale_factors": (2, 2)},),
        ({"width": 128, "chunk_size": 16, "scale_factors": (2, 2, 2)},),
        ({"width": 256, "chunk_size": 16, "scale_factors": (2, 2, 2, 2)},),
    ],
    indirect=["sdata_with_image"],
)
def test_write_image_multiscale_performance(sdata_with_image: SpatialData, tmp_path: Path, mocker):
    # Writing multiscale images with several pyramid levels should be efficient.
    # Specifically, it should not read the input image more often than necessary
    # (see issue https://github.com/scverse/spatialdata/issues/577).
    # Instead of measuring the time (which would have high variation if not using big datasets),
    # we watch the number of read and write accesses and compare to the theoretical number.
    zarr_chunk_write_spy = mocker.spy(zarr.core.Array, "__setitem__")
    zarr_chunk_read_spy = mocker.spy(zarr.core.Array, "__getitem__")

    image_name, image = next(iter(sdata_with_image.images.items()))
    element_type_group = zarr.group(store=tmp_path / "sdata.zarr", path="/images")

    write_image(
        image=image,
        group=element_type_group,
        name=image_name,
        format=CurrentRasterFormat(),
    )

    # The number of chunks of scale level 0
    num_chunks_scale0 = count_chunks(image.scale0 if isinstance(image, xr.DataTree) else image)
    # The total number of chunks of all scale levels
    num_chunks_all_scales = (
        sum(count_chunks(pyramid) for pyramid in image.children.values())
        if isinstance(image, xr.DataTree)
        else count_chunks(image)
    )

    actual_num_chunk_writes = zarr_chunk_write_spy.call_count
    actual_num_chunk_reads = zarr_chunk_read_spy.call_count
    assert actual_num_chunk_writes == num_chunks_all_scales
    assert actual_num_chunk_reads == num_chunks_scale0
