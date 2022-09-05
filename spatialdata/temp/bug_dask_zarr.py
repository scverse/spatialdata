##
import os.path
import shutil

import dask.array.core
import numpy as np
import zarr
from ome_zarr.format import CurrentFormat
from ome_zarr.io import ZarrLocation, parse_url
from ome_zarr.reader import Multiscales, Reader
from ome_zarr.writer import write_image

from spatialdata.utils import are_directories_identical

im = np.random.normal(size=(3, 100, 100))
fmt = CurrentFormat()

##
# write


def write_to_zarr(im, f):
    if os.path.isdir(f):
        shutil.rmtree(f)
    store = parse_url(f, mode="w").store
    group = zarr.group(store=store)
    if isinstance(im, np.ndarray) or isinstance(im, dask.array.core.Array):
        # compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.NOSHUFFLE)
        write_image(im, group, axes=["c", "x", "y"], fmt=fmt, storage_options={"compressor": None})
        # write_image(im, group, axes=["c", "x", "y"], fmt=fmt, storage_options={"compressor": compressor}.copy())
    # elif isinstance(im, dask.array.core.Array):
    #     slow, we don't want to load data from disk to memory to write it back to disk
    # write_image(np.array(im), group, axes=["c", "x", "y"], fmt=fmt)
    else:
        raise ValueError("the array to write must be a numpy array or a dask array")


write_to_zarr(im, "debug0.zarr")

##
# read
loc = ZarrLocation("debug0.zarr")
reader = Reader(loc)()
nodes = list(reader)
assert len(nodes) == 1
node = nodes[0]
im_read = node.load(Multiscales).array(resolution="0", version=fmt.version)

##
# write again (error)
write_to_zarr(im_read, "debug1.zarr")

##

# assert are_directories_identical("debug0.zarr", "debug1.zarr", exclude_regexp="[1-9][0-9]*.*")
assert are_directories_identical("debug0.zarr", "debug1.zarr")

##
