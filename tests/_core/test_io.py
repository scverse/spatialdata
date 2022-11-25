import numpy as np
import anndata as ad
import xarray as xr
from spatialdata import SpatialData


def test_io_points():
    n = 1000
    coords = np.random.rand(n, 2)
    adata = ad.AnnData(shape=(n, 0), obsm={"spatial": coords})
    sdata = SpatialData(points={"points": adata})
    sdata.write("~/temp/test.zarr")
