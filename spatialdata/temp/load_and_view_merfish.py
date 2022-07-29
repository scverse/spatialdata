import json
import shutil
import tempfile

import numpy as np
import scanpy as sc
import os
import imageio
from pathlib import Path
import spatialdata as sd
import anndata as ad

##
# empty = sd.SpatialData(ad.AnnData(np.random.rand(10, 10)))
# print(empty)

#
# # use symlinks to make the spatialdata-sandbox datasets repository available
data_dir = Path("spatialdata-sandbox/merfish")
assert data_dir.exists()

output_dir = data_dir / "data.zarr"



# shutil.rmtree(output_dir, ignore_errors=True)
# sdata.to_zarr(output_dir)
#
#
# ----- napari -----
from spatialdata.temp.napari_viewer import view_with_napari

# view_with_napari(output_dir)
#
# ----- loading -----
sdata = sd.SpatialData.from_zarr(output_dir)
##
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    td / 'data.zarr'
    sdata.to_zarr(td)
    view_with_napari(td)
##
# print(sdata)
