import json
import shutil

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
data_dir = Path("spatialdata-sandbox/merfish_mouse_brain/processed/")
assert data_dir.exists()

output_dir = data_dir.parent / "spatialdata.zarr"
#
# cells = sc.read_h5ad(data_dir / "cells.h5ad")
# img = np.asarray(imageio.imread(data_dir / "image.png"))
# single_molecule = sc.read_h5ad(data_dir / "single_molecule.h5ad")
# j = json.load(open(data_dir / "image_transform.json", "r"))
# image_translation = np.array([j["translation_x"], j["translation_y"]])
# image_scale_factors = np.array([j["scale_factor_x"], j["scale_factor_y"]])
#
# expression = cells.copy()
# del expression.obsm["region_radius"]
# del expression.obsm["spatial"]
#
# regions = cells.copy()
# del regions.X
#
# transform = sd.Transform(translation=image_translation, scale_factors=image_scale_factors)
#
# sdata = sd.SpatialData(
#     adata=expression,
#     regions={"cells": regions},
#     images={"rasterized": img},
#     images_transform={"rasterized": transform},
#     points=single_molecule,
# )
# print(sdata)
#
# # shutil.rmtree(output_dir, ignore_errors=True)
# # sdata.to_zarr(output_dir)
# #
# #
# # from spatialdata.temp.napari_viewer import view_with_napari
# #
# # view_with_napari(output_dir)

# sdata = sd.SpatialData.from_zarr(output_dir)
sdata = sd.SpatialData()
print(sdata)
