from pathlib import Path

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

# view_with_napari(output_dir)
#
# ----- loading -----
# sdata = sd.SpatialData.read(output_dir)
# ##
# with tempfile.TemporaryDirectory() as td:
#     td = Path(td)
#     td / "data.zarr"
#     sdata.to_zarr(td)
#     view_with_napari(td)
##
# print(sdata)
