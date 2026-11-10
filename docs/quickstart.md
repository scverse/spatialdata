# Quickstart

A {class}`~spatialdata.SpatialData` object holds the images, labels, points, shapes and tables of a dataset, aligned through shared coordinate systems.
The example below builds one from scratch out of an image, a segmentation mask and a gene expression table, then crops it to a region of interest (ROI), measures image intensity per cell in the ROI, and writes the ROI to and reads it back from Zarr.

```python
import anndata as ad
import numpy as np
import pandas as pd

import spatialdata as sd
from spatialdata.models import Image2DModel, Labels2DModel, TableModel

rng = np.random.default_rng(0)

# A 3-channel image (c, y, x) ...
image = rng.random((3, 100, 100))

# ... a segmentation mask with three cells, labelled 1, 2, 3 (0 is background) ...
mask = np.zeros((100, 100), dtype=np.int32)
mask[10:30, 10:30] = 1
mask[40:70, 15:45] = 2
mask[30:60, 60:90] = 3

# ... and a cells x genes expression matrix, one row per cell in the mask.
expression = ad.AnnData(
    X=rng.poisson(5, size=(3, 4)).astype(float),
    obs=pd.DataFrame({"region": pd.Categorical(["cells"] * 3), "cell_id": [1, 2, 3]}, index=["a", "b", "c"]),
    var=pd.DataFrame(index=["GeneA", "GeneB", "GeneC", "GeneD"]),
)

sdata = sd.SpatialData(
    images={"image": Image2DModel.parse(image, dims=("c", "y", "x"))},
    labels={"cells": Labels2DModel.parse(mask, dims=("y", "x"))},
    tables={"table": TableModel.parse(expression, region="cells", region_key="region", instance_key="cell_id")},
)

# Crop all elements to a region of interest; the table keeps only the cells inside it.
sdata_ROI = sd.bounding_box_query(
    sdata,
    axes=("x", "y"),
    min_coordinate=[0, 0],
    max_coordinate=[50, 50],
    target_coordinate_system="global",
)

# Mean intensity of each image channel per cell in the ROI.
intensities = sd.aggregate(values=sdata_ROI["image"], by=sdata_ROI["cells"], agg_func="mean")

sdata_ROI.write("my_roi.zarr")
sdata_ROI = sd.read_zarr("my_roi.zarr")
```

To load data from a specific technology, see [spatialdata-io](https://spatialdata.scverse.org/projects/io/en/stable/), and to plot it, [spatialdata-plot](https://spatialdata.scverse.org/projects/plot/en/stable/).
The {doc}`user_guide` points you to the tutorial for your question, and {doc}`api` lists every function.
