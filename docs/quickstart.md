# Quickstart

A {class}`~spatialdata.SpatialData` object holds the images, labels, points, shapes and tables of a dataset, aligned through shared coordinate systems.
The example below loads a small synthetic dataset, crops it, counts points per shape, and writes it to and reads it back from Zarr.

```python
import spatialdata as sd
from spatialdata.datasets import blobs

sdata = blobs()

crop = sd.bounding_box_query(
    sdata,
    axes=("x", "y"),
    min_coordinate=[0, 0],
    max_coordinate=[200, 200],
    target_coordinate_system="global",
)

counts = sd.aggregate(
    values=sdata["blobs_points"],
    by=sdata["blobs_circles"],
    value_key="genes",
    agg_func="count",
    target_coordinate_system="global",
)

sdata.write("blobs.zarr")
sdata = sd.read_zarr("blobs.zarr")
```

To load data from a specific technology, see [spatialdata-io](https://spatialdata.scverse.org/projects/io/en/stable/), and to plot it, [spatialdata-plot](https://spatialdata.scverse.org/projects/plot/en/stable/).
The {doc}`user_guide` points you to the tutorial for your question, and {doc}`api` lists every function.
