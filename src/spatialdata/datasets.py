"""SpatialData datasets."""
from typing import Any

import numpy as np
import pandas as pd
from dask.dataframe.core import DataFrame as DaskDataFrame
from geopandas import GeoDataFrame
from numpy.random import default_rng
from shapely.geometry import Polygon
from spatial_image import SpatialImage

from spatialdata import SpatialData
from spatialdata._core.operations.aggregate import aggregate
from spatialdata._types import ArrayLike
from spatialdata.models import (
    Image2DModel,
    Labels2DModel,
    PointsModel,
    ShapesModel,
    TableModel,
)

rng = default_rng(42)

__all__ = ["blobs"]


def blobs(length: int = 512, n_points: int = 200, n_shapes: int = 5) -> SpatialData:
    """
    Blobs dataset.

    Parameters
    ----------
    length
        Length of the image/labels.
    n_points
        Number of points to generate.
    n_shapes
        Number of max shapes to generate.
        At most, as if overlapping they will be discarded

    Returns
    -------
    SpatialData
        SpatialData object with blobs dataset.
    """
    image = _image_blobs(length)
    labels = _blobs(length)
    points = _points_blobs(length, n_points)
    shapes = _shapes_blobs(length, n_shapes)

    adata = aggregate(image, labels)
    adata.obs["region"] = "blobs_labels"
    adata.obs["instance_id"] = adata.obs_names
    table = TableModel.parse(adata, region="blobs_labels", region_key="region", instance_key="instance_id")

    return SpatialData(
        images={"blobs_image": image},
        labels={"blobs_labels": labels},
        points={"blobs_points": points},
        shapes={"blobs_shapes": shapes},
        table=table,
    )


def _image_blobs(length: int = 512) -> SpatialImage:
    masks = []
    for i in range(3):
        rng = default_rng(i)
        mask = _generate_blobs(rng, length=length)
        mask = (mask - mask.min()) / mask.ptp()
        masks.append(mask)

    return Image2DModel.parse(np.stack(masks, axis=0))


def _blobs(length: int = 512) -> SpatialImage:
    """Create a 2D labels."""
    from scipy.ndimage import watershed_ift

    # from skimage
    mask = _generate_blobs(rng, length=length)
    threshold = np.percentile(mask, 100 * (1 - 0.3))
    inputs = np.logical_not(mask < threshold).astype(np.uint8)
    # use wastershed from scipy
    xm, ym = np.ogrid[0:length:10, 0:length:10]
    markers = np.zeros_like(inputs).astype(np.int16)
    markers[xm, ym] = np.arange(xm.size * ym.size).reshape((xm.size, ym.size))
    out = watershed_ift(inputs, markers)
    out[xm, ym] = out[xm - 1, ym - 1]  # remove the isolate seeds
    # reindex by frequency
    val, counts = np.unique(out, return_counts=True)
    sorted_idx = np.argsort(counts)
    for i, idx in enumerate(sorted_idx[::-1]):
        if (not i % 7) or (i == 0):
            out[out == val[idx]] = 0
        else:
            out[out == val[idx]] = i
    return Labels2DModel.parse(out)


def _generate_blobs(rng: Any, length: int = 512) -> ArrayLike:
    from scipy.ndimage import gaussian_filter

    # from skimage
    shape = tuple([length] * 2)
    mask = np.zeros(shape)
    n_pts = max(int(1.0 / 0.1) ** 2, 1)
    points = (length * rng.random((2, n_pts))).astype(int)
    mask[tuple(indices for indices in points)] = 1
    mask = gaussian_filter(mask, sigma=0.25 * length * 0.1)
    return mask


def _points_blobs(length: int = 512, n_points: int = 200) -> DaskDataFrame:
    arr = rng.integers(10, length - 10, size=(n_points, 2)).astype(np.int_)
    # randomly assign some values from v to the points
    points_assignment0 = rng.integers(0, 10, size=arr.shape[0]).astype(np.int_)
    genes = rng.choice(["a", "b"], size=arr.shape[0])
    annotation = pd.DataFrame(
        {
            "genes": genes,
            "instance_id": points_assignment0,
        },
    )
    return PointsModel.parse(arr, annotation=annotation, feature_key="genes", instance_key="instance_id")


def _shapes_blobs(length: int = 512, n_shapes: int = 5) -> GeoDataFrame:
    midpoint = length // 2
    halfmidpoint = midpoint // 2
    poly = GeoDataFrame(
        {"geometry": _generate_random_polygons(n_shapes, (midpoint - halfmidpoint, midpoint + halfmidpoint))}
    )
    return ShapesModel.parse(poly)


# function that generates random shapely polygons given a bounding box
def _generate_random_polygons(n: int, bbox: tuple[int, int]) -> list[Polygon]:
    minx = miny = bbox[0]
    maxx = maxy = bbox[1]
    polygons: list[Polygon] = []
    for i in range(n):
        # generate random points
        rng1 = default_rng(i)
        x = rng1.uniform(minx, maxx)
        y = rng1.uniform(miny, maxy)
        # generate random polygon
        poly = Polygon(
            [
                (x + default_rng(i + 1).uniform(0, maxx // 4), y + default_rng(i).uniform(0, maxy // 4)),
                (x + default_rng(i + 2).uniform(0, maxx // 4), y),
                (x, y + default_rng(i + 3).uniform(0, maxy // 4)),
            ]
        )
        # check if the polygon overlaps with any of the existing polygons
        if not any(poly.overlaps(p) for p in polygons):
            polygons.append(poly)
    return polygons
