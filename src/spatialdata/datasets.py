"""SpatialData datasets."""
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
import scipy
from dask.dataframe.core import DataFrame as DaskDataFrame
from geopandas import GeoDataFrame
from multiscale_spatial_image import MultiscaleSpatialImage
from numpy.random import default_rng
from shapely.geometry import MultiPolygon, Point, Polygon
from skimage.segmentation import slic
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
from spatialdata.transformations import Identity

__all__ = ["blobs", "raccoon"]


def blobs(
    length: int = 512,
    n_points: int = 200,
    n_shapes: int = 5,
    extra_coord_system: Optional[str] = None,
    n_channels: int = 3,
) -> SpatialData:
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
    extra_coord_system
        Extra coordinate space on top of the standard global coordinate space. Will have only identity transform.
    n_channels
        Number of channels of the image


    Returns
    -------
    SpatialData
        SpatialData object with blobs dataset.
    """
    return BlobsDataset(
        length=length,
        n_points=n_points,
        n_shapes=n_shapes,
        extra_coord_system=extra_coord_system,
        n_channels=n_channels,
    ).blobs()


def raccoon() -> SpatialData:
    """Raccoon dataset."""
    return RaccoonDataset().raccoon()


class RaccoonDataset:
    """Raccoon dataset."""

    def __init__(self) -> None:
        """Raccoon dataset."""

    def raccoon(
        self,
    ) -> SpatialData:
        """Raccoon dataset."""
        im_data = scipy.misc.face()
        im = Image2DModel.parse(im_data, dims=["y", "x", "c"])
        labels_data = slic(im_data, n_segments=100, compactness=10, sigma=1)
        labels = Labels2DModel.parse(labels_data, dims=["y", "x"])
        coords = np.array([[610, 450], [730, 325], [575, 300], [480, 90]])
        circles = ShapesModel.parse(coords, geometry=0, radius=np.array([30, 30, 30, 50]))
        return SpatialData(images={"raccoon": im}, labels={"segmentation": labels}, shapes={"circles": circles})


class BlobsDataset:
    """Blobs dataset."""

    def __init__(
        self,
        length: int = 512,
        n_points: int = 200,
        n_shapes: int = 5,
        extra_coord_system: Optional[str] = None,
        n_channels: int = 3,
    ) -> None:
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
        extra_coord_system
            Extra coordinate space on top of the standard global coordinate space. Will have only identity transform.
        n_channels
            Number of channels of the image
        """
        self.length = length
        self.n_points = n_points
        self.n_shapes = n_shapes
        self.transformations = {"global": Identity()}
        self.n_channels = n_channels
        if extra_coord_system:
            self.transformations[extra_coord_system] = Identity()

    def blobs(
        self,
    ) -> SpatialData:
        """Blobs dataset."""
        image = self._image_blobs(self.transformations, self.length, self.n_channels)
        multiscale_image = self._image_blobs(self.transformations, self.length, self.n_channels, multiscale=True)
        labels = self._labels_blobs(self.transformations, self.length)
        multiscale_labels = self._labels_blobs(self.transformations, self.length, multiscale=True)
        points = self._points_blobs(self.transformations, self.length, self.n_points)
        circles = self._circles_blobs(self.transformations, self.length, self.n_shapes)
        polygons = self._polygons_blobs(self.transformations, self.length, self.n_shapes)
        multipolygons = self._polygons_blobs(self.transformations, self.length, self.n_shapes, multipolygons=True)
        adata = aggregate(values=image, by=labels).table
        adata.obs["region"] = pd.Categorical(["blobs_labels"] * len(adata))
        adata.obs["instance_id"] = adata.obs_names.astype(int)
        del adata.uns[TableModel.ATTRS_KEY]
        table = TableModel.parse(adata, region="blobs_labels", region_key="region", instance_key="instance_id")

        return SpatialData(
            images={"blobs_image": image, "blobs_multiscale_image": multiscale_image},
            labels={"blobs_labels": labels, "blobs_multiscale_labels": multiscale_labels},
            points={"blobs_points": points},
            shapes={"blobs_circles": circles, "blobs_polygons": polygons, "blobs_multipolygons": multipolygons},
            table=table,
        )

    def _image_blobs(
        self,
        transformations: Optional[dict[str, Any]] = None,
        length: int = 512,
        n_channels: int = 3,
        multiscale: bool = False,
    ) -> Union[SpatialImage, MultiscaleSpatialImage]:
        masks = []
        for i in range(n_channels):
            mask = self._generate_blobs(length=length, seed=i)
            mask = (mask - mask.min()) / mask.ptp()
            masks.append(mask)

        x = np.stack(masks, axis=0)
        dims = ["c", "y", "x"]
        if not multiscale:
            return Image2DModel.parse(x, transformations=transformations, dims=dims)
        return Image2DModel.parse(x, transformations=transformations, dims=dims, scale_factors=[2, 2])

    def _labels_blobs(
        self, transformations: Optional[dict[str, Any]] = None, length: int = 512, multiscale: bool = False
    ) -> Union[SpatialImage, MultiscaleSpatialImage]:
        """Create a 2D labels."""
        from scipy.ndimage import watershed_ift

        # from skimage
        mask = self._generate_blobs(length=length)
        threshold = np.percentile(mask, 100 * (1 - 0.3))
        inputs = np.logical_not(mask < threshold).astype(np.uint8)
        # use watershed from scipy
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
        dims = ["y", "x"]
        if not multiscale:
            return Labels2DModel.parse(out, transformations=transformations, dims=dims)
        return Labels2DModel.parse(out, transformations=transformations, dims=dims, scale_factors=[2, 2])

    def _generate_blobs(self, length: int = 512, seed: Optional[int] = None) -> ArrayLike:
        from scipy.ndimage import gaussian_filter

        rng = default_rng(42) if seed is None else default_rng(seed)
        # from skimage
        shape = tuple([length] * 2)
        mask = np.zeros(shape)
        n_pts = max(int(1.0 / 0.1) ** 2, 1)
        points = (length * rng.random((2, n_pts))).astype(int)
        mask[tuple(indices for indices in points)] = 1
        mask = gaussian_filter(mask, sigma=0.25 * length * 0.1)
        return mask

    def _points_blobs(
        self, transformations: Optional[dict[str, Any]] = None, length: int = 512, n_points: int = 200
    ) -> DaskDataFrame:
        rng = default_rng(42)
        arr = rng.integers(10, length - 10, size=(n_points, 2)).astype(np.int64)
        # randomly assign some values from v to the points
        points_assignment0 = rng.integers(0, 10, size=arr.shape[0]).astype(np.int64)
        genes = rng.choice(["a", "b"], size=arr.shape[0])
        annotation = pd.DataFrame(
            {
                "genes": genes,
                "instance_id": points_assignment0,
            },
        )
        return PointsModel.parse(
            arr, transformations=transformations, annotation=annotation, feature_key="genes", instance_key="instance_id"
        )

    def _circles_blobs(
        self, transformations: Optional[dict[str, Any]] = None, length: int = 512, n_shapes: int = 5
    ) -> GeoDataFrame:
        midpoint = length // 2
        halfmidpoint = midpoint // 2
        radius = length // 10
        circles = GeoDataFrame(
            {
                "geometry": self._generate_random_points(n_shapes, (midpoint - halfmidpoint, midpoint + halfmidpoint)),
                "radius": radius,
            }
        )
        return ShapesModel.parse(circles, transformations=transformations)

    def _polygons_blobs(
        self,
        transformations: Optional[dict[str, Any]] = None,
        length: int = 512,
        n_shapes: int = 5,
        multipolygons: bool = False,
    ) -> GeoDataFrame:
        midpoint = length // 2
        halfmidpoint = midpoint // 2
        poly = GeoDataFrame(
            {
                "geometry": self._generate_random_polygons(
                    n_shapes, (midpoint - halfmidpoint, midpoint + halfmidpoint), multipolygons=multipolygons
                )
            }
        )
        return ShapesModel.parse(poly, transformations=transformations)

    # function that generates random shapely polygons given a bounding box
    def _generate_random_polygons(
        self, n: int, bbox: tuple[int, int], multipolygons: bool = False
    ) -> list[Union[Polygon, MultiPolygon]]:
        def get_poly() -> Polygon:
            return Polygon(
                [
                    (x + default_rng(i + 1).uniform(0, maxx // 4), y + default_rng(i).uniform(0, maxy // 4)),
                    (x + default_rng(i + 2).uniform(0, maxx // 4), y),
                    (x, y + default_rng(i + 3).uniform(0, maxy // 4)),
                ]
            )

        minx = miny = bbox[0]
        maxx = maxy = bbox[1]
        polygons: list[Polygon] = []
        for i in range(n):
            # generate random points
            rng1 = default_rng(i)
            x = rng1.uniform(minx, maxx)
            y = rng1.uniform(miny, maxy)
            # generate random polygon
            poly = get_poly()
            # check if the polygon overlaps with any of the existing polygons
            if not any(poly.overlaps(p) for p in polygons):
                polygons.append(poly)
            if multipolygons:
                # add a second polygon even if it overlaps
                poly2 = get_poly()
                last = polygons.pop()
                polygons.append(MultiPolygon([last, poly2]))
        return polygons

    # function that generates random shapely points given a bounding box
    def _generate_random_points(self, n: int, bbox: tuple[int, int]) -> list[Point]:
        minx = miny = bbox[0]
        maxx = maxy = bbox[1]
        points: list[Point] = []
        for i in range(n):
            # generate random points
            rng1 = default_rng(i)
            x = rng1.uniform(minx, maxx)
            y = rng1.uniform(miny, maxy)
            # generate random polygon
            point = Point(x, y)
            points.append(point)
        return points
