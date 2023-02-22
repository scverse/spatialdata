from typing import Callable, Optional

import numpy as np
from torch.utils.data import Dataset

from spatialdata import SpatialData
from spatialdata._core._spatial_query import BoundingBoxRequest
from spatialdata._core.core_utils import get_dims
from spatialdata._types import ArrayLike


class SpotCropDataset(Dataset):
    def __init__(
        self,
        sdata: SpatialData,
        spots_element_keys: list[str],
        transform: Optional[Callable[[SpatialData], SpatialData]] = None,
    ):
        self.sdata = sdata
        self.spots_element_keys = spots_element_keys
        self.transform = transform

        self.min_coordinates, self.max_coordinates, self.spots_dims = self._get_bounding_box_coordinates()
        self.n_spots = len(self.min_coordinates)

    def _get_centroids_and_metadata(self) -> None:
        for key in self.spots_element_keys:
            print(key)

    def _get_bounding_box_coordinates(self) -> tuple[ArrayLike, ArrayLike, tuple[str, ...]]:
        """Get the coordinates for the corners of the bounding box of that encompasses a given spot.

        Returns
        -------
        min_coordinate
            The minimum coordinate of the bounding box.
        max_coordinate
            The maximum coordinate of the bounding box.
        """
        spots_element = self.sdata.shapes[self.spots_element_keys[0]]
        spots_dims = get_dims(spots_element)

        centroids = []
        for dim_name in spots_dims:
            centroids.append(getattr(spots_element["geometry"], dim_name).to_numpy())
        centroids_array = np.column_stack(centroids)
        radius = np.expand_dims(spots_element["radius"].to_numpy(), axis=1)

        min_coordinates = (centroids_array - radius).astype(int)
        max_coordinates = (centroids_array + radius).astype(int)

        return min_coordinates, max_coordinates, spots_dims

    def __len__(self) -> int:
        return self.n_spots

    def __getitem__(self, idx: int) -> SpatialData:
        min_coordinate = self.min_coordinates[idx]
        max_coordinate = self.max_coordinates[idx]

        request = BoundingBoxRequest(min_coordinate=min_coordinate, max_coordinate=max_coordinate, axes=self.spots_dims)
        sdata_item = self.sdata.query.bounding_box(request=request)

        if self.transform is not None:
            sdata_item = self.transform(sdata_item)

        return sdata_item
