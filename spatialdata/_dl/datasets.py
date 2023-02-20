from typing import Callable, Optional

from torch.utils.data import Dataset

from spatialdata import SpatialData
from spatialdata._core._spatial_query import BoundingBoxRequest
from spatialdata._types import ArrayLike


class SpotCropDataset(Dataset):
    def __init__(
        self,
        sdata: SpatialData,
        coordinates: ArrayLike,
        transform: Optional[Callable[[SpatialData], SpatialData]] = None,
    ):
        self.sdata = sdata
        self.transform = transform

        self.coordinates = coordinates
        self.radius = 5

    def _get_bounding_box_coordinates(self, spot_index: int) -> tuple[ArrayLike, ArrayLike]:
        """Get the coordinates for the corners of the bounding box of that encompasses a given spot.

        Parameters
        ----------
        spot_index
            The row index of the spot.

        Returns
        -------
        min_coordinate
            The minimum coordinate of the bounding box.
        max_coordinate
            The maximum coordinate of the bounding box.
        """
        centroid = self.coordinates[spot_index]
        min_coordinate = centroid - self.radius
        max_coordinate = centroid + self.radius

        return min_coordinate, max_coordinate

    def __len___(self) -> int:
        return len(self.coordiantes)

    def _getitem__(self, idx: int) -> SpatialData:
        min_coordinate, max_coordinate = self._get_bounding_box_coordinates(spot_index=idx)

        request = BoundingBoxRequest(min_coordinate=min_coordinate, max_coordinate=max_coordinate, axes=("y", "x"))
        sdata_item = self.sdata.query.bounding_box(request=request)

        if self.transform is not None:
            sdata_item = self.transform(sdata_item)

        return sdata_item
