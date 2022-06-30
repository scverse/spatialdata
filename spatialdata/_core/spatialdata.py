from types import MappingProxyType
from typing import Any, Mapping, Optional

from anndata import AnnData


class SpatialData:
    """Spatial data structure."""

    adata: AnnData
    regions: Mapping[str, Any]
    images: Mapping[str, Any]
    points: Optional[AnnData]

    def __init__(
        self,
        adata: AnnData,
        regions: Mapping[str, Any] = MappingProxyType({}),
        images: Mapping[str, Any] = MappingProxyType({}),
        points: Optional[AnnData] = None,
    ) -> None:
        self.adata = adata
        self.regions = dict(regions)
        self.images = dict(images)
        self.points = points
