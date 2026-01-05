from __future__ import annotations

from typing import TYPE_CHECKING, Any

import spatialdata

if TYPE_CHECKING:
    from spatialdata.dataloader.datasets import ImageTilesDataset

__all__ = [
    "ImageTilesDataset",
]


def __getattr__(attr_name: str) -> ImageTilesDataset | Any:
    if attr_name == "ImageTilesDataset":
        from spatialdata.dataloader.datasets import ImageTilesDataset

        return ImageTilesDataset

    return getattr(spatialdata.dataloader, attr_name)
