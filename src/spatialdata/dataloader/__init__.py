from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from spatialdata.dataloader.datasets import ImageTilesDataset as _ImageTilesDataset


class ImageTilesDataset:  # noqa: D101
    _target_class: type[_ImageTilesDataset] | None = None

    def __new__(cls, *args: Any, **kwargs: Any) -> _ImageTilesDataset:  # noqa: D102
        if cls._target_class is None:
            try:
                from spatialdata.dataloader.datasets import (
                    ImageTilesDataset as ActualImageTilesDataset,
                )

                cls._target_class = ActualImageTilesDataset

            except ImportError as error:
                raise ImportError(
                    "ImageTilesDataset could not be imported. This usually means the 'torch' dependency is missing."
                ) from error

        return cls._target_class(*args, **kwargs)
