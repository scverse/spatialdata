from typing import Union

try:
    from spatialdata.dataloader.datasets import ImageTilesDataset
except ImportError as e:
    _error: Union[str, None] = str(e)
else:
    _error = None

__all__ = ["ImageTilesDataset"]
