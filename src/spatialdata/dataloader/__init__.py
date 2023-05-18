import contextlib

with contextlib.suppress(ImportError):
    from spatialdata.dataloader.datasets import ImageTilesDataset

__all__ = ["ImageTilesDataset"]
