try:
    from spatialdata.dataloader.datasets import ImageTilesDataset
except ImportError:
    ImageTilesDataset = None  # type: ignore[assignment, misc]
