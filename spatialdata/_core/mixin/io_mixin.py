from abc import abstractmethod
from typing import Any, Optional


class IoMixin:
    @classmethod
    @abstractmethod
    def from_zarr(cls, path: str) -> Any:
        """Load from Zarr file."""
        pass

    @abstractmethod
    def to_zarr(self, to_path: str):
        """Save to Zarr file."""
        pass
