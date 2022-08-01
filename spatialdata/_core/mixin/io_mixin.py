from abc import abstractmethod
from typing import Any


class IoMixin:
    @classmethod
    @abstractmethod
    def from_zarr(cls, path: str) -> Any:
        """Load from Zarr file."""

    @abstractmethod
    def to_zarr(self, to_path: str) -> None:
        """Save to Zarr file."""
