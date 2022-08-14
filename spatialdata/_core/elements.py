from abc import ABC, abstractmethod, abstractproperty
from typing import Any, Optional, Tuple

import zarr
from anndata import AnnData
from ome_zarr.scale import Scaler

from spatialdata._core.transform import Transform
from spatialdata._io.write import (
    write_image,
    write_labels,
    write_points,
    write_polygons,
    write_table,
)
from spatialdata._types import ArrayLike

__all__ = ["Image", "Labels", "Points", "Polygons"]


class BaseElement(ABC):
    # store the actual data (e.g., array for image, coordinates for points, etc.)
    data: Any

    # the annotation table (if present)
    # TODO: do we want to store it here?
    table: Optional[AnnData] = None

    # store the transform objects as a dictionary with keys (source_coordinate_space, destination_coordinate_space)
    transforms: Optional[Transform] = None

    @abstractmethod
    def to_zarr(self, group: zarr.Group, name: str, scaler: Optional[Scaler] = None) -> None:
        """Write element to file."""

    @abstractmethod
    def transform(self, new_coordinate_space: str, inplace: bool = False) -> "BaseElement":
        """Transform the object to a new coordinate space."""

    @abstractmethod
    def from_path(
        self,
    ) -> "BaseElement":
        """Construct Element from file."""
        # TODO: maybe split into local and remote?

    @abstractproperty
    def shape(self) -> Tuple[int, ...]:
        """Return shape of element."""


class Image(BaseElement):
    def __init__(self, image: ArrayLike, transform: Transform) -> None:
        self.data = image
        self.transforms = transform
        super().__init__()

    def to_zarr(self, group: zarr.Group, name: str, scaler: Optional[Scaler] = None) -> None:
        # TODO: write coordinate transforms
        # TODO: allow to write from path
        write_image(
            image=self.data.data,
            group=group,
            axes=["c", "y", "x"],  # TODO: infer before.
            scaler=scaler,
        )

    @classmethod
    def transform(cls, new_coordinate_space: str, inplace: bool = False) -> "BaseElement":
        pass

    @classmethod
    def from_path(
        cls,
    ) -> "BaseElement":
        pass

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape  # type: ignore[no-any-return]


class Labels(BaseElement):
    def __init__(self, labels: ArrayLike, transform: Transform, table: AnnData) -> None:
        self.data = labels
        self.transforms = transform
        self.table = table
        super().__init__()

    def to_zarr(self, group: zarr.Group, name: str, scaler: Optional[Scaler] = None) -> None:
        write_labels(
            labels=self.data.data,
            group=group,
            name=name,
            axes=["y", "x"],  # TODO: infer before.
            scaler=scaler,
        )
        if self.table is not None:
            write_table(tables=self.table, group=group, name=name)

    @classmethod
    def transform(cls, new_coordinate_space: str, inplace: bool = False) -> "BaseElement":
        pass

    @classmethod
    def from_path(
        cls,
    ) -> "BaseElement":
        pass

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape  # type: ignore[no-any-return]


class Points(BaseElement):
    def __init__(self, points: AnnData, transform: Transform, table: AnnData) -> None:
        self.data = points
        self.transforms = transform
        self.table = table
        super().__init__()

    def to_zarr(self, group: zarr.Group, name: str, scaler: Optional[Scaler] = None) -> None:
        write_points(
            points=self.data,
            group=group,
            name=name,
            axes=["y", "x"],  # TODO: infer before.
        )
        if self.table is not None:
            write_table(tables=self.table, group=group, name=name)

    @classmethod
    def transform(cls, new_coordinate_space: str, inplace: bool = False) -> "BaseElement":
        pass

    @classmethod
    def from_path(
        cls,
    ) -> "BaseElement":
        pass

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape  # type: ignore[no-any-return]


class Polygons(BaseElement):
    def __init__(self, polygons: Any, transform: Transform, table: AnnData) -> None:
        self.data = polygons
        self.transforms = transform
        self.table = table
        super().__init__()

    def to_zarr(self, group: zarr.Group, name: str, scaler: Optional[Scaler] = None) -> None:
        write_polygons(
            polygons=self.data,
            group=group,
            name=name,
            axes=["y", "x"],  # TODO: infer before.
        )
        if self.table is not None:
            write_table(tables=self.table, group=group, name=name)

    @classmethod
    def transform(cls, new_coordinate_space: str, inplace: bool = False) -> "BaseElement":
        pass

    @classmethod
    def from_path(
        cls,
    ) -> "BaseElement":
        pass

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape  # type: ignore[no-any-return]
