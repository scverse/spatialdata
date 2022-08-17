from abc import ABC, abstractmethod, abstractproperty
from functools import singledispatch
from typing import Any, List, Optional, Tuple

import numpy as np
import zarr
from anndata import AnnData
from dask.array.core import Array as DaskArray
from ome_zarr.scale import Scaler
from xarray import DataArray

from spatialdata._core.transform import Transform, get_transform
from spatialdata._io.write import (
    write_image,
    write_labels,
    write_points,
    write_polygons,
)

__all__ = ["Image", "Labels", "Points", "Polygons"]


class BaseElement(ABC):
    # store the actual data (e.g., array for image, coordinates for points, etc.)
    data: Any

    # store the transform objects as a dictionary with keys (source_coordinate_space, destination_coordinate_space)
    transforms: Optional[Transform] = None

    @abstractmethod
    def to_zarr(self, group: zarr.Group, name: str, scaler: Optional[Scaler] = None) -> None:
        """Write element to file."""

    @abstractmethod
    def transform_to(self, new_coordinate_space: str, inplace: bool = False) -> "BaseElement":
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
    def __init__(self, image: DataArray, transform: Transform) -> None:
        self.data: DataArray = image
        self.transforms = transform
        self.axes = self._infer_axes(image.shape)
        super().__init__()

    @staticmethod
    def parse_image(data: Any, transform: Optional[Any] = None) -> "Image":
        data, transform = parse_dataset(data, transform)
        return Image(data, transform)

    def to_zarr(self, group: zarr.Group, name: str, scaler: Optional[Scaler] = None) -> None:
        # TODO: allow to write from path
        assert isinstance(self.transforms, Transform)
        coordinate_transformations = self.transforms._to_ngff_transform()
        write_image(
            image=self.data.data,
            group=group,
            axes=self.axes,
            scaler=scaler,
            coordinate_transformations=coordinate_transformations,
        )

    def _infer_axes(self, shape: Tuple[int]) -> List[str]:
        # TODO: improve (this information can be already present in the data, as for xarrays, and the constructor
        # should have an argument so that the user can modify this
        return ["c", "y", "x"][3 - len(shape) :]

    @classmethod
    def transform_to(cls, new_coordinate_space: str, inplace: bool = False) -> "Image":
        raise NotImplementedError()

    @classmethod
    def from_path(
        cls,
    ) -> "BaseElement":
        raise NotImplementedError()

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape  # type: ignore[no-any-return]


class Labels(BaseElement):
    def __init__(self, labels: DataArray, transform: Transform) -> None:
        self.data: DataArray = labels
        self.transforms = transform
        super().__init__()

    @staticmethod
    def parse_labels(data: Any, transform: Optional[Any] = None) -> "Labels":
        data, transform = parse_dataset(data, transform)
        return Labels(data, transform)

    def to_zarr(self, group: zarr.Group, name: str, scaler: Optional[Scaler] = None) -> None:
        assert isinstance(self.transforms, Transform)
        self.transforms._to_ngff_transform()
        write_labels(
            labels=self.data.data,
            group=group,
            name=name,
            axes=["y", "x"],  # TODO: infer before.
            scaler=scaler,
        )

    @classmethod
    def transform_to(cls, new_coordinate_space: str, inplace: bool = False) -> "Labels":
        raise NotImplementedError()

    @classmethod
    def from_path(
        cls,
    ) -> "BaseElement":
        raise NotImplementedError()

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape  # type: ignore[no-any-return]


class Points(BaseElement):
    def __init__(self, points: AnnData, transform: Transform) -> None:
        self.data: AnnData = points
        self.transforms = transform
        super().__init__()

    @staticmethod
    def parse_points(data: AnnData, transform: Optional[Any] = None) -> "Points":
        data, transform = parse_dataset(data, transform)
        return Points(data, transform)

    def to_zarr(self, group: zarr.Group, name: str, scaler: Optional[Scaler] = None) -> None:
        write_points(
            points=self.data,
            group=group,
            name=name,
            axes=["y", "x"],  # TODO: infer before.
        )

    @classmethod
    def transform_to(cls, new_coordinate_space: str, inplace: bool = False) -> "Points":
        raise NotImplementedError()

    @classmethod
    def from_path(
        cls,
    ) -> "BaseElement":
        raise NotImplementedError()

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape  # type: ignore[no-any-return]


class Polygons(BaseElement):
    def __init__(self, polygons: Any, transform: Transform) -> None:
        self.data: Any = polygons
        self.transforms = transform
        super().__init__()

    @staticmethod
    def parse_polygons(data: AnnData, transform: Optional[Any] = None) -> "Polygons":
        data, transform = parse_dataset(data, transform)
        return Polygons(data, transform)

    def to_zarr(self, group: zarr.Group, name: str, scaler: Optional[Scaler] = None) -> None:
        write_polygons(
            polygons=self.data,
            group=group,
            name=name,
            axes=["y", "x"],  # TODO: infer before.
        )

    @classmethod
    def transform_to(cls, new_coordinate_space: str, inplace: bool = False) -> "Polygons":
        raise NotImplementedError()

    @classmethod
    def from_path(
        cls,
    ) -> "BaseElement":
        raise NotImplementedError()

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape  # type: ignore[no-any-return]


@singledispatch
def parse_dataset(data: Any, transform: Optional[Any] = None) -> Any:
    raise NotImplementedError(f"`parse_dataset` not implemented for {type(data)}")


# TODO: ome_zarr reads images/labels as dask arrays
# given curren behaviour, equality fails (since we don't cast to dask arrays)
# should we?
@parse_dataset.register
def _(data: DataArray, transform: Optional[Any] = None) -> Tuple[DataArray, Transform]:
    if transform is not None:
        transform = get_transform(transform)
    return data, transform


@parse_dataset.register
def _(data: np.ndarray, transform: Optional[Any] = None) -> Tuple[DataArray, Transform]:  # type: ignore[type-arg]
    data = DataArray(data)
    if transform is not None:
        transform = get_transform(transform)
    return data, transform


@parse_dataset.register
def _(data: DaskArray, transform: Optional[Any] = None) -> Tuple[DataArray, Transform]:
    data = DataArray(data)
    if transform is not None:
        transform = get_transform(transform)
    return data, transform


@parse_dataset.register
def _(data: AnnData, transform: Optional[Any] = None) -> Tuple[AnnData, Transform]:
    if transform is not None:
        transform = get_transform(transform)
    return data, transform
