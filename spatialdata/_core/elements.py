import json
import re
from abc import ABC, abstractmethod, abstractproperty
from functools import singledispatch
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import zarr
from anndata import AnnData
from ome_zarr.scale import Scaler
from xarray import DataArray

from spatialdata._core.transform import BaseTransformation, Identity  # , get_transform
from spatialdata._io.write import (
    write_image,
    write_labels,
    write_points,
    write_polygons,
)
from spatialdata._types import ArrayLike

__all__ = ["Image", "Labels", "Points", "Polygons"]


class BaseElement(ABC):
    # store the actual data (e.g., array for image, coordinates for points, etc.)
    data: Any

    # store the transform objects as a dictionary with keys (source_coordinate_space, destination_coordinate_space)
    transforms: Optional[BaseTransformation] = None

    @abstractmethod
    def to_zarr(self, group: zarr.Group, name: str, scaler: Optional[Scaler] = None) -> None:
        """Write element to file."""

    @abstractmethod
    def transform_to(self, new_coordinate_space: str, inplace: bool = False) -> "BaseElement":
        """BaseTransformation the object to a new coordinate space."""

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
    def __init__(self, image: DataArray, transform: BaseTransformation) -> None:
        self.data: DataArray = image
        self.transforms = transform
        self.axes = self._infer_axes(image.shape)
        super().__init__()

    @staticmethod
    def parse_image(data: Any, transform: Optional[Any] = None) -> "Image":
        # data, transform = parse_dataset(data, transform)
        if transform is None:
            transform = Identity()
        return Image(data, transform)

    def to_zarr(self, group: zarr.Group, name: str, scaler: Optional[Scaler] = None) -> None:
        # TODO: allow to write from path
        assert isinstance(self.transforms, BaseTransformation)
        coordinate_transformations = self.transforms._to_ngff_transform()
        # at the moment we don't use the compressor because of the bug described here (it makes some tests the
        # test_readwrite_roundtrip fail) https://github.com/ome/ome-zarr-py/issues/219
        write_image(
            image=self.data.data,
            group=group,
            axes=self.axes,
            scaler=scaler,
            coordinate_transformations=coordinate_transformations,
            storage_options={"compressor": None},
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
    def __init__(self, labels: DataArray, transform: BaseTransformation) -> None:
        self.data: DataArray = labels
        self.transforms = transform
        super().__init__()

    @staticmethod
    def parse_labels(data: Any, transform: Optional[Any] = None) -> "Labels":
        # data, transform = parse_dataset(data, transform)
        if transform is None:
            transform = Identity()
        return Labels(data, transform)

    def to_zarr(self, group: zarr.Group, name: str, scaler: Optional[Scaler] = None) -> None:
        assert isinstance(self.transforms, BaseTransformation)
        self.transforms._to_ngff_transform()
        # at the moment we don't use the compressor because of the bug described here (it makes some tests the
        # test_readwrite_roundtrip fail) https://github.com/ome/ome-zarr-py/issues/219
        write_labels(
            labels=self.data.data,
            group=group,
            name=name,
            axes=["y", "x"],  # TODO: infer before.
            scaler=scaler,
            storage_options={"compressor": None},
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
    def __init__(self, points: AnnData, transform: BaseTransformation) -> None:
        self.data: AnnData = points
        self.transforms = transform
        super().__init__()

    @staticmethod
    def parse_points(data: AnnData, transform: Optional[Any] = None) -> "Points":
        # data, transform = parse_dataset(data, transform)
        if transform is None:
            transform = Identity()
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
        return self.data.obsm["spatial"].shape  # type: ignore[no-any-return]


class Polygons(BaseElement):
    def __init__(self, polygons: Any, transform: BaseTransformation) -> None:
        self.data: Any = polygons
        self.transforms = transform
        super().__init__()

    @staticmethod
    def parse_polygons(data: AnnData, transform: Optional[Any] = None) -> "Polygons":
        # data, transform = parse_dataset(data, transform)
        if transform is None:
            transform = Identity()
        return Polygons(data, transform)

    @staticmethod
    def tensor_to_string(x: ArrayLike) -> str:
        s = repr(x).replace("\n", "").replace(" ", "")[len("array(") : -1]
        # consistently check
        y = eval(s)
        assert np.allclose(x, y)
        return s

    @staticmethod
    def string_to_tensor(s: str) -> Union[ArrayLike, None]:
        pattern = r"^\[(?:\[[0-9]+\.[0-9]+,[0-9]+\.[0-9]+\],?)+\]$"
        if re.fullmatch(pattern, s) is not None:
            x: ArrayLike = np.array(eval(s))
            return x

    @staticmethod
    def anndata_from_geojson(path: str) -> AnnData:
        with open(path) as f:
            j = json.load(f)

        names = []
        coordinates = []
        assert "geometries" in j
        for region in j["geometries"]:
            if region["type"] == "Polygon":
                names.append(region["name"])
                vertices: ArrayLike = np.array(region["coordinates"])
                vertices = np.squeeze(vertices, 0)
                assert len(vertices.shape) == 2
                coordinates.append(vertices)
            else:
                print(f'ignoring "{region["type"]}" from geojson')

        string_coordinates = [Polygons.tensor_to_string(c) for c in coordinates]
        a = AnnData(shape=(len(names), 0), obs=pd.DataFrame({"name": names, "spatial": string_coordinates}))
        return a

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


# @singledispatch
# def parse_dataset(data: Any, transform: Optional[Any] = None) -> Any:
#     raise NotImplementedError(f"`parse_dataset` not implemented for {type(data)}")
#
#
# # TODO: ome_zarr reads images/labels as dask arrays
# # given current behavior, equality fails (since we don't cast to dask arrays)
# # should we?
# @parse_dataset.register
# def _(data: DataArray, transform: Optional[Any] = None) -> Tuple[DataArray, BaseTransformation]:
#     if transform is None:
#         transform = get_transform(data)
#     return data, transform
#
#
# @parse_dataset.register
# def _(data: np.ndarray, transform: Optional[Any] = None) -> Tuple[DataArray, BaseTransformation]:  # type: ignore[type-arg]
#     data = DataArray(data)
#     if transform is None:
#         transform = get_transform(data)
#     return data, transform
#
#
# @parse_dataset.register
# def _(data: DaskArray, transform: Optional[Any] = None) -> Tuple[DataArray, BaseTransformation]:
#     data = DataArray(data)
#     if transform is None:
#         transform = get_transform(data)
#     return data, transform
#
#
# @parse_dataset.register
# def _(data: AnnData, transform: Optional[Any] = None) -> Tuple[AnnData, BaseTransformation]:
#     if transform is None:
#         transform = get_transform(data)
#     return data, transform
#
#
# if __name__ == "__main__":
#     geojson_path = "spatialdata-sandbox/merfish/data/processed/anatomical.geojson"
#     a = Polygons.anndata_from_geojson(path=geojson_path)
#     print(a)
