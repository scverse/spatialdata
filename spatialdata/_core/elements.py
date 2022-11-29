import json
import re
from abc import ABC, abstractmethod, abstractproperty
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import zarr
from anndata import AnnData
from dask.array.core import Array as DaskArray
from ome_zarr.scale import Scaler
from xarray import DataArray

from spatialdata._core.transformations import (  # , get_transform
    BaseTransformation, Identity)
from spatialdata._io.write import (write_image, write_labels, write_points,
                                   write_polygons)
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
        if isinstance(image, DataArray):
            self.data = image
        elif isinstance(image, np.ndarray):
            self.data = DataArray(image)
        elif isinstance(image, DaskArray):
            self.data = DataArray(image)
        else:
            raise TypeError("Image must be a DataArray, numpy array or dask array")
        self.transforms = transform
        self.axes = self._infer_axes(image.shape)
        super().__init__()

    @staticmethod
    def parse_image(data: Any, transform: Optional[Any] = None) -> "Image":
        # data, transform = parse_dataset(data, transform)
        if transform is None:
            transform = Identity()
        return Image(data, transform)

    def to_zarr(
        self,
        group: zarr.Group,
        name: str,
        scaler: Optional[Scaler] = None,
        storage_options: Optional[Dict[str, Any]] = None,
        chunks: Optional[Union[Tuple[Any, ...], int]] = None,
    ) -> None:
        if storage_options is None:
            # at the moment we don't use the compressor because of the bug described here (it makes some tests the
            # test_readwrite_roundtrip fail) https://github.com/ome/ome-zarr-py/issues/219
            storage_options = {"compressor": None}
        assert isinstance(self.transforms, BaseTransformation)
        coordinate_transformations = self.transforms.to_dict()
        write_image(
            image=self.data.data,
            name=name,
            group=group,
            axes=self.axes,
            scaler=scaler,
            coordinate_transformations=[[coordinate_transformations]],
            storage_options=storage_options,
            chunks=chunks,
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
        if isinstance(labels, DataArray):
            self.data = labels
        elif isinstance(labels, np.ndarray):
            self.data = DataArray(labels)
        elif isinstance(labels, DaskArray):
            self.data = DataArray(labels)
        else:
            raise TypeError("Labels must be a DataArray, numpy array or dask array")
        self.transforms = transform
        super().__init__()

    @staticmethod
    def parse_labels(data: Any, transform: Optional[Any] = None) -> "Labels":
        # data, transform = parse_dataset(data, transform)
        if transform is None:
            transform = Identity()
        return Labels(data, transform)

    def to_zarr(
        self,
        group: zarr.Group,
        name: str,
        scaler: Optional[Scaler] = None,
        storage_options: Optional[Dict[str, Any]] = None,
        chunks: Optional[Union[Tuple[Any, ...], int]] = None,
    ) -> None:
        if storage_options is None:
            # at the moment we don't use the compressor because of the bug described here (it makes some tests the
            # test_readwrite_roundtrip fail) https://github.com/ome/ome-zarr-py/issues/219
            storage_options = {"compressor": None}
        assert isinstance(self.transforms, BaseTransformation)
        coordinate_transformations = self.transforms.to_dict()
        # at the moment we don't use the compressor because of the bug described here (it makes some tests the
        # test_readwrite_roundtrip fail) https://github.com/ome/ome-zarr-py/issues/219
        write_labels(
            labels=self.data.data,
            group=group,
            name=name,
            axes=["y", "x"],  # TODO: infer before.
            scaler=scaler,
            coordinate_transformations=[[coordinate_transformations]],
            storage_options=storage_options,
            chunks=chunks,
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
        assert isinstance(self.transforms, BaseTransformation)
        coordinate_transformations = self.transforms.to_dict()
        write_points(
            points=self.data,
            group=group,
            name=name,
            axes=["y", "x"],  # TODO: infer before.
            coordinate_transformations=[[coordinate_transformations]],
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
        assert isinstance(self.transforms, BaseTransformation)
        coordinate_transformations = self.transforms.to_dict()
        write_polygons(
            polygons=self.data,
            group=group,
            name=name,
            axes=["y", "x"],  # TODO: infer before.
            coordinate_transformations=[[coordinate_transformations]],
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
