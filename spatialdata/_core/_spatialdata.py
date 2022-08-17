from __future__ import annotations

from functools import singledispatch
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Optional, Tuple

import numpy as np
import xarray as xr
import zarr
from anndata import AnnData
from dask.array.core import Array as DaskArray
from ome_zarr.io import parse_url

from spatialdata._core.elements import Image, Labels, Points, Polygons
from spatialdata._core.transform import Transform, get_transform


class SpatialData:
    """Spatial data structure."""

    images: Optional[Mapping[str, Image]] = None
    labels: Optional[Mapping[str, Labels]] = None
    points: Optional[Mapping[str, Points]] = None
    polygons: Optional[Mapping[str, Polygons]] = None
    tables: Optional[Mapping[str, AnnData]] = None

    def __init__(
        self,
        images: Optional[Mapping[str, Any]] = MappingProxyType({}),
        labels: Optional[Mapping[str, Any]] = MappingProxyType({}),
        points: Optional[Mapping[str, AnnData]] = MappingProxyType({}),
        polygons: Optional[Mapping[str, AnnData]] = MappingProxyType({}),
        tables: Optional[Mapping[str, AnnData]] = MappingProxyType({}),
        images_transform: Optional[Mapping[str, Any]] = None,
        labels_transform: Optional[Mapping[str, Any]] = None,
        points_transform: Optional[Mapping[str, Any]] = None,
        polygons_transform: Optional[Mapping[str, Any]] = None,
    ) -> None:

        _validate_dataset(images, images_transform)
        _validate_dataset(labels, labels_transform)
        _validate_dataset(points, points_transform)
        _validate_dataset(polygons, polygons_transform)
        _validate_dataset(tables)

        if images is not None:
            self.images = {
                k: self.parse_image(data, transform)
                for (k, data), transform, _ in _iter_elems(images, images_transform)
            }

        if labels is not None:
            self.labels = {
                k: self.parse_labels(data, transform, table)
                for (k, data), transform, table in _iter_elems(labels, labels_transform, tables)
            }

        if points is not None:
            self.points = {
                k: self.parse_points(data, transform, table)
                for (k, data), transform, table in _iter_elems(points, points_transform, tables)
            }
        if polygons is not None:
            self.polygons = {
                k: self.parse_polygons(data, transform, table)
                for (k, data), transform, table in _iter_elems(polygons, polygons_transform, tables)
            }

        if tables is not None:
            # TODO: concatenate here? do views above?
            self.tables = {k: self.parse_table(tables[k]) for k in tables.keys()}

    @classmethod
    def parse_image(cls, data: Any, transform: Optional[Any] = None) -> Image:
        data, transform = parse_dataset(data, transform)
        return Image(data, transform)

    @classmethod
    def parse_labels(cls, data: Any, transform: Optional[Any] = None, table: Optional[Any] = None) -> Labels:
        data, transform = parse_dataset(data, transform)
        # TODO: validate table e.g.
        # - do table unique id and labels unique id match?
        # - are table unique id a subset of labels unique id?
        # ...
        return Labels(data, transform, table)

    @classmethod
    def parse_points(cls, data: AnnData, transform: Optional[Any] = None, table: Optional[Any] = None) -> Points:
        data, transform = parse_dataset(data, transform)
        # TODO: validate table
        # TODO: if not visium Points, table should be None
        return Points(data, transform, table)

    @classmethod
    def parse_polygons(cls, data: AnnData, transform: Optional[Any] = None, table: Optional[Any] = None) -> Polygons:
        data, transform = parse_dataset(data, transform)
        # TODO: validate table
        return Polygons(data, transform, table)

    @classmethod
    def parse_table(cls, data: AnnData) -> AnnData:
        data, _ = parse_dataset(data)
        if isinstance(data, AnnData):
            return data
        else:
            raise ValueError("table must be an AnnData object.")

    def write(self, file_path: str) -> None:
        """Write to Zarr file."""

        store = parse_url(file_path, mode="w").store
        root = zarr.group(store=store)

        # get union of unique ids of all elements
        elems = set().union(*[set(i.keys()) for i in self.__dict__.values()])

        for el in elems:
            elem_group = root.create_group(name=el)
            if self.images is not None and el in self.images.keys():
                self.images[el].to_zarr(elem_group, name=el)
            if self.labels is not None and el in self.labels.keys():
                self.labels[el].to_zarr(elem_group, name=el)
            if self.points is not None and el in self.points.keys():
                self.points[el].to_zarr(elem_group, name=el)
            if self.polygons is not None and el in self.polygons.keys():
                self.polygons[el].to_zarr(elem_group, name=el)
            # TODO: shall we write tables?
            # if el in self.tables.keys():
            #     self.tables[el].to_zarr(elem_group, name=el)

    @classmethod
    def read(cls, file_path: str) -> SpatialData:
        from spatialdata._io.read import read_zarr

        sdata = read_zarr(file_path)
        return sdata

    def __repr__(self) -> str:
        return self._gen_repr()

    def _gen_repr(
        self,
    ) -> str:
        def rreplace(s: str, old: str, new: str, occurrence: int) -> str:
            li = s.rsplit(old, occurrence)
            return new.join(li)

        def h(s: str) -> str:
            return s
            # return hashlib.md5(repr(s).encode()).hexdigest()

        ##
        descr = "SpatialData object with:"
        for attr in ["images", "labels", "points", "polygons", "tables"]:
            attribute = getattr(self, attr)
            if len(attribute) > 0:
                descr += f"\n{h('level0')}{attr.capitalize()}"
                # descr = rreplace(descr, h("level0"), "└── ", 1)
                for k, v in attribute.items():
                    descr += f"{h('empty_line')}"
                    if isinstance(v, AnnData):
                        descr_class = v.__class__.__name__
                    else:
                        descr_class = v.data.__class__.__name__
                    descr += f"{h(attr + 'level1.1')}'{k}': {descr_class} {v.shape}"
                    # descr = rreplace(descr, h("level1.0"), "    └── ", 1)
            if attr == "tables":
                descr = descr.replace(h("empty_line"), "\n  ")
            else:
                descr = descr.replace(h("empty_line"), "\n│ ")

        descr = rreplace(descr, h("level0"), "└── ", 1)
        descr = descr.replace(h("level0"), "├── ")

        for attr in ["images", "labels", "points", "polygons", "tables"]:
            descr = rreplace(descr, h(attr + "level1.1"), "    └── ", 1)
            descr = descr.replace(h(attr + "level1.1"), "    ├── ")
        ##
        return descr


@singledispatch
def parse_dataset(data: Any, transform: Optional[Any] = None) -> Any:
    raise NotImplementedError(f"`parse_dataset` not implemented for {type(data)}")


# TODO: ome_zarr reads images/labels as dask arrays
# given curren behaviour, equality fails (since we don't cast to dask arrays)
# should we?
@parse_dataset.register
def _(data: xr.DataArray, transform: Optional[Any] = None) -> Tuple[xr.DataArray, Transform]:
    if transform is None:
        transform = get_transform(data)
    return data, transform


@parse_dataset.register
def _(data: np.ndarray, transform: Optional[Any] = None) -> Tuple[xr.DataArray, Transform]:  # type: ignore[type-arg]
    data = xr.DataArray(data)
    if transform is None:
        transform = get_transform(data)
    return data, transform


@parse_dataset.register
def _(data: DaskArray, transform: Optional[Any] = None) -> Tuple[xr.DataArray, Transform]:
    data = xr.DataArray(data)
    if transform is None:
        transform = get_transform(data)
    return data, transform


@parse_dataset.register
def _(data: AnnData, transform: Optional[Any] = None) -> Tuple[AnnData, Transform]:
    if transform is None:
        transform = get_transform(data)
    return data, transform


def _iter_elems(
    data: Mapping[str, Any], transforms: Optional[Mapping[str, Any]] = None, tables: Optional[Mapping[str, Any]] = None
) -> Iterable[Tuple[Tuple[str, Any], Any, Any]]:
    # TODO: handle logic for multiple tables for a single labels/points
    # TODO: handle logic for multiple labels/points for a single table
    # ...
    return zip(
        data.items(),
        [transforms.get(k, None) if transforms is not None else None for k in data.keys()],
        [tables.get(k, None) if tables is not None else None for k in data.keys()],
    )


def _validate_dataset(
    dataset: Optional[Mapping[str, Any]] = None,
    dataset_transform: Optional[Mapping[str, Any]] = None,
) -> None:
    if dataset is None:
        if dataset_transform is None:
            return
        else:
            raise ValueError("`dataset_transform` is only allowed if dataset is provided.")
    if isinstance(dataset, Mapping):
        if dataset_transform is not None:
            if not set(dataset).issuperset(dataset_transform):
                raise ValueError(
                    f"Invalid `dataset_transform` keys not present in `dataset`: `{set(dataset_transform).difference(set(dataset))}`."
                )
