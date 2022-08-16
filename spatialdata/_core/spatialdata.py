import hashlib
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Optional, Tuple

import zarr
from anndata import AnnData
from ome_zarr.io import parse_url

from spatialdata._core.elements import Image, Labels, Points, Polygons


class SpatialData:
    """Spatial data structure."""

    images: Mapping[str, Image] = MappingProxyType({})
    labels: Mapping[str, Labels] = MappingProxyType({})
    points: Mapping[str, Points] = MappingProxyType({})
    polygons: Mapping[str, Polygons] = MappingProxyType({})
    _table: Optional[AnnData] = None

    def __init__(
        self,
        images: Mapping[str, Any] = MappingProxyType({}),
        labels: Mapping[str, Any] = MappingProxyType({}),
        points: Mapping[str, Any] = MappingProxyType({}),
        polygons: Mapping[str, Any] = MappingProxyType({}),
        table: Optional[AnnData] = None,
        images_transform: Optional[Mapping[str, Any]] = None,
        labels_transform: Optional[Mapping[str, Any]] = None,
        points_transform: Optional[Mapping[str, Any]] = None,
        polygons_transform: Optional[Mapping[str, Any]] = None,
    ) -> None:

        _validate_dataset(images, images_transform)
        _validate_dataset(labels, labels_transform)
        _validate_dataset(points, points_transform)
        _validate_dataset(polygons, polygons_transform)

        if images is not None:
            self.images = {
                k: Image.parse_image(data, transform) for (k, data), transform in _iter_elems(images, images_transform)
            }

        if labels is not None:
            self.labels = {
                k: Labels.parse_labels(data, transform)
                for (k, data), transform in _iter_elems(labels, labels_transform)
            }

        if points is not None:
            self.points = {
                k: Points.parse_points(data, transform)
                for (k, data), transform in _iter_elems(points, points_transform)
            }
        if polygons is not None:
            self.polygons = {
                k: Polygons.parse_polygons(data, transform)
                for (k, data), transform in _iter_elems(polygons, polygons_transform)
            }

        if table is not None:
            self.table = table

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

    @property
    def table(self) -> AnnData:
        self._table

    @table.setter
    def table(self, table: AnnData) -> None:
        self._table = table

    def __repr__(self) -> str:
        return self._gen_repr()

    def _gen_repr(
        self,
    ) -> str:
        def rreplace(s: str, old: str, new: str, occurrence: int) -> str:
            li = s.rsplit(old, occurrence)
            return new.join(li)

        def h(s: str) -> str:
            return hashlib.md5(repr(s).encode()).hexdigest()

        descr = "SpatialData object with:"
        for attr in ["images", "labels", "points", "polygons", "tables"]:
            attribute = getattr(self, attr)
            descr += f"\n{h('level0')}{attr.capitalize()}"
            descr = rreplace(descr, h("level0"), "└── ", 1)
            if attribute is not None:
                for k, v in attribute.items():
                    descr += f"{h('empty_line')}"
                    if isinstance(v, AnnData):
                        descr_class = v.__class__.__name__
                    else:
                        descr_class = v.data.__class__.__name__
                    descr += f"{h('level1.0')}'{k}': {descr_class} {v.shape}"
                    descr = rreplace(descr, h("level1.0"), "    └── ", 1)
            if attr == "tables":
                descr = descr.replace(h("empty_line"), "\n  ")
            else:
                descr = descr.replace(h("empty_line"), "\n│ ")
        return descr


def _iter_elems(
    data: Mapping[str, Any], transforms: Optional[Mapping[str, Any]] = None
) -> Iterable[Tuple[Tuple[str, Any], Any]]:
    # TODO: handle logic for multiple coordinate transforms and elements
    # ...
    return zip(
        data.items(),
        [transforms.get(k, None) if transforms is not None else None for k in data.keys()],
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
