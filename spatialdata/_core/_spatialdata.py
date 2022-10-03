from __future__ import annotations

import copy
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, Union, List

import zarr
from anndata import AnnData
from ome_zarr.io import parse_url

from spatialdata._core.elements import Image, Labels, Points, Polygons
from spatialdata._io.write import write_table
from spatialdata._core.transform import BaseTransformation, get_transformation_from_dict
from spatialdata._core.coordinate_system import CoordSystem_t, CoordinateSystem


# def spatialdata_from_base_elements(
#     images: Optional[Dict[str, Image]] = None,
#     labels: Optional[Dict[str, Labels]] = None,
#     points: Optional[Dict[str, Points]] = None,
#     polygons: Optional[Dict[str, Polygons]] = None,
#     table: Optional[AnnData] = None,
# ) -> SpatialData:
#     # transforms
#     images_transforms = {k: t for k, t in images.items()} if images is not None else None
#     labels_transforms = {k: t for k, t in labels.items()} if labels is not None else None
#     points_transforms = {k: t for k, t in points.items()} if points is not None else None
#     polygons_transforms = {k: t for k, t in polygons.items()} if polygons is not None else None
#     # axes information
#     # TODO:
#
#     return SpatialData(
#         images=images if images is not None else {},
#         labels=labels if labels is not None else {},
#         points=points if points is not None else {},
#         polygons=polygons if polygons is not None else {},
#         table=table,
#         images_transforms=images_transforms,
#         labels_transforms=labels_transforms,
#         points_transforms=points_transforms,
#         polygons_transforms=polygons_transforms,
#     )


class SpatialData:
    """Spatial data structure."""

    images: Mapping[str, Image] = MappingProxyType({})
    labels: Mapping[str, Labels] = MappingProxyType({})
    points: Mapping[str, Points] = MappingProxyType({})
    polygons: Mapping[str, Polygons] = MappingProxyType({})
    _table: Optional[AnnData] = None

    def __init__(
        self,
        # base elements
        images: Mapping[str, Any] = MappingProxyType({}),
        labels: Mapping[str, Any] = MappingProxyType({}),
        points: Mapping[str, Any] = MappingProxyType({}),
        polygons: Mapping[str, Any] = MappingProxyType({}),
        table: Optional[AnnData] = None,
        # # transforms
        # images_transforms: Optional[Mapping[str, Any]] = None,
        # labels_transforms: Optional[Mapping[str, Any]] = None,
        # points_transforms: Optional[Mapping[str, Any]] = None,
        # polygons_transforms: Optional[Mapping[str, Any]] = None,
        # axes information
        images_axes: Optional[Mapping[str, Any]] = None,
        labels_axes: Optional[Mapping[str, Any]] = None,
        # transformations and coordinate systems
        transformations: Mapping[(str, str), Union[BaseTransformation, Dict[Any]]] = MappingProxyType({}),
        coordinate_systems: Optional[List[Union[CoordSystem_t, CoordinateSystem]]] = None,
    ) -> None:
        if coordinate_systems is None:
            raise ValueError("Coordinate systems must be provided.")
        validated_coordinate_systems = _validate_coordinate_systems(coordinate_systems)

        images_transformations = _validate_transformations(images, transformations, validated_coordinate_systems)
        labels_transformations = _validate_transformations(labels, transformations, validated_coordinate_systems)
        points_transformations = _validate_transformations(points, transformations, validated_coordinate_systems)
        polygons_transformations = _validate_transformations(polygons, transformations, validated_coordinate_systems)

        for target, element_class, elements, element_transformations in zip(
            [self.images, self.labels, self.points, self.polygons],
            [Image, Labels, Points, Polygons],
            [images, labels, points, polygons],
            [images_transformations, labels_transformations, points_transformations, polygons_transformations],
        ):
            target = {}
            for name, data in elements.items():
                alignment_info = {validated_coordinate_systems[des]: element_transformations[name][des] for des in element_transformations[name]}
                obj = element_class(data, alignment_info=alignment_info)
                target[name] = obj
        # if images is not None:
        #     self.images = {
        #         k: Image.parse_image(data, transform) for (k, data), transform in _iter_elems(images, images_transforms)
        #     }
        #
        # if labels is not None:
        #     self.labels = {
        #         k: Labels.parse_labels(data, transform)
        #         for (k, data), transform in _iter_elems(labels, labels_transforms)
        #     }
        #
        # if points is not None:
        #     self.points = {
        #         k: Points.parse_points(data, transform)
        #         for (k, data), transform in _iter_elems(points, points_transforms)
        #     }
        # if polygons is not None:
        #     self.polygons = {
        #         k: Polygons.parse_polygons(data, transform)
        #         for (k, data), transform in _iter_elems(polygons, polygons_transforms)
        #     }

        if table is not None:
            self._table = table

    def write(self, file_path: str) -> None:
        """Write to Zarr file."""

        store = parse_url(file_path, mode="w").store
        root = zarr.group(store=store)

        # get union of unique ids of all elements
        elems = set().union(*[set(i) for i in [self.images, self.labels, self.points, self.polygons]])

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

        if self.table is not None:
            write_table(tables=self.table, group=root, name="table")

    @property
    def table(self) -> AnnData:
        return self._table

    @classmethod
    def read(cls, file_path: str) -> SpatialData:
        from spatialdata._io.read import read_zarr

        sdata = read_zarr(file_path)
        return sdata

    @property
    def coordinate_syestems(self) -> List[CoordinateSystem]:
        raise NotImplementedError(
            "get the corodinate systems from the elements and check for consistency (same name "
            "-> same coordinate system"
        )

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
        for attr in ["images", "labels", "points", "polygons", "table"]:
            attribute = getattr(self, attr)
            if attribute is not None and len(attribute) > 0:
                descr += f"\n{h('level0')}{attr.capitalize()}"
                if isinstance(attribute, AnnData):
                    descr += f"{h('empty_line')}"
                    descr_class = attribute.__class__.__name__
                    descr += f"{h('level1.0')}'{attribute}': {descr_class} {attribute.shape}"
                    descr = rreplace(descr, h("level1.0"), "    └── ", 1)
                else:
                    # descr = rreplace(descr, h("level0"), "└── ", 1)
                    for k, v in attribute.items():
                        descr += f"{h('empty_line')}"
                        descr_class = v.data.__class__.__name__
                        if attr == "points":
                            descr += f"{h(attr + 'level1.1')}'{k}': {descr_class} with osbm.spatial {v.shape}"
                        elif attr == "polygons":
                            # assuming 2d
                            descr += (
                                f"{h(attr + 'level1.1')}'{k}': {descr_class} with osb.spatial describing "
                                f"{len(v.data.obs)} 2D polygons"
                            )
                        else:
                            descr += f"{h(attr + 'level1.1')}'{k}': {descr_class} {v.shape}"
                        # descr = rreplace(descr, h("level1.0"), "    └── ", 1)
            if attr == "table":
                descr = descr.replace(h("empty_line"), "\n  ")
            else:
                descr = descr.replace(h("empty_line"), "\n│ ")

        descr = rreplace(descr, h("level0"), "└── ", 1)
        descr = descr.replace(h("level0"), "├── ")

        for attr in ["images", "labels", "points", "polygons", "table"]:
            descr = rreplace(descr, h(attr + "level1.1"), "    └── ", 1)
            descr = descr.replace(h(attr + "level1.1"), "    ├── ")
        ##
        return descr


# def _iter_elems(
#     data: Mapping[str, Any], transforms: Optional[Mapping[str, Any]] = None
# ) -> Iterable[Tuple[Tuple[str, Any], Any]]:
#     # TODO: handle logic for multiple coordinate transforms and elements
#     # ...
#     return zip(
#         data.items(),
#         [transforms.get(k, None) if transforms is not None else None for k in data.keys()],
#     )


def _validate_coordinate_systems(
    coordinate_systems: Optional[List[Union[CoordSystem_t, CoordinateSystem]]]
) -> Dict[str, CoordinateSystem]:
    validated = []
    for c in coordinate_systems:
        if isinstance(c, CoordinateSystem):
            validated.append(copy.deepcopy(c))
        elif type(c) == CoordSystem_t:
            v = CoordinateSystem()
            v.from_dict(c)
            validated.append(v)
        else:
            raise TypeError(f"Invalid type for coordinate system: {type(c)}")
    assert len(coordinate_systems) == len(validated)
    assert len(validated) == len(set(validated))
    d = {v.name: v for v in validated}
    assert len(d) == len(validated)
    return d


# def _validate_dataset(
#     dataset: Optional[Mapping[str, Any]],
#     transformations: Mapping[(str, str), Union[BaseTransformation, Dict[Any]]],
#     coordinate_systems: List[Union[CoordSystem_t, CoordinateSystem]],
# ) -> Tuple[Dict[(str, str), BaseTransformation], List[CoordinateSystem]]:
#     validated_transformations = {}
#     if dataset is None:
#         return validated_transformations, coordinate_systems
#     elif isinstance(dataset, Mapping):
#         for name, element in dataset.items():
#             _validate_element(element, transformations)
#         # if tra is not None:
#         #     if not set(dataset).issuperset(dataset_transform):
#         #         raise ValueError(
#         #             f"Invalid `dataset_transform` keys not present in `dataset`: `{set(dataset_transform).difference(set(dataset))}`."
#         #         )
#     else:
#         raise TypeError('invalid type for "dataset"')


def _validate_transformations(
    elements: Mapping[str, Any],
    transformations: Mapping[Tuple[str, str], Union[BaseTransformation, Dict[str, Any]]],
    coordinate_systems: Dict[str, CoordinateSystem],
) -> Dict[str, Dict[str, BaseTransformation]]:
    validated: Dict[str, Dict[str, BaseTransformation]] = {}
    for name, element in elements.items():
        if not isinstance(element, (Image, Labels, Points, Polygons)):
            raise TypeError(f"Invalid type for element: {type(element)}")
        validated[name] = {}
    for (src, des), t in transformations.items():
        assert des in coordinate_systems.keys()
        element_types = set([type(e) for e in elements.values()])
        assert len(element_types) == 1
        element_type = element_types.pop()
        prefix = {Image: 'image', Labels: 'labels', Points: 'points', Polygons: 'polygons'}[element_type]
        if src.startswith(prefix):
            src_name = src[len(prefix) + 1:]
            if src_name in elements:
                if isinstance(t, BaseTransformation):
                    v = copy.deepcopy(t)
                elif type(t) == Dict[str, Any]:
                    v = get_transformation_from_dict(t)
                else:
                    raise TypeError(f"Invalid type for transformation: {type(t)}")
                validated[src_name][des] = v
    return validated


if __name__ == "__main__":
    sdata = SpatialData.read("spatialdata-sandbox/merfish/data.zarr")
    s = sdata.polygons["anatomical"].data.obs.iloc[0]["spatial"]
    print(Polygons.string_to_tensor(s))
    print(sdata)
    print("ehi")
