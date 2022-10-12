from __future__ import annotations

import copy
import os
import shutil
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import zarr
from anndata import AnnData
from ome_zarr.io import parse_url

from spatialdata._core.coordinate_system import CoordinateSystem
from spatialdata._core.elements import Image, Labels, Points, Polygons
from spatialdata._core.transform import (
    Affine,
    BaseTransformation,
    Sequence,
    get_transformation_from_dict,
)
from spatialdata._io.write import write_table

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
        # axes information
        images_axes: Optional[Mapping[str, Tuple[str, ...]]] = None,
        labels_axes: Optional[Mapping[str, Tuple[str, ...]]] = None,
        # transformations and coordinate systems
        transformations: Mapping[(str, str), Optional[Union[BaseTransformation, Dict[Any]]]] = MappingProxyType({}),
        _validate_transformations: bool = True,
    ) -> None:
        self.file_path: Optional[str] = None

        # reorders the axes to follow the ngff 0.4 convention (t, c, z, y, x)
        for d in [images, labels]:
            for k, v in d.items():
                if not isinstance(v, list):
                    ordered = tuple([d for d in ["t", "c", "z", "y", "x"] if d in v.dims])
                    d[k] = v.transpose(*ordered)
                else:
                    # TODO: multiscale image loaded from disk, check that all is right
                    pass

        def _add_prefix(d: Mapping[str, Any], prefix: str) -> Mapping[str, Any]:
            return {f"/{prefix}/{k}": v for k, v in d.items()}

        ndims = _get_ndims(
            _add_prefix(images, "images")
            | _add_prefix(labels, "labels")
            | _add_prefix(points, "points")
            | _add_prefix(polygons, "polygons")
        )
        coordinate_systems = _infer_coordinate_systems(transformations, ndims)
        images_axes = {f"/images/{k}": v.dims if not isinstance(v, list) else v[0].dims for k, v in images.items()}
        labels_axes = {f"/labels/{k}": v.dims if not isinstance(v, list) else v[0].dims for k, v in labels.items()}
        if _validate_transformations:
            _infer_and_validate_transformations(
                transformations=transformations,
                coordinate_systems=coordinate_systems,
                ndims=ndims,
                images_axes=images_axes,
                labels_axes=labels_axes,
            )
        for element_class, elements, prefix in zip(
            [Image, Labels, Points, Polygons],
            [images, labels, points, polygons],
            ["images", "labels", "points", "polygons"],
        ):
            self.__setattr__(prefix, {})
            expanded_transformations = _expand_transformations(
                list(elements.keys()), prefix, transformations, coordinate_systems
            )
            for name, data in elements.items():
                alignment_info = {
                    coordinate_systems[des]: expanded_transformations[f"/{prefix}/{name}"][des]
                    for des in expanded_transformations[f"/{prefix}/{name}"]
                }
                obj = element_class(data, alignment_info=alignment_info)
                self.__getattribute__(prefix)[name] = obj

        if table is not None:
            self._table = table

    def _save_element(
        self,
        element_type: str,
        name: str,
        overwrite: bool = False,
        zarr_root: Optional[zarr.Group] = None,
        path: Optional[str] = None,
    ):
        if element_type not in ["images", "labels", "points", "polygons"]:
            raise ValueError(f"Element type {element_type} not supported.")
        if zarr_root is not None:
            assert path is None
            assert not self.is_backed()
            root = zarr_root
        elif path is not None:
            if self.is_backed():
                assert path == self.file_path
            store = parse_url(path, mode="a").store
            root = zarr.group(store=store)
        else:
            if not self.is_backed():
                raise ValueError("No backed storage found")
            store = parse_url(self.file_path, mode="a").store
            root = zarr.group(store=store)
        if overwrite:
            if element_type in "images":
                raise ValueError(
                    "Overwriting images is not supported. This is a current limitation of the storage (labels may be "
                    "be contained in the same zarr group as the images). Please open a GitHue issue and we will "
                    "address this problem."
                )
            full_path_group = os.path.join(root.path, f"{name}/{element_type}/{name}")
            if os.path.isdir(full_path_group):
                shutil.rmtree(full_path_group)
        elem_group = root.require_group(name=name)
        self.__getattribute__(element_type)[name].to_zarr(elem_group, name=name)

    def write(self, file_path: str) -> None:
        """Write to Zarr file."""

        store = parse_url(file_path, mode="w").store
        root = zarr.group(store=store)

        # get union of unique ids of all elements
        elems = set().union(*[set(i) for i in [self.images, self.labels, self.points, self.polygons]])

        for el in elems:
            for element_type in ["images", "labels", "points", "polygons"]:
                if el in self.__getattribute__(element_type):
                    self._save_element(element_type, el, zarr_root=root)

        if self.table is not None:
            write_table(tables=self.table, group=root, name="table")

    @property
    def table(self) -> AnnData:
        return self._table

    @classmethod
    def read(
        cls, file_path: str, coordinate_system_names: Optional[Union[str, List[str]]] = None, filter_table: bool = False
    ) -> SpatialData:
        """

        Parameters
        ----------
        file_path : str
            The path to the zarr store or the zarr group.
        coordinate_system_names : Optional[Union[str, List[str]]]
            The names of the coordinate systems to read. If None, all coordinate systems are read.
        filter_table : bool
            If True, the table is filtered to only contain rows that are associated to regions in the specified
            coordinate systems.
        Returns
        -------
        SpatialData
            The spatial data object.

        """

        from spatialdata._io.read import read_zarr

        sdata = read_zarr(file_path, coordinate_system_names=coordinate_system_names, filter_table=filter_table)
        sdata.file_path = file_path
        return sdata

    def is_backed(self) -> bool:
        return self.file_path is not None

    def filter_by_coordinate_system(self, coordinate_system_names: Union[str, List[str]]) -> SpatialData:
        """Filter the spatial data by coordinate system names.

        Parameters
        ----------
        coordinate_system_names
            The coordinate system names to filter by.

        Returns
        -------
        SpatialData
            The filtered spatial data.
        """
        if isinstance(coordinate_system_names, str):
            coordinate_system_names = [coordinate_system_names]
        transformations = {}
        filtered_elements = {}
        for element_type in ["images", "labels", "points", "polygons"]:
            filtered_elements[element_type] = {
                k: v.data
                for k, v in self.__getattribute__(element_type).items()
                if any([s in coordinate_system_names for s in v.coordinate_systems.keys()])
            }
            transformations.update(
                {
                    (f"/{element_type}/{name}", cs_name): ct
                    for name, v in self.__getattribute__(element_type).items()
                    for cs_name, ct in v.transformations.items()
                    if cs_name in coordinate_system_names
                }
            )
        if all([len(v) == 0 for v in filtered_elements.values()]):
            raise ValueError("No elements found in the specified coordinate systems.")
        regions_key = self.table.uns["mapping_info"]["regions_key"]
        elements_in_coordinate_systems = [
            src for src, cs_name in transformations.keys() if cs_name in coordinate_system_names
        ]
        table = self.table[self.table.obs[regions_key].isin(elements_in_coordinate_systems)].copy()
        sdata = SpatialData(
            images=filtered_elements["images"],
            labels=filtered_elements["labels"],
            points=filtered_elements["points"],
            polygons=filtered_elements["polygons"],
            transformations=transformations,
            table=table,
            _validate_transformations=False,
        )
        ##
        return sdata

    @classmethod
    def concatenate(self, *sdatas: SpatialData) -> SpatialData:
        """Concatenate multiple spatial data objects.

        Parameters
        ----------
        sdatas
            The spatial data objects to concatenate.

        Returns
        -------
        SpatialData
            The concatenated spatial data object.
        """
        assert type(sdatas) == tuple
        assert len(sdatas) == 1
        sdatas_ = sdatas[0]
        # TODO: check that .uns['mapping_info'] is the same for all tables and if not warn that it will be
        #  discarded
        # by AnnData.concatenate
        list_of_tables = [sdata.table for sdata in sdatas_ if sdata.table is not None]
        if len(list_of_tables) > 0:
            merged_table = AnnData.concatenate(*list_of_tables, join="outer", uns_merge="same")
            if "mapping_info" in merged_table.uns:
                if (
                    "regions_key" in merged_table.uns["mapping_info"]
                    and "instance_key" in merged_table.uns["mapping_info"]
                ):
                    merged_regions = []
                    for sdata in sdatas_:
                        regions = sdata.table.uns["mapping_info"]["regions"]
                        if isinstance(regions, str):
                            merged_regions.append(regions)
                        else:
                            merged_regions.extend(regions)
                    merged_table.uns["mapping_info"]["regions"] = merged_regions
        else:
            merged_table = None
        ##
        transformations = {}
        for sdata in sdatas_:
            for element_type in ["images", "labels", "points", "polygons"]:
                for name, element in sdata.__getattribute__(element_type).items():
                    for cs_name, ct in element.transformations.items():
                        transformations[(f"/{element_type}/{name}", cs_name)] = ct
        ##
        sdata = SpatialData(
            images={**{k: v.data for sdata in sdatas_ for k, v in sdata.images.items()}},
            labels={**{k: v.data for sdata in sdatas_ for k, v in sdata.labels.items()}},
            points={**{k: v.data for sdata in sdatas_ for k, v in sdata.points.items()}},
            polygons={**{k: v.data for sdata in sdatas_ for k, v in sdata.polygons.items()}},
            transformations=transformations,
            table=merged_table,
            _validate_transformations=False,
        )
        ##
        return sdata

    def _gen_spatial_elements(self):
        # notice that this does not return a table, so we assume that the table does not contain spatial information;
        # this needs to be checked in the future as the specification evolves
        for k in ["images", "labels", "points", "polygons"]:
            d = getattr(self, k)
            for name, obj in d.items():
                yield k, name, obj

    @property
    def coordinate_systems(self) -> Dict[CoordinateSystem]:
        ##
        all_cs = {}
        gen = self._gen_spatial_elements()
        for _, _, obj in gen:
            for name, cs in obj.coordinate_systems.items():
                if name in all_cs:
                    added = all_cs[name]
                    assert cs == added
                else:
                    all_cs[name] = cs
        ##
        return all_cs

    def __repr__(self) -> str:
        return self._gen_repr()

    def _gen_repr(
        self,
    ) -> str:
        def rreplace(s: str, old: str, new: str, occurrence: int) -> str:
            """Reverse replace a up to a certain number of occurences."""
            li = s.rsplit(old, occurrence)
            return new.join(li)

        def h(s: str) -> str:
            return s
            # return hashlib.md5(repr(s).encode()).hexdigest()

        ##
        descr = "SpatialData object with:"
        attributes = ["images", "labels", "points", "polygons", "table"]
        for attr in attributes:
            attribute = getattr(self, attr)
            if attribute is not None and len(attribute) > 0:
                descr += f"\n{h('level0')}{attr}"
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
                            axes = ["x", "y", "z"][: v.ndim]
                            descr += (
                                f"{h(attr + 'level1.1')}'{k}': {descr_class} with osbm.spatial {v.shape}, "
                                f"with axes {', '.join(axes)}"
                            )
                        elif attr == "polygons":
                            # assuming 2d
                            axes = ["x", "y", "z"][: v.ndim]
                            descr += (
                                f"{h(attr + 'level1.1')}'{k}': {descr_class} with obs.spatial describing "
                                f"{len(v.data.obs)} polygons, with axes {', '.join(axes)}"
                            )
                        else:
                            assert attr in ["images", "labels"]
                            descr += (
                                f"{h(attr + 'level1.1')}'{k}': {descr_class} {v.shape}, with axes: "
                                f"{', '.join(v.data.dims)}"
                            )
                        # descr = rreplace(descr, h("level1.0"), "    └── ", 1)
            # the following lines go from this
            #     SpatialData object with:
            #     ├── Images
            #     │     └── 'image': DataArray (200, 100)
            #     └── Points
            #     │     ├── 'points': AnnData with osbm.spatial (50, 2)
            #     │     └── 'circles': AnnData with osbm.spatial (56, 2)
            # to this
            #     SpatialData object with:
            #     ├── Images
            #     │     └── 'image': DataArray (200, 100)
            #     └── Points
            #           ├── 'points': AnnData with osbm.spatial (50, 2)
            #           └── 'circles': AnnData with osbm.spatial (56, 2)
            latest_attribute_present = [
                attr
                for attr in attributes
                if getattr(self, attr) is not None and (attr == "table" or getattr(self, attr) != {})
            ][-1]
            if attr == latest_attribute_present:
                descr = descr.replace(h("empty_line"), "\n  ")
            else:
                descr = descr.replace(h("empty_line"), "\n│ ")

        descr = rreplace(descr, h("level0"), "└── ", 1)
        descr = descr.replace(h("level0"), "├── ")

        for attr in ["images", "labels", "points", "polygons", "table"]:
            descr = rreplace(descr, h(attr + "level1.1"), "    └── ", 1)
            descr = descr.replace(h(attr + "level1.1"), "    ├── ")
        ##
        descr += "\nwith coordinate systems:\n"
        for cs in self.coordinate_systems.values():
            descr += f"▸ {cs.name}\n" f'    with axes: {", ".join([axis.name for axis in cs.axes])}\n'
            gen = self._gen_spatial_elements()
            elements_in_cs = []
            for k, name, obj in gen:
                if cs.name in obj.coordinate_systems:
                    elements_in_cs.append(f"/{k}/{name}")
            if len(elements_in_cs) > 0:
                descr += f'    with elements: {", ".join(elements_in_cs)}\n'
        ##
        return descr


def _get_ndims(d: Dict[Any, Union[Image, Label, Point, Polygon]]) -> int:
    if len(d) == 0:
        return 0
    ndims = {}
    for k, v in d.items():
        if k.startswith("/images") or k.startswith("/labels"):
            if not isinstance(v, list):
                w = v
            else:
                w = v[0]
            ndims[k] = len(set(w.dims).difference({"c", "t"}))
        elif k.startswith("/points"):
            ndims[k] = v.obsm["spatial"].shape[1]
        elif k.startswith("/polygons"):
            ndims[k] = Polygons.string_to_tensor(v.obs["spatial"][0]).shape[1]
        else:
            raise ValueError(f"Unknown key {k}")
    return ndims


def _infer_coordinate_systems(
    transformations: Mapping[(str, str), Optional[Union[BaseTransformation, Dict[Any]]]], ndims: Dict[str, int]
) -> Dict[str, CoordinateSystem]:
    def _default_coordinate_system(name: str, ndim: int) -> CoordinateSystem:
        assert ndim in [2, 3]
        from spatialdata._core.coordinate_system import Axis

        axes = [Axis("c", "channel"), Axis("y", "space", "micrometer"), Axis("x", "space", "micrometer")]
        if ndim == 3:
            axes.insert(1, Axis("z", "space", "micrometer"))
        return CoordinateSystem(name, axes=axes)

    if len(transformations) == 0:
        return {"global": _default_coordinate_system("global", max(ndims.values()))}
    targets = set()
    for _, target in transformations.keys():
        targets.add(target)
    coordinate_systems = {}
    for target in targets:
        ndim = max([ndims[src] for src, des in transformations.keys() if des == target])
        coordinate_systems[target] = _default_coordinate_system(target, ndim)
    return coordinate_systems


def _infer_and_validate_transformations(
    transformations: Mapping[Tuple[str, str], Union[BaseTransformation, Dict[str, Any]]],
    coordinate_systems: Dict[str, CoordinateSystem],
    ndims: Dict[str, int],
    images_axes: Dict[str, List[str]],
    labels_axes: Dict[str, List[str]],
) -> None:
    # We check that each element has at least one transformation. If this doesn't happen, we raise an error,
    # unless there is only a single coordinate system, in which case we assign each element to the global coordinate system.
    sources = [src for src, _ in transformations.keys()]
    for name in ndims.keys():
        if name not in sources:
            if len(coordinate_systems) == 1:
                cs = coordinate_systems.values().__iter__().__next__()
                transformations[(name, cs.name)] = None
            else:
                raise ValueError(
                    f"Element {name} has no transformation to any other coordinate system. "
                    f"Please specify a transformation to at least one coordinate system."
                )
    # We populate transformations entries like this ('/images/image', 'target_space', None) by replacing None
    # with the "default" transformations that maps axes to the same axes.
    # We expect (and we check) that images and labels have axes that are a subset of ['t', 'c', 'z', 'y', 'x'],
    # in this order, and that points and polygons have axes that are a subset of ['x', 'y', 'z'], in this order.
    # The correction matrix is to deal with a subtle consequence that the order of the axes of the target coordinate
    # space (which is expected to be a subset of ['t', 'c', 'z', 'y', 'x'], in this order) does not respect the order
    # of points and polygons. So the "default" transformation described above would correct for this, while a user
    # specified transformation, like Scale(), would not.
    for (src, des), transform in transformations.items():
        ss = src.split("/")
        assert len(ss) == 3
        assert ss[0] == ""
        prefix, _ = ss[1:]
        if prefix == "images":
            src_axes = images_axes[src]
            correction_matrix = np.eye(len(src_axes))
        elif prefix == "labels":
            src_axes = labels_axes[src]
            correction_matrix = np.eye(len(src_axes))
        else:
            if prefix == "points":
                ndim = ndims[src]
            elif prefix == "polygons":
                ndim = ndims[src]
            else:
                raise ValueError(f"Element {element} not supported.")
            src_axes = ("x", "y", "z")[:ndim]
            correction_matrix = np.fliplr(np.eye(ndim))
        correction_matrix = np.hstack(
            [
                np.vstack([correction_matrix, np.zeros((1, len(correction_matrix)))]),
                np.zeros((len(correction_matrix) + 1, 1)),
            ]
        )
        correction_matrix[-1, -1] = 1
        des_axes_obj = tuple(coordinate_systems[des].axes)
        des_axes = tuple(axis.name for axis in des_axes_obj)
        affine_matrix = Affine._get_affine_iniection_from_axes(src_axes, des_axes) @ correction_matrix
        from spatialdata._core.transform import get_transformation_from_dict

        affine = get_transformation_from_dict({"type": "affine", "affine": affine_matrix[:-1, :].tolist()})
        if transform is None:
            transformations[(src, des)] = affine
        else:
            transformations[(src, des)] = Sequence([transform, affine])


# )
#             expanded_transformations = _expand_transformations(
#                 list(elements.keys()), prefix, transformations, coordinate_systems
#             )
#             for name, data in elements.i
def _expand_transformations(
    element_keys: List[str],
    prefix: str,
    transformations: Dict[Tuple[str, str], BaseTransformation],
    coordinate_systems: Dict[str, CoordinateSystem],
) -> Dict[str, Dict[str, BaseTransformation]]:
    expanded = {}
    for name in element_keys:
        expanded[f"/{prefix}/{name}"] = {}
    for (src, des), t in transformations.items():
        assert des in coordinate_systems.keys()
        if src.startswith(f"/{prefix}/"):
            src_name = src[len(f"/{prefix}/") :]
            if src_name in element_keys:
                if isinstance(t, BaseTransformation):
                    v = copy.deepcopy(t)
                elif isinstance(t, dict):
                    # elif type(t) == Dict[str, Any]:
                    v = get_transformation_from_dict(t)
                else:
                    raise TypeError(f"Invalid type for transformation: {type(t)}")
                expanded[src][des] = v
    return expanded


if __name__ == "__main__":
    sdata = SpatialData.read("spatialdata-sandbox/merfish/data.zarr")
    s = sdata.polygons["anatomical"].data.obs.iloc[0]["spatial"]
    print(Polygons.string_to_tensor(s))
    print(sdata)
    print("ehi")
