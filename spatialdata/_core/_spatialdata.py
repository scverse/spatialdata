import hashlib
import os
import tempfile
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping, Optional, Set, Tuple

import numpy as np
import xarray as xr
import zarr
from anndata import AnnData
from ome_zarr.io import parse_url

from spatialdata._core.mixin.io_mixin import IoMixin

# from spatialdata._core.writer import write_spatial_anndata
from spatialdata._core.transform import Transform, get_transform, set_transform
from spatialdata._core.writer import (
    write_image,
    write_labels,
    write_points,
    write_shapes,
    write_tables,
)
from spatialdata.utils import are_directories_identical


class SpatialData(IoMixin):
    """Spatial data structure."""

    tables: Optional[AnnData]
    labels: Optional[Mapping[str, Any]]
    images: Optional[Mapping[str, Any]]
    points: Optional[Mapping[str, AnnData]]
    shapes: Optional[Mapping[str, AnnData]]
    elems: Set[str]

    def __init__(
        self,
        tables: Optional[AnnData] = None,
        labels: Optional[Mapping[str, Any]] = MappingProxyType({}),
        labels_transform: Optional[Mapping[str, Any]] = None,
        images: Optional[Mapping[str, Any]] = MappingProxyType({}),
        images_transform: Optional[Mapping[str, Any]] = None,
        points: Optional[Mapping[str, AnnData]] = MappingProxyType({}),
        points_transform: Optional[Mapping[str, Any]] = None,
        shapes: Optional[Mapping[str, AnnData]] = MappingProxyType({}),
        shapes_transform: Optional[Mapping[str, Any]] = None,
    ) -> None:

        elems = []

        images, images_transform, elem_images = _validate_dataset(images, images_transform)
        labels, labels_transform, elem_labels = _validate_dataset(labels, labels_transform)
        points, points_transform, elem_points = _validate_dataset(points, points_transform)
        shapes, shapes_transform, elem_shapes = _validate_dataset(shapes, shapes_transform)

        if TYPE_CHECKING:
            assert images_transform is not None
            assert labels_transform is not None
            assert points_transform is not None
            assert shapes_transform is not None

        if images is not None:
            self.images = {
                k: self.parse_image(image, image_transform)
                for k, (image, image_transform) in zip(images.keys(), zip(images.values(), images_transform.values()))
            }
            elems.append(elem_images)
        else:
            self.images = None

        if labels is not None:
            self.labels = {
                k: self.parse_image(labels, labels_transform)
                for k, (labels, labels_transform) in zip(labels.keys(), zip(labels.values(), labels_transform.values()))
            }
            elems.append(elem_labels)
        else:
            self.labels = None

        if points is not None:
            self.points = {k: self.parse_tables(points[k], points_transform[k]) for k in points.keys()}
            elems.append(elem_points)
        else:
            self.points = None

        if shapes is not None:
            self.shapes = {k: self.parse_tables(shapes[k], shapes_transform[k]) for k in shapes.keys()}
            elems.append(elem_shapes)
        else:
            self.shapes = None

        self.tables = tables
        self.elems = set().union(*elems)  # type: ignore[arg-type]

    @classmethod
    def parse_image(cls, image: Any, image_transform: Optional[Transform] = None) -> Any:
        """Parse image into a xarray.DataArray."""
        if isinstance(image, xr.DataArray):
            if image_transform is not None:
                set_transform(image, image_transform)
            return image
        elif isinstance(image, np.ndarray):
            xa = xr.DataArray(image)
            if image_transform is not None:
                set_transform(xa, image_transform)
            return xa
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")

    @classmethod
    def parse_tables(cls, tables: AnnData, tables_transform: Optional[Transform] = None) -> Any:
        """Parse AnnData in SpatialData."""
        if isinstance(tables, AnnData):
            if tables_transform is not None:
                set_transform(tables, tables_transform)
            return tables
        else:
            raise ValueError(f"Unsupported tables type: {type(tables)}")

    # @classmethod
    def write(self, file_path: str) -> None:
        """Write to Zarr file."""

        store = parse_url(file_path, mode="w").store
        root = zarr.group(store=store)

        for elem in self.elems:
            elem_group = root.create_group(name=elem)
            if self.images is not None and elem in self.images.keys():
                # TODO: get transform
                write_image(
                    image=self.images[elem].data,
                    group=elem_group,
                    axes=["c", "y", "x"],  # TODO: it's not gonna work, need to validate/infer before.
                    scaler=None,
                )
            if self.labels is not None and elem in self.labels.keys():
                # TODO: get transform
                write_labels(
                    labels=self.labels[elem].data,
                    group=elem_group,
                    name=elem,
                    axes=["y", "x"],  # TODO: it's not gonna work, need to validate/infer before.
                    scaler=None,
                )
            if self.points is not None and elem in self.points.keys():
                # TODO: get transform
                write_points(
                    points=self.points[elem],
                    group=elem_group,
                    name=elem,
                    axes=["y", "x"],  # TODO: it's not gonna work, need to validate/infer before.
                )
            if self.shapes is not None and elem in self.shapes.keys():
                # TODO: get transform
                write_shapes(
                    shapes=self.shapes[elem],
                    group=elem_group,
                    name=elem,
                    shapes_parameters=self.shapes[elem].uns["shape_parameters"],
                    axes=["y", "x"],  # TODO: it's not gonna work, need to validate/infer before.
                )

        if self.tables is not None:
            tables_group = root.create_group(name="tables")

            write_tables(
                tables=self.tables,
                group=tables_group,
                name="tables",
                region=list(self.elems),
            )

        # if len(self.images) == 0:
        #     pass
        # else:
        #     # simple case for the moment
        #     assert len(self.images) == 1
        #     transform = get_transform(self.images.values().__iter__().__next__())
        #     transform.translation
        #     transform.scale_factors

        # self.images.values().__iter__().__next__().to_numpy()

        # if len(self.regions) == 0:
        #     pass
        # else:
        #     # simple case for the moment
        #     assert len(self.regions) == 1
        #     self.regions.values().__iter__().__next__()
        # regions_name = self.regions.keys().__iter__().__next__()

    # @classmethod
    # def from_zarr(self, file_path: str) -> "SpatialData":
    #     """Load from Zarr file."""

    #     ome_zarr = parse_url(file_path)
    #     reader = Reader(ome_zarr)
    #     ome_zarr.__dir__()

    #     feature_table = None
    #     regions = {}
    #     images = {}
    #     points = None

    #     ##
    #     import os

    #     groups = os.listdir(ome_zarr.path)
    #     if "tables" in groups:
    #         tables = os.listdir(os.path.join(ome_zarr.path, "tables"))
    #         if len(tables) > 2:
    #             raise RuntimeError("Currently it is possible to work with only one feature table")
    #         for table in tables:
    #             if table == ".zgroup":
    #                 continue
    #             else:
    #                 table_path = os.path.join(ome_zarr.path, "tables", table)
    #                 feature_table = read_zarr(table_path)
    #     if "circles" in groups:
    #         tables = os.listdir(os.path.join(ome_zarr.path, "circles"))
    #         for table in tables:
    #             if table == ".zgroup":
    #                 continue
    #             else:
    #                 table_path = os.path.join(ome_zarr.path, "circles", table)
    #                 adata = read_zarr(table_path)
    #                 regions[table] = adata
    #     if "points" in groups:
    #         tables = os.listdir(os.path.join(ome_zarr.path, "points"))
    #         if len(tables) > 2:
    #             raise RuntimeError("Currently it is possible to work with only one points table")
    #         for table in tables:
    #             if table == ".zgroup":
    #                 continue
    #             else:
    #                 table_path = os.path.join(ome_zarr.path, "points", table)
    #                 points = read_zarr(table_path)
    #     ##
    #     content = [node for node in reader()]
    #     assert len(content) == 1
    #     node = content[0]
    #     data = node.data
    #     metadata = node.metadata
    #     # ignoring pyramidal information for the moment
    #     largest_image = xr.DataArray(data[0]).load()
    #     largest_image = largest_image.transpose()
    #     largest_image_transform = metadata["coordinateTransformations"][0]
    #     d = {}
    #     for e in largest_image_transform:
    #         d[e["type"]] = np.flip(e[e["type"]])
    #     transform = Transform(translation=d["translation"], scale_factors=d["scale"])
    #     set_transform(largest_image, transform)
    #     images["image"] = largest_image
    #     ##
    #     sdata = SpatialData(adata=feature_table, regions=regions, images=images, points=points)
    #     return sdata

    def __repr__(self) -> str:
        def repr_regions(regions: Any) -> str:
            return f"regions with n_obs x n_vars = {regions.n_obs} x {regions.n_vars}"

        def repr_image(ar: xr.DataArray) -> str:
            return f"image with shape {ar.shape}"

        def h(s: str) -> str:
            return hashlib.md5(repr(s).encode()).hexdigest()

        descr = f"SpatialData object with "  # noqa: F541
        if self.tables is not None:
            descr += f"n_obs x n_vars = {self.tables.n_obs} x {self.tables.n_vars}"
        else:
            descr += "no feature table"
        n = 0
        for attr in [
            "regions",
            "images",
        ]:
            attribute = getattr(self, attr)
            keys = attribute.keys()
            if len(keys) > 0:
                n = 0
                descr += f"\n{h('level0')}{attr}: {str(list(keys))[1:-1]}"
                repr_function = {"regions": repr_regions, "images": repr_image}[attr]
                for key in keys:
                    descr += f"\n{h('level1.0')}{h(attr + 'level1.1')}'{key}': {repr_function(attribute[key])}"
                    n += 1
                descr += f"{h('empty_line')}"

        if self.points is not None:
            n = 1
            # descr += f"\n{h('level0')}points with n_obs x n_vars = {self.points.n_obs} x {self.points.n_vars}" # TODO: returns error atm.
            descr += f"{h('empty_line') + h('level1.0')}"

        def rreplace(s: str, old: str, new: str, occurrence: int) -> str:
            li = s.rsplit(old, occurrence)
            return new.join(li)

        descr = rreplace(descr, h("empty_line"), "", 1)
        descr = descr.replace(h("empty_line"), "\n│ ")

        descr = rreplace(descr, h("level0"), "└── ", 1)
        descr = descr.replace(h("level0"), "├── ")

        for attr in ["regions", "images"]:
            descr = rreplace(descr, h(attr + "level1.1"), "└── ", 1)
            descr = descr.replace(h(attr + "level1.1"), "├── ")

        descr = rreplace(descr, h("level1.0"), "    ", n)
        descr = descr.replace(h("level1.0"), "│   ")
        return descr

    def __eq__(self, other: Any) -> bool:
        # new comparison: dumping everything to zarr and comparing bytewise
        with tempfile.TemporaryDirectory() as tmpdir:
            self.write(os.path.join(tmpdir, "self.zarr"))
            other.write(os.path.join(tmpdir, "other.zarr"))
            return are_directories_identical(os.path.join(tmpdir, "self.zarr"), os.path.join(tmpdir, "other.zarr"))
        # old comparison: comparing piece by piece
        # if not isinstance(other, SpatialData):
        #     return False
        #
        # def both_none(a, b):
        #     return a is None and b is None
        #
        # def any_none(a, b):
        #     return a is None or b is None
        #
        # def check_with_default_none(a, b):
        #     if both_none(a, b):
        #         return True
        #     if any_none(a, b):
        #         return False
        #     return a == b
        #
        # for attr in ["adata", "points"]:
        #     if not check_with_default_none(getattr(self, attr), getattr(other, attr)):
        #         return False
        #
        # for attr in ["regions", "images"]:
        #     if not getattr(self, attr) == getattr(other, attr):
        #         return False
        # return True


def _validate_dataset(
    dataset: Optional[Mapping[str, Any]] = None, dataset_transform: Optional[Mapping[str, Any]] = None
) -> Tuple[Optional[Mapping[str, Any]], Optional[Mapping[str, Any]], Optional[Set[str]]]:
    if dataset is None or not len(dataset):
        return None, None, None
    if isinstance(dataset, dict):
        if dataset_transform is None:
            dataset_transform = {k: Transform(ndim=2) for k in dataset.keys()}
        for k, v in dataset.items():  # TODO: this is probably wrong, it should validate keys
            if k not in dataset_transform:
                dataset_transform[k] = get_transform(v)  # type: ignore [index]
        assert set(dataset.keys()).issuperset(set(dataset_transform.keys())), "TODO: superset check."
        return dataset, dataset_transform, set(dataset.keys())
    raise ValueError(f"`dataset` must be a `dict`, not `{type(dataset)}`.")
