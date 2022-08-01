import hashlib
import os

# import colorama
# colorama.init(strip=False)
import tempfile
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping, Optional

import numpy as np
import xarray as xr
from anndata import AnnData
from anndata._io import read_zarr
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader

from spatialdata._core.mixin.io_mixin import IoMixin

# from spatialdata._core.writer import write_spatial_anndata
from spatialdata._core.transform import Transform, get_transform, set_transform
from spatialdata.utils import are_directories_identical


class SpatialData(IoMixin):
    """Spatial data structure."""

    adata: AnnData
    regions: Mapping[str, Any]
    images: Mapping[str, Any]
    points: Optional[AnnData]

    def __init__(
        self,
        adata: Optional[AnnData] = None,
        regions: Mapping[str, Any] = MappingProxyType({}),
        images: Mapping[str, Any] = MappingProxyType({}),
        images_transform: Optional[Mapping[str, Any]] = None,
        points: Optional[AnnData] = None,
    ) -> None:
        # current limitations:
        # - only 2d data
        # - only first table, image, regions of the dict are used
        self.adata = adata
        self.regions = dict(regions)
        if images_transform is None:
            images_transform = {k: Transform(ndim=2) for k in images}
        assert set(images.keys()).issuperset(set(images_transform.keys()))
        for k, v in images.items():
            if TYPE_CHECKING:
                assert isinstance(images_transform, dict)
            images_transform[k] = get_transform(v)
        self.images = {
            k: self.parse_image(image, image_transform)
            for k, (image, image_transform) in zip(images.keys(), zip(images.values(), images_transform.values()))
        }
        self.points = points

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
    def from_zarr(self, file_path: str) -> "SpatialData":
        """Load from Zarr file."""

        ome_zarr = parse_url(file_path)
        reader = Reader(ome_zarr)
        ome_zarr.__dir__()

        feature_table = None
        regions = {}
        images = {}
        points = None

        ##
        import os

        groups = os.listdir(ome_zarr.path)
        if "tables" in groups:
            tables = os.listdir(os.path.join(ome_zarr.path, "tables"))
            if len(tables) > 2:
                raise RuntimeError("Currently it is possible to work with only one feature table")
            for table in tables:
                if table == ".zgroup":
                    continue
                else:
                    table_path = os.path.join(ome_zarr.path, "tables", table)
                    feature_table = read_zarr(table_path)
        if "circles" in groups:
            tables = os.listdir(os.path.join(ome_zarr.path, "circles"))
            for table in tables:
                if table == ".zgroup":
                    continue
                else:
                    table_path = os.path.join(ome_zarr.path, "circles", table)
                    adata = read_zarr(table_path)
                    regions[table] = adata
        if "points" in groups:
            tables = os.listdir(os.path.join(ome_zarr.path, "points"))
            if len(tables) > 2:
                raise RuntimeError("Currently it is possible to work with only one points table")
            for table in tables:
                if table == ".zgroup":
                    continue
                else:
                    table_path = os.path.join(ome_zarr.path, "points", table)
                    points = read_zarr(table_path)
        ##
        content = [node for node in reader()]  # noqa: C416
        assert len(content) == 1
        node = content[0]
        data = node.data
        metadata = node.metadata
        # ignoring pyramidal information for the moment
        largest_image = xr.DataArray(data[0]).load()
        largest_image = largest_image.transpose()
        largest_image_transform = metadata["coordinateTransformations"][0]
        d = {}
        for e in largest_image_transform:
            d[e["type"]] = np.flip(e[e["type"]])
        transform = Transform(translation=d["translation"], scale_factors=d["scale"])
        set_transform(largest_image, transform)
        images["image"] = largest_image
        ##
        sdata = SpatialData(adata=feature_table, regions=regions, images=images, points=points)
        return sdata

    def to_zarr(self, file_path: str) -> None:
        """Save to Zarr file."""
        if len(self.images) == 0:
            pass
        else:
            # simple case for the moment
            assert len(self.images) == 1
            transform = get_transform(self.images.values().__iter__().__next__())
            transform.translation
            transform.scale_factors

            self.images.values().__iter__().__next__().to_numpy()

        if len(self.regions) == 0:
            pass
        else:
            # simple case for the moment
            assert len(self.regions) == 1
            self.regions.values().__iter__().__next__()
            # regions_name = self.regions.keys().__iter__().__next__()

        # write_spatial_anndata(
        #     file_path=file_path,
        #     image=img,
        #     image_axes=["y", "x"],
        #     image_translation=image_translation,
        #     image_scale_factors=image_scale_factors,
        #     tables_adata=self.adata,
        #     tables_region="circles/circles_table",
        #     # tables_region_key=regions_name,
        #     # tables_instance_key=None,
        #     circles_adata=regions,
        #     points_adata=self.points,
        # )

    def __repr__(self) -> str:
        def repr_regions(regions: Any) -> str:
            return f"regions with n_obs x n_vars = {regions.n_obs} x {regions.n_vars}"

        def repr_image(ar: xr.DataArray) -> str:
            return f"image with shape {ar.shape}"

        def h(s: str) -> str:
            return hashlib.md5(repr(s).encode()).hexdigest()

        descr = f"SpatialData object with "  # noqa: F541
        if self.adata is not None:
            descr += f"n_obs x n_vars = {self.adata.n_obs} x {self.adata.n_vars}"
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
            descr += f"\n{h('level0')}points with n_obs x n_vars = {self.points.n_obs} x {self.points.n_vars}"
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
            self.to_zarr(os.path.join(tmpdir, "self.zarr"))
            other.to_zarr(os.path.join(tmpdir, "other.zarr"))
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
