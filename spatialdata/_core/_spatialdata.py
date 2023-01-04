from __future__ import annotations

import hashlib
from collections.abc import Generator
from types import MappingProxyType
from typing import Optional, Union

import pyarrow as pa
import zarr
from anndata import AnnData
from geopandas import GeoDataFrame
from multiscale_spatial_image.multiscale_spatial_image import MultiscaleSpatialImage
from ome_zarr.io import parse_url
from ome_zarr.types import JSONDict
from spatial_image import SpatialImage

from spatialdata._core.core_utils import SpatialElement, get_dims
from spatialdata._core.models import (
    Image2DModel,
    Image3DModel,
    Labels2DModel,
    Labels3DModel,
    PointsModel,
    PolygonsModel,
    ShapesModel,
    TableModel,
)
from spatialdata._io.write import (
    write_image,
    write_labels,
    write_points,
    write_polygons,
    write_shapes,
    write_table,
)
from spatialdata._logging import logger

# schema for elements
Label2D_s = Labels2DModel()
Label3D_s = Labels3DModel()
Image2D_s = Image2DModel()
Image3D_s = Image3DModel()
Polygon_s = PolygonsModel()
Point_s = PointsModel()
Shape_s = ShapesModel()
Table_s = TableModel()


class SpatialData:
    """
    The SpatialData object.

    The SpatialData object is a modular container for arbitrary combinations of spatial elements. The elements
    can be accesses separately and are stored as standard types (:class:`anndata.AnnData`,
    :class:`geopandas.GeoDataFrame`, :class:`xarray.DataArray`).


    Parameters
    ----------
    images
        Dict of 2D and 3D image elements. The following parsers are available: :class:`~spatialdata.Image2DModel`,
        :class:`~spatialdata.Image3DModel`.
    labels
        Dict of 2D and 3D labels elements. Labels are regions, they can't contain annotation, but they can be
        annotated by a table. The following parsers are available: :class:`~spatialdata.Labels2DModel`,
        :class:`~spatialdata.Labels3DModel`.
    points
        Dict of points elements. Points can contain annotations. The following parsers is available:
        :class:`~spatialdata.PointsModel`.
    polygons
        Dict of 2D polygons elements. They can't contain annotation but they can be annotated
        by a table. The following parsers is available: :class:`~spatialdata.PolygonsModel`.
    shapes
        Dict of 2D shapes elements (circles, squares). Shapes are regions, they can't contain annotation but they
        can be annotated by a table. The following parsers is available: :class:`~spatialdata.ShapesModel`.
    table
        AnnData table containing annotations for regions (labels, polygons, shapes). The following parsers is
        available: :class:`~spatialdata.TableModel`.

    Notes
    -----
    The spatial elements are stored with standard types:

        - images and labels are stored as :class:`spatial_image.SpatialImage` or :class:`multiscale_spatial_image.MultiscaleSpatialImage` objects, which are respectively equivalent to :class:`xarray.DataArray` and to a :class:`datatree.DataTree` of :class:`xarray.DataArray` objects.
        - points and shapes are stored as :class:`anndata.AnnData` objects, with the spatial coordinates stored in the obsm slot.
        - polygons are stored as :class:`geopandas.GeoDataFrame`.
        - the table are stored as :class:`anndata.AnnData` objects, with the spatial coordinates stored in the obsm slot.

    The table can annotate regions (shapes, polygons or labels) and can be used to store additional information.
    Points are not regions but 0-dimensional locations. They can't be annotated by a table, but they can store
    annotation directly.

    The elements need to pass a validation step. To construct valid elements you can use the parsers that we
    provide (:class:`~spatialdata.Image2DModel`, :class:`~spatialdata.Image3DModel`, :class:`~spatialdata.Labels2DModel`, :class:`~spatialdata.Labels3DModel`, :class:`~spatialdata.PointsModel`, :class:`~spatialdata.PolygonsModel`, :class:`~spatialdata.ShapesModel`, :class:`~spatialdata.TableModel`).
    """

    _images: dict[str, Union[SpatialImage, MultiscaleSpatialImage]] = MappingProxyType({})  # type: ignore[assignment]
    _labels: dict[str, Union[SpatialImage, MultiscaleSpatialImage]] = MappingProxyType({})  # type: ignore[assignment]
    _points: dict[str, pa.Table] = MappingProxyType({})  # type: ignore[assignment]
    _polygons: dict[str, GeoDataFrame] = MappingProxyType({})  # type: ignore[assignment]
    _shapes: dict[str, AnnData] = MappingProxyType({})  # type: ignore[assignment]
    _table: Optional[AnnData] = None
    path: Optional[str] = None

    def __init__(
        self,
        images: dict[str, Union[SpatialImage, MultiscaleSpatialImage]] = MappingProxyType({}),  # type: ignore[assignment]
        labels: dict[str, Union[SpatialImage, MultiscaleSpatialImage]] = MappingProxyType({}),  # type: ignore[assignment]
        points: dict[str, pa.Table] = MappingProxyType({}),  # type: ignore[assignment]
        polygons: dict[str, GeoDataFrame] = MappingProxyType({}),  # type: ignore[assignment]
        shapes: dict[str, AnnData] = MappingProxyType({}),  # type: ignore[assignment]
        table: Optional[AnnData] = None,
    ) -> None:
        self.path = None
        if images is not None:
            self._images: dict[str, Union[SpatialImage, MultiscaleSpatialImage]] = {}
            for k, v in images.items():
                self._add_image_in_memory(name=k, image=v)

        if labels is not None:
            self._labels: dict[str, Union[SpatialImage, MultiscaleSpatialImage]] = {}
            for k, v in labels.items():
                self._add_labels_in_memory(name=k, labels=v)

        if polygons is not None:
            self._polygons: dict[str, GeoDataFrame] = {}
            for k, v in polygons.items():
                self._add_polygons_in_memory(name=k, polygons=v)

        if shapes is not None:
            self._shapes: dict[str, AnnData] = {}
            for k, v in shapes.items():
                self._add_shapes_in_memory(name=k, shapes=v)

        if points is not None:
            self._points: dict[str, pa.Table] = {}
            for k, v in points.items():
                self._add_points_in_memory(name=k, points=v)

        if table is not None:
            Table_s.validate(table)
            self._table = table

    def _add_image_in_memory(self, name: str, image: Union[SpatialImage, MultiscaleSpatialImage]) -> None:
        ndim = len(get_dims(image))
        if ndim == 3:
            Image2D_s.validate(image)
            self._images[name] = image
        elif ndim == 4:
            Image3D_s.validate(image)
            self._images[name] = image
        else:
            raise ValueError("Only czyx and cyx images supported")

    def _add_labels_in_memory(self, name: str, labels: Union[SpatialImage, MultiscaleSpatialImage]) -> None:
        ndim = len(get_dims(labels))
        if ndim == 2:
            Label2D_s.validate(labels)
            self._labels[name] = labels
        elif ndim == 3:
            Label3D_s.validate(labels)
            self._labels[name] = labels
        else:
            raise ValueError(f"Only yx and zyx labels supported, got {ndim} dimensions")

    def _add_polygons_in_memory(self, name: str, polygons: GeoDataFrame) -> None:
        Polygon_s.validate(polygons)
        self._polygons[name] = polygons

    def _add_shapes_in_memory(self, name: str, shapes: AnnData) -> None:
        Shape_s.validate(shapes)
        self._shapes[name] = shapes

    def _add_points_in_memory(self, name: str, points: pa.Table) -> None:
        Point_s.validate(points)
        self._points[name] = points

    def is_backed(self) -> bool:
        """Check if the data is backed by a Zarr storage or it is in-memory."""
        return self.path is not None

    def _init_add_element(self, name: str, element_type: str, overwrite: bool) -> zarr.Group:
        if self.path is None:
            # in the future we can relax this, but this ensures that we don't have objects that are partially backed
            # and partially in memory
            raise RuntimeError(
                "The data is not backed by a Zarr storage. In order to add new elements after "
                "initializing a SpatialData object you need to call SpatialData.write() first"
            )
        store = parse_url(self.path, mode="r+").store
        root = zarr.group(store=store)
        assert element_type in ["images", "labels", "points", "polygons", "shapes"]
        # not need to create the group for labels as it is already handled by ome-zarr-py
        if element_type != "labels":
            if element_type not in root:
                elem_group = root.create_group(name=element_type)
            else:
                elem_group = root[element_type]
        if overwrite:
            if element_type == "labels":
                if element_type in root:
                    elem_group = root[element_type]
            if name in elem_group:
                del elem_group[name]
        else:
            # bypass is to ensure that elem_group is defined. I don't want to define it as None but either having it
            # or not having it, so if the code tries to access it and it should not be there, it will raise an error
            bypass = False
            if element_type == "labels":
                if element_type in root:
                    elem_group = root[element_type]
                else:
                    bypass = True
            if not bypass:
                if name in elem_group:
                    raise ValueError(f"Element {name} already exists, use overwrite=True to overwrite it")

        if element_type != "labels":
            return elem_group
        else:
            return root

    def add_image(
        self,
        name: str,
        image: Union[SpatialImage, MultiscaleSpatialImage],
        storage_options: Optional[Union[JSONDict, list[JSONDict]]] = None,
        overwrite: bool = False,
    ) -> None:
        """
        Add an image to the SpatialData object.

        Parameters
        ----------
        name
            Key to the element inside the SpatialData object.
        image
            The image to add, the object needs to pass validation (see :class:`~spatialdata.Image2DModel` and :class:`~spatialdata.Image3DModel`).
        storage_options
            Storage options for the Zarr storage.
            See https://zarr.readthedocs.io/en/stable/api/storage.html for more details.
        overwrite
            If True, overwrite the element if it already exists.

        Notes
        -----
        If the SpatialData object is backed by a Zarr storage, the image will be written to the Zarr storage.
        """
        # _init_add_element() needs to be called before _add_image_in_memory(), and same for the other elements
        # otherwise if a element is added and saved to disk, and another element with the same name but different
        # content is added, _init_add_element() will raise an exception. If _add_image_in_memory() is called first,
        # then the memory content will be overwritten, but the disk content will remain the old one
        elem_group = self._init_add_element(name=name, element_type="images", overwrite=overwrite)
        if name not in self.images:
            self._add_image_in_memory(name=name, image=image)
        write_image(
            image=self.images[name],
            group=elem_group,
            name=name,
            storage_options=storage_options,
        )

    def add_labels(
        self,
        name: str,
        labels: Union[SpatialImage, MultiscaleSpatialImage],
        storage_options: Optional[Union[JSONDict, list[JSONDict]]] = None,
        overwrite: bool = False,
    ) -> None:
        """
        Add labels to the SpatialData object.

        Parameters
        ----------
        name
            Key to the element inside the SpatialData object.
        labels
            The labels (masks) to add, the object needs to pass validation (see :class:`~spatialdata.Labels2DModel` and :class:`~spatialdata.Labels3DModel`).
        storage_options
            Storage options for the Zarr storage.
            See https://zarr.readthedocs.io/en/stable/api/storage.html for more details.
        overwrite
            If True, overwrite the element if it already exists.

        Notes
        -----
        If the SpatialData object is backed by a Zarr storage, the image will be written to the Zarr storage.
        """
        elem_group = self._init_add_element(name=name, element_type="labels", overwrite=overwrite)
        if name not in self.labels:
            self._add_labels_in_memory(name=name, labels=labels)
        write_labels(
            labels=self.labels[name],
            group=elem_group,
            name=name,
            storage_options=storage_options,
        )

    def add_points(
        self,
        name: str,
        points: pa.Table,
        overwrite: bool = False,
    ) -> None:
        """
        Add points to the SpatialData object.

        Parameters
        ----------
        name
            Key to the element inside the SpatialData object.
        points
            The points to add, the object needs to pass validation (see :class:`~spatialdata.PointsModel`).
        storage_options
            Storage options for the Zarr storage.
            See https://zarr.readthedocs.io/en/stable/api/storage.html for more details.
        overwrite
            If True, overwrite the element if it already exists.

        Notes
        -----
        If the SpatialData object is backed by a Zarr storage, the image will be written to the Zarr storage.
        """
        elem_group = self._init_add_element(name=name, element_type="points", overwrite=overwrite)
        if name not in self.points:
            self._add_points_in_memory(name=name, points=points)
        write_points(
            points=self.points[name],
            group=elem_group,
            name=name,
        )

    def add_polygons(
        self,
        name: str,
        polygons: GeoDataFrame,
        overwrite: bool = False,
    ) -> None:
        """
        Add polygons to the SpatialData object.

        Parameters
        ----------
        name
            Key to the element inside the SpatialData object.
        polygons
            The polygons to add, the object needs to pass validation (see :class:`~spatialdata.PolygonsModel`).
        storage_options
            Storage options for the Zarr storage.
            See https://zarr.readthedocs.io/en/stable/api/storage.html for more details.
        overwrite
            If True, overwrite the element if it already exists.

        Notes
        -----
        If the SpatialData object is backed by a Zarr storage, the image will be written to the Zarr storage.
        """
        elem_group = self._init_add_element(name=name, element_type="polygons", overwrite=overwrite)
        if name not in self.polygons:
            self._add_polygons_in_memory(name=name, polygons=polygons)
        write_polygons(
            polygons=self.polygons[name],
            group=elem_group,
            name=name,
        )

    def add_shapes(
        self,
        name: str,
        shapes: AnnData,
        overwrite: bool = False,
    ) -> None:
        """
        Add shapes to the SpatialData object.

        Parameters
        ----------
        name
            Key to the element inside the SpatialData object.
        shapes
            The shapes to add, the object needs to pass validation (see :class:`~spatialdata.ShapesModel`).
        storage_options
            Storage options for the Zarr storage.
            See https://zarr.readthedocs.io/en/stable/api/storage.html for more details.
        overwrite
            If True, overwrite the element if it already exists.

        Notes
        -----
        If the SpatialData object is backed by a Zarr storage, the image will be written to the Zarr storage.
        """
        elem_group = self._init_add_element(name=name, element_type="shapes", overwrite=overwrite)
        if name not in self.shapes:
            self._add_shapes_in_memory(name=name, shapes=shapes)
        write_shapes(
            shapes=self.shapes[name],
            group=elem_group,
            name=name,
        )

    def write(
        self, file_path: str, storage_options: Optional[Union[JSONDict, list[JSONDict]]] = None, overwrite: bool = False
    ) -> None:
        """Write the SpatialData object to Zarr."""

        if self.path == file_path:
            raise ValueError("Can't overwrite the original file")
        elif self.path != file_path and self.path is not None:
            logger.info(f"The Zarr file used for backing will now change from {self.path} to {file_path}")
        self.path = file_path

        if not overwrite and parse_url(self.path, mode="r") is not None:
            raise ValueError("The Zarr store already exists. Use overwrite=True to overwrite the store.")
        else:
            store = parse_url(self.path, mode="w").store
            root = zarr.group(store=store)
            store.close()

        if len(self.images):
            elem_group = root.create_group(name="images")
            for el in self.images.keys():
                self.add_image(name=el, image=self.images[el], storage_options=storage_options)

        if len(self.labels):
            elem_group = root.create_group(name="labels")
            for el in self.labels.keys():
                self.add_labels(name=el, labels=self.labels[el], storage_options=storage_options)

        if len(self.points):
            elem_group = root.create_group(name="points")
            for el in self.points.keys():
                self.add_points(name=el, points=self.points[el])

        if len(self.polygons):
            elem_group = root.create_group(name="polygons")
            for el in self.polygons.keys():
                self.add_polygons(name=el, polygons=self.polygons[el])

        if len(self.shapes):
            elem_group = root.create_group(name="shapes")
            for el in self.shapes.keys():
                self.add_shapes(name=el, shapes=self.shapes[el])

        if self.table is not None:
            elem_group = root.create_group(name="table")
            write_table(table=self.table, group=elem_group, name="table")

    @property
    def table(self) -> AnnData:
        return self._table

    @staticmethod
    def read(file_path: str) -> SpatialData:
        from spatialdata._io.read import read_zarr

        sdata = read_zarr(file_path)
        return sdata

    @property
    def images(self) -> dict[str, Union[SpatialImage, MultiscaleSpatialImage]]:
        """Return images as a Dict of name to image data."""
        return self._images

    @property
    def labels(self) -> dict[str, Union[SpatialImage, MultiscaleSpatialImage]]:
        """Return labels as a Dict of name to label data."""
        return self._labels

    @property
    def points(self) -> dict[str, pa.Table]:
        """Return points as a Dict of name to point data."""
        return self._points

    @property
    def polygons(self) -> dict[str, GeoDataFrame]:
        """Return polygons as a Dict of name to polygon data."""
        return self._polygons

    @property
    def shapes(self) -> dict[str, AnnData]:
        """Return shapes as a Dict of name to shape data."""
        return self._shapes

    def _non_empty_elements(self) -> list[str]:
        """Get the names of the elements that are not empty.

        Returns
        -------
        non_empty_elements
            The names of the elements that are not empty.
        """
        all_elements = ["images", "labels", "points", "polygons", "shapes", "table"]
        return [
            element
            for element in all_elements
            if (getattr(self, element) is not None) and (len(getattr(self, element)) > 0)
        ]

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

        non_empty_elements = self._non_empty_elements()
        last_element_index = len(non_empty_elements) - 1
        for attr_index, attr in enumerate(non_empty_elements):
            last_attr = True if (attr_index == last_element_index) else False
            attribute = getattr(self, attr)

            descr += f"\n{h('level0')}{attr.capitalize()}"
            if isinstance(attribute, AnnData):
                descr += f"{h('empty_line')}"
                descr_class = attribute.__class__.__name__
                descr += f"{h('level1.0')}'{attribute}': {descr_class} {attribute.shape}"
                descr = rreplace(descr, h("level1.0"), "    └── ", 1)
            else:
                for k, v in attribute.items():
                    descr += f"{h('empty_line')}"
                    descr_class = v.__class__.__name__
                    if attr == "shapes":
                        descr += (
                            f"{h(attr + 'level1.1')}'{k}': {descr_class} with osbm.spatial "
                            f"{v.obsm['spatial'].shape}"
                        )
                    elif attr == "polygons":
                        descr += f"{h(attr + 'level1.1')}'{k}': {descr_class} " f"shape: {v.shape} (2D polygons)"
                    elif attr == "points":
                        if len(v) > 0:
                            n = len(get_dims(v))
                            dim_string = f"({n}D points)"
                        else:
                            dim_string = ""
                        if descr_class == "Table":
                            descr_class = "pyarrow.Table"
                        descr += f"{h(attr + 'level1.1')}'{k}': {descr_class} " f"shape: {v.shape} {dim_string}"
                    else:
                        if isinstance(v, SpatialImage):
                            descr += f"{h(attr + 'level1.1')}'{k}': {descr_class}[{''.join(v.dims)}] {v.shape}"
                        elif isinstance(v, MultiscaleSpatialImage):
                            shapes = []
                            dims: Optional[str] = None
                            for pyramid_level in v.keys():
                                dataset_names = list(v[pyramid_level].keys())
                                assert len(dataset_names) == 1
                                dataset_name = dataset_names[0]
                                vv = v[pyramid_level][dataset_name]
                                shape = vv.shape
                                if dims is None:
                                    dims = "".join(vv.dims)
                                shapes.append(shape)
                            descr += (
                                f"{h(attr + 'level1.1')}'{k}': {descr_class}[{dims}] " f"{', '.join(map(str, shapes))}"
                            )
                        else:
                            raise TypeError(f"Unknown type {type(v)}")
            if last_attr is True:
                descr = descr.replace(h("empty_line"), "\n  ")
            else:
                descr = descr.replace(h("empty_line"), "\n│ ")

        descr = rreplace(descr, h("level0"), "└── ", 1)
        descr = descr.replace(h("level0"), "├── ")

        for attr in ["images", "labels", "points", "polygons", "table", "shapes"]:
            descr = rreplace(descr, h(attr + "level1.1"), "    └── ", 1)
            descr = descr.replace(h(attr + "level1.1"), "    ├── ")
        return descr

    def _gen_elements(self) -> Generator[SpatialElement, None, None]:
        for element_type in ["images", "labels", "points", "polygons", "shapes"]:
            d = getattr(SpatialData, element_type).fget(self)
            yield from d.values()
