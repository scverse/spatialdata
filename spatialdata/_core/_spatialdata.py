from __future__ import annotations

import hashlib
from types import MappingProxyType
from typing import Dict, Generator, List, Mapping, Optional, Union

import pyarrow as pa
import zarr
from anndata import AnnData
from geopandas import GeoDataFrame
from multiscale_spatial_image.multiscale_spatial_image import MultiscaleSpatialImage
from ome_zarr.io import parse_url
from ome_zarr.types import JSONDict
from spatial_image import SpatialImage

from spatialdata._core._spatial_query import (
    BaseSpatialRequest,
    BoundingBoxRequest,
    _bounding_box_query_points,
)
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
        Mapping of 2D and 3D image elements. The following parsers are available: :class:`~spatialdata.Image2DModel`,
        :class:`~spatialdata.Image3DModel`.
    labels
        Mapping of 2D and 3D labels elements. Labels are regions, they can't contain annotation, but they can be
        annotated by a table. The following parsers are available: :class:`~spatialdata.Labels2DModel`,
        :class:`~spatialdata.Labels3DModel`.
    points
        Mapping of points elements. Points can contain annotations. The following parsers is available:
        :class:`~spatialdata.PointsModel`.
    polygons
        Mapping of 2D polygons elements. They can't contain annotation but they can be annotated
        by a table. The following parsers is available: :class:`~spatialdata.PolygonsModel`.
    shapes
        Mapping of 2D shapes elements (circles, squares). Shapes are regions, they can't contain annotation but they
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

    _images: Mapping[str, Union[SpatialImage, MultiscaleSpatialImage]] = MappingProxyType({})
    _labels: Mapping[str, Union[SpatialImage, MultiscaleSpatialImage]] = MappingProxyType({})
    _points: Mapping[str, pa.Table] = MappingProxyType({})
    _polygons: Mapping[str, GeoDataFrame] = MappingProxyType({})
    _shapes: Mapping[str, AnnData] = MappingProxyType({})
    _table: Optional[AnnData] = None

    def __init__(
        self,
        images: Mapping[str, Union[SpatialImage, MultiscaleSpatialImage]] = MappingProxyType({}),
        labels: Mapping[str, Union[SpatialImage, MultiscaleSpatialImage]] = MappingProxyType({}),
        points: Mapping[str, pa.Table] = MappingProxyType({}),
        polygons: Mapping[str, GeoDataFrame] = MappingProxyType({}),
        shapes: Mapping[str, AnnData] = MappingProxyType({}),
        table: Optional[AnnData] = None,
    ) -> None:

        if images is not None:
            self._images: Dict[str, Union[SpatialImage, MultiscaleSpatialImage]] = {}
            for k, v in images.items():
                ndim = len(get_dims(v))
                if ndim == 3:
                    Image2D_s.validate(v)
                    self._images[k] = v
                elif ndim == 4:
                    Image3D_s.validate(v)
                    self._images[k] = v
                else:
                    raise ValueError("Only czyx and cyx images supported")

        if labels is not None:
            self._labels: Dict[str, Union[SpatialImage, MultiscaleSpatialImage]] = {}
            for k, v in labels.items():
                ndim = len(get_dims(v))
                if ndim == 2:
                    Label2D_s.validate(v)
                    self._labels[k] = v
                elif ndim == 3:
                    Label3D_s.validate(v)
                    self._labels[k] = v
                else:
                    raise ValueError(f"Invalid label dimensions: {ndim}")

        if polygons is not None:
            self._polygons: Dict[str, GeoDataFrame] = {}
            for k, v in polygons.items():
                Polygon_s.validate(v)
                self._polygons[k] = v

        if shapes is not None:
            self._shapes: Dict[str, AnnData] = {}
            for k, v in shapes.items():
                Shape_s.validate(v)
                self._shapes[k] = v

        if points is not None:
            self._points: Dict[str, pa.Table] = {}
            for k, v in points.items():
                Point_s.validate(v)
                self._points[k] = v

        if table is not None:
            Table_s.validate(table)
            self._table = table

        self._spatial_query = SpatialQueryManager(self)

    @property
    def spatial(self) -> SpatialQueryManager:
        return self._spatial_query

    def write(
        self,
        file_path: str,
        storage_options: Optional[Union[JSONDict, List[JSONDict]]] = None,
    ) -> None:
        """Write to Zarr file."""

        store = parse_url(file_path, mode="w").store
        root = zarr.group(store=store)

        if len(self.images):
            elem_group = root.create_group(name="images")
            for el in self.images.keys():
                write_image(
                    image=self.images[el],
                    group=elem_group,
                    name=el,
                    storage_options=storage_options,
                )
        if len(self.labels):
            # no need to create group handled by ome_zarr
            for el in self.labels.keys():
                write_labels(
                    labels=self.labels[el],
                    group=root,
                    name=el,
                    storage_options=storage_options,
                )
        if len(self.points):
            elem_group = root.create_group(name="points")
            for el in self.points.keys():
                write_points(
                    points=self.points[el],
                    group=elem_group,
                    name=el,
                )
        if len(self.polygons):
            elem_group = root.create_group(name="polygons")
            for el in self.polygons.keys():
                write_polygons(
                    polygons=self.polygons[el],
                    group=elem_group,
                    name=el,
                )
        if len(self.shapes):
            elem_group = root.create_group(name="shapes")
            for el in self.shapes.keys():
                write_shapes(
                    shapes=self.shapes[el],
                    group=elem_group,
                    name=el,
                )
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
    def images(self) -> Mapping[str, Union[SpatialImage, MultiscaleSpatialImage]]:
        """Return images as a mapping of name to image data."""
        return self._images

    @property
    def labels(self) -> Mapping[str, Union[SpatialImage, MultiscaleSpatialImage]]:
        """Return labels as a mapping of name to label data."""
        return self._labels

    @property
    def points(self) -> Mapping[str, pa.Table]:
        """Return points as a mapping of name to point data."""
        return self._points

    @property
    def polygons(self) -> Mapping[str, GeoDataFrame]:
        """Return polygons as a mapping of name to polygon data."""
        return self._polygons

    @property
    def shapes(self) -> Mapping[str, AnnData]:
        """Return shapes as a mapping of name to shape data."""
        return self._shapes

    def _non_empty_elements(self) -> List[str]:
        """Get the names of the elements that are not empty.

        Returns
        -------
        non_empty_elements : List[str]
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


class SpatialQueryManager:
    """Perform spatial queries on SpatialData objects"""

    def __init__(self, sdata: SpatialData):
        self._sdata = sdata

    def bounding_box(self, request: BoundingBoxRequest) -> SpatialData:
        """Perform a bounding box query on the SpatialData object.

        Parameters
        ----------
        request : BoundingBoxRequest
            The bounding box request.

        Returns
        -------
        requested_sdata : SpatialData
            The SpatialData object containing the requested data.
            Elements with no valid data are omitted.
        """
        requested_points = {}
        for points_name, points_data in self._sdata.points.items():
            points = _bounding_box_query_points(points_data, request)
            if len(points) > 0:
                # do not include elements with no data
                requested_points[points_name] = points

        return SpatialData(points=requested_points)

    def __call__(self, request: BaseSpatialRequest) -> SpatialData:
        if isinstance(request, BoundingBoxRequest):
            return self.bounding_box(request)
        else:
            raise TypeError("unknown request type")
