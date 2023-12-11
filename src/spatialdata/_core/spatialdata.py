from __future__ import annotations

import hashlib
import os
import warnings
from collections.abc import Generator
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Union

import pandas as pd
import zarr
from anndata import AnnData
from dask.dataframe import read_parquet
from dask.dataframe.core import DataFrame as DaskDataFrame
from dask.delayed import Delayed
from geopandas import GeoDataFrame
from multiscale_spatial_image.multiscale_spatial_image import MultiscaleSpatialImage
from ome_zarr.io import parse_url
from ome_zarr.types import JSONDict
from spatial_image import SpatialImage

from spatialdata._io import (
    write_image,
    write_labels,
    write_points,
    write_shapes,
    write_table,
)
from spatialdata._io._utils import get_backing_files
from spatialdata._logging import logger
from spatialdata._types import ArrayLike
from spatialdata._utils import _natural_keys, deprecation_alias
from spatialdata.models import (
    Image2DModel,
    Image3DModel,
    Labels2DModel,
    Labels3DModel,
    PointsModel,
    ShapesModel,
    SpatialElement,
    TableModel,
    get_axes_names,
    get_model,
)
from spatialdata.models.models import check_target_region_column_symmetry, get_table_keys

if TYPE_CHECKING:
    from spatialdata._core.query.spatial_query import BaseSpatialRequest

# schema for elements
Label2D_s = Labels2DModel()
Label3D_s = Labels3DModel()
Image2D_s = Image2DModel()
Image3D_s = Image3DModel()
Shape_s = ShapesModel()
Point_s = PointsModel()
Table_s = TableModel()

# create a shorthand for raster image types
Raster_T = Union[SpatialImage, MultiscaleSpatialImage]


class SpatialData:
    """
    The SpatialData object.

    The SpatialData object is a modular container for arbitrary combinations of SpatialElements. The elements
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
    shapes
        Dict of 2D shapes elements (circles, polygons, multipolygons).
        Shapes are regions, they can't contain annotation, but they can be annotated by a table.
        The following parsers are available: :class:`~spatialdata.ShapesModel`.
    table
        AnnData table containing annotations for regions (labels and shapes). The following parsers is
        available: :class:`~spatialdata.TableModel`.

    Notes
    -----
    The SpatialElements are stored with standard types:

        - images and labels are stored as :class:`spatial_image.SpatialImage` or
            :class:`multiscale_spatial_image.MultiscaleSpatialImage` objects, which are respectively equivalent to
            :class:`xarray.DataArray` and to a :class:`datatree.DataTree` of :class:`xarray.DataArray` objects.
        - points are stored as :class:`dask.dataframe.DataFrame` objects.
        - shapes are stored as :class:`geopandas.GeoDataFrame`.
        - the table are stored as :class:`anndata.AnnData` objects,  with the spatial coordinates stored in the obsm
            slot.

    The table can annotate regions (shapesor labels) and can be used to store additional information.
    Points are not regions but 0-dimensional locations. They can't be annotated by a table, but they can store
    annotation directly.

    The elements need to pass a validation step. To construct valid elements you can use the parsers that we
    provide:

        - :class:`~spatialdata.Image2DModel`,
        - :class:`~spatialdata.Image3DModel`,
        - :class:`~spatialdata.Labels2DModel`,
        - :class:`~spatialdata.Labels3DModel`,
        - :class:`~spatialdata.PointsModel`,
        - :class:`~spatialdata.ShapesModel`,
        - :class:`~spatialdata.TableModel`

    """

    _images: dict[str, Raster_T] = MappingProxyType({})  # type: ignore[assignment]
    _labels: dict[str, Raster_T] = MappingProxyType({})  # type: ignore[assignment]
    _points: dict[str, DaskDataFrame] = MappingProxyType({})  # type: ignore[assignment]
    _shapes: dict[str, GeoDataFrame] = MappingProxyType({})  # type: ignore[assignment]
    _tables: dict[str, AnnData] = MappingProxyType({})  # type: ignore[assignment]
    path: str | None = None

    @deprecation_alias(table="tables")
    def __init__(
        self,
        images: dict[str, Raster_T] = MappingProxyType({}),  # type: ignore[assignment]
        labels: dict[str, Raster_T] = MappingProxyType({}),  # type: ignore[assignment]
        points: dict[str, DaskDataFrame] = MappingProxyType({}),  # type: ignore[assignment]
        shapes: dict[str, GeoDataFrame] = MappingProxyType({}),  # type: ignore[assignment]
        tables: AnnData | dict[str, AnnData] = MappingProxyType({}),  # type: ignore[assignment]
    ) -> None:
        self.path = None

        # Work around to allow for backward compatibility
        if isinstance(tables, AnnData):
            tables = {"table": tables}

        self._validate_unique_element_names(
            list(images.keys())
            + list(labels.keys())
            + list(points.keys())
            + list(shapes.keys())
            + (list(tables.keys() if tables else []))
        )

        if images is not None:
            self._images: dict[str, SpatialImage | MultiscaleSpatialImage] = {}
            for k, v in images.items():
                self._add_image_in_memory(name=k, image=v)

        if labels is not None:
            self._labels: dict[str, SpatialImage | MultiscaleSpatialImage] = {}
            for k, v in labels.items():
                self._add_labels_in_memory(name=k, labels=v)

        if shapes is not None:
            self._shapes: dict[str, GeoDataFrame] = {}
            for k, v in shapes.items():
                self._add_shapes_in_memory(name=k, shapes=v)

        if points is not None:
            self._points: dict[str, DaskDataFrame] = {}
            for k, v in points.items():
                self._add_points_in_memory(name=k, points=v)

        if tables is not None:
            self._tables: dict[str, AnnData] = {}

            self._add_tables(table_mapping=tables)

        self._query = QueryManager(self)

    def validate_table_in_spatialdata(self, data: AnnData) -> None:
        TableModel().validate(data)
        element_names = [
            element_name for element_type, element_name, _ in self._gen_elements() if element_type != "tables"
        ]
        if TableModel.ATTRS_KEY in data.uns:
            attrs = data.uns[TableModel.ATTRS_KEY]
            regions = (
                attrs[TableModel.REGION_KEY]
                if isinstance(attrs[TableModel.REGION_KEY], list)
                else [attrs[TableModel.REGION_KEY]]
            )
            if not all(element_name in element_names for element_name in regions):
                warnings.warn(
                    "The table is annotating elements not present in the SpatialData object", UserWarning, stacklevel=2
                )

    @staticmethod
    def from_elements_dict(elements_dict: dict[str, SpatialElement | AnnData]) -> SpatialData:
        """
        Create a SpatialData object from a dict of elements.

        Parameters
        ----------
        elements_dict
            Dict of elements. The keys are the names of the elements and the values are the elements.
            A table can be present in the dict, but only at most one; its name is not used and can be anything.

        Returns
        -------
        The SpatialData object.
        """
        d: dict[str, dict[str, SpatialElement] | AnnData | None] = {
            "images": {},
            "labels": {},
            "points": {},
            "shapes": {},
            "table": {},
        }
        for k, e in elements_dict.items():
            schema = get_model(e)
            if schema in (Image2DModel, Image3DModel):
                assert isinstance(d["images"], dict)
                d["images"][k] = e
            elif schema in (Labels2DModel, Labels3DModel):
                assert isinstance(d["labels"], dict)
                d["labels"][k] = e
            elif schema == PointsModel:
                assert isinstance(d["points"], dict)
                d["points"][k] = e
            elif schema == ShapesModel:
                assert isinstance(d["shapes"], dict)
                d["shapes"][k] = e
            elif schema == TableModel:
                d["table"][k] = e
            else:
                raise ValueError(f"Unknown schema {schema}")
        return SpatialData(**d)  # type: ignore[arg-type]

    @staticmethod
    def get_annotated_regions(table: AnnData) -> str | list[str]:
        regions, _, _ = get_table_keys(table)
        return regions

    @staticmethod
    def get_region_key_column(table: AnnData) -> str:
        _, region_key, _ = get_table_keys(table)
        if table.obs.get(region_key):
            return table.obs[region_key]
        raise KeyError(f"{region_key} is set as region key column. However the column is not found in table.obs.")

    @staticmethod
    def get_instance_key_column(table: AnnData) -> str:
        _, _, instance_key = get_table_keys(table)
        if table.obs.get(instance_key):
            return table.obs[instance_key]
        raise KeyError(f"{instance_key} is set as instance key column. However the column is not found in table.obs.")

    @staticmethod
    def _set_table_annotation_target(
        table: AnnData,
        target_element_name: str | pd.Series,
        region_key: str,
        instance_key: str,
    ) -> None:
        if region_key not in table.obs:
            raise ValueError(f"Specified region_key, {region_key}, not in table.obs")
        if instance_key not in table.obs:
            raise ValueError(f"Specified instance_key, {instance_key}, not in table.obs")
        attrs = {
            TableModel.REGION_KEY: target_element_name,
            TableModel.REGION_KEY_KEY: region_key,
            TableModel.INSTANCE_KEY: instance_key,
        }
        check_target_region_column_symmetry(table, region_key, target_element_name)
        table.uns[TableModel.ATTRS_KEY] = attrs

    @staticmethod
    def _change_table_annotation_target(
        table: AnnData,
        target_element_name: str | pd.Series,
        region_key: None | str = None,
        instance_key: None | str = None,
    ) -> None:
        attrs = table.uns[TableModel.ATTRS_KEY]
        if not region_key:
            if attrs.get(TableModel.REGION_KEY_KEY) and table.obs.get(attrs[TableModel.REGION_KEY_KEY]) is not None:
                if not instance_key and (
                    not attrs.get(TableModel.INSTANCE_KEY) or attrs[TableModel.INSTANCE_KEY] not in table.obs
                ):
                    raise ValueError(
                        "Instance key and/or instance key column not found in table.obs. Please pass instance_key as "
                        "parameter."
                    )
                if instance_key:
                    if instance_key in table.obs:
                        attrs[TableModel.INSTANCE_KEY] = instance_key
                    else:
                        raise ValueError(f"Instance key {instance_key} not found in table.obs.")
                check_target_region_column_symmetry(table, attrs[TableModel.REGION_KEY_KEY], target_element_name)
                attrs[TableModel.REGION_KEY] = target_element_name
            else:
                raise ValueError(
                    "Region key and/or region key column not found in table. Please pass region_key as a parameter"
                )
        else:
            if region_key not in table.obs:
                raise ValueError(f"{region_key} not present in table.obs")

            if not instance_key and (
                not attrs.get(TableModel.INSTANCE_KEY) or attrs[TableModel.INSTANCE_KEY] not in table.obs
            ):
                raise ValueError(
                    "Instance key and/or instance key column not found in table.obs. Please pass instance_key as "
                    "parameter."
                )
            if instance_key:
                if instance_key in table.obs:
                    attrs[TableModel.INSTANCE_KEY] = instance_key
                else:
                    raise ValueError(f"Instance key {instance_key} not found in table.obs.")
            check_target_region_column_symmetry(table, attrs[TableModel.REGION_KEY_KEY], target_element_name)
            attrs[TableModel.REGION_KEY] = target_element_name

    def set_table_annotation_target(
        self,
        table_name: str,
        target_element_name: str | pd.Series,
        region_key: None | str = None,
        instance_key: None | str = None,
    ):
        table = self._tables[table_name]
        element_names = {element[1] for element in self._gen_elements()}
        if target_element_name not in element_names:
            raise ValueError(f"Annotation target {target_element_name} not present in SpatialData object.")

        if table.uns.get(TableModel.ATTRS_KEY):
            self._change_table_annotation_target(table, target_element_name, region_key, instance_key)
        else:
            self._set_table_annotation_target(table, target_element_name, region_key, instance_key)

    @property
    def query(self) -> QueryManager:
        return self._query

    def aggregate(
        self,
        values_sdata: SpatialData | None = None,
        values: DaskDataFrame | GeoDataFrame | SpatialImage | MultiscaleSpatialImage | str | None = None,
        by_sdata: SpatialData | None = None,
        by: GeoDataFrame | SpatialImage | MultiscaleSpatialImage | str | None = None,
        value_key: list[str] | str | None = None,
        agg_func: str | list[str] = "sum",
        target_coordinate_system: str = "global",
        fractions: bool = False,
        region_key: str = "region",
        instance_key: str = "instance_id",
        deepcopy: bool = True,
        **kwargs: Any,
    ) -> SpatialData:
        """
        Aggregate values by given region.

        Notes
        -----
        This function calls :func:`spatialdata.aggregate` with the convenience that values and by can be string
        without having to specify the values_sdata and by_sdata, which in that case will be replaced by `self`.

        Please see
        :func:`spatialdata.aggregate` for the complete docstring.
        """
        from spatialdata._core.operations.aggregate import aggregate

        if isinstance(values, str) and values_sdata is None:
            values_sdata = self
        if isinstance(by, str) and by_sdata is None:
            by_sdata = self

        return aggregate(
            values_sdata=values_sdata,
            values=values,
            by_sdata=by_sdata,
            by=by,
            value_key=value_key,
            agg_func=agg_func,
            target_coordinate_system=target_coordinate_system,
            fractions=fractions,
            region_key=region_key,
            instance_key=instance_key,
            deepcopy=deepcopy,
            **kwargs,
        )

    @staticmethod
    def _validate_unique_element_names(element_names: list[str]) -> None:
        if len(element_names) != len(set(element_names)):
            duplicates = {x for x in element_names if element_names.count(x) > 1}
            raise ValueError(
                f"Element names must be unique. The following element names are used multiple times: {duplicates}"
            )

    def _add_image_in_memory(
        self, name: str, image: SpatialImage | MultiscaleSpatialImage, overwrite: bool = False
    ) -> None:
        """Add an image element to the SpatialData object.

        Parameters
        ----------
        name
            name of the image
        image
            the image element to be added
        overwrite
            whether to overwrite the image if the name already exists.
        """
        self._validate_unique_element_names(
            list(self.labels.keys()) + list(self.points.keys()) + list(self.shapes.keys()) + [name]
        )
        if name in self._images and not overwrite:
            raise KeyError(f"Image {name} already exists in the dataset.")
        ndim = len(get_axes_names(image))
        if ndim == 3:
            Image2D_s.validate(image)
            self._images[name] = image
        elif ndim == 4:
            Image3D_s.validate(image)
            self._images[name] = image
        else:
            raise ValueError("Only czyx and cyx images supported")

    def _add_labels_in_memory(
        self, name: str, labels: SpatialImage | MultiscaleSpatialImage, overwrite: bool = False
    ) -> None:
        """
        Add a labels element to the SpatialData object.

        Parameters
        ----------
        name
            name of the labels
        labels
            the labels element to be added
        overwrite
            whether to overwrite the labels if the name already exists.
        """
        self._validate_unique_element_names(
            list(self.images.keys()) + list(self.points.keys()) + list(self.shapes.keys()) + [name]
        )
        if name in self._labels and not overwrite:
            raise KeyError(f"Labels {name} already exists in the dataset.")
        ndim = len(get_axes_names(labels))
        if ndim == 2:
            Label2D_s.validate(labels)
            self._labels[name] = labels
        elif ndim == 3:
            Label3D_s.validate(labels)
            self._labels[name] = labels
        else:
            raise ValueError(f"Only yx and zyx labels supported, got {ndim} dimensions")

    def _add_shapes_in_memory(self, name: str, shapes: GeoDataFrame, overwrite: bool = False) -> None:
        """
        Add a shapes element to the SpatialData object.

        Parameters
        ----------
        name
            name of the shapes
        shapes
            the shapes element to be added
        overwrite
            whether to overwrite the shapes if the name already exists.
        """
        self._validate_unique_element_names(
            list(self.images.keys()) + list(self.points.keys()) + list(self.labels.keys()) + [name]
        )
        if name in self._shapes and not overwrite:
            raise KeyError(f"Shapes {name} already exists in the dataset.")
        Shape_s.validate(shapes)
        self._shapes[name] = shapes

    def _add_points_in_memory(self, name: str, points: DaskDataFrame, overwrite: bool = False) -> None:
        """
        Add a points element to the SpatialData object.

        Parameters
        ----------
        name
            name of the points element
        points
            the points to be added
        overwrite
            whether to overwrite the points if the name already exists.
        """
        self._validate_unique_element_names(
            list(self.images.keys()) + list(self.labels.keys()) + list(self.shapes.keys()) + [name]
        )
        if name in self._points and not overwrite:
            raise KeyError(f"Points {name} already exists in the dataset.")
        Point_s.validate(points)
        self._points[name] = points

    def is_backed(self) -> bool:
        """Check if the data is backed by a Zarr storage or if it is in-memory."""
        return self.path is not None

    # TODO: from a commennt from Giovanni: consolite somewhere in
    #  a future PR (luca: also _init_add_element could be cleaned)
    def _get_group_for_element(self, name: str, element_type: str) -> zarr.Group:
        """
        Get the group for an element, creates a new one if the element doesn't exist.

        Parameters
        ----------
        name
            name of the element
        element_type
            type of the element. Should be in ["images", "labels", "points", "polygons", "shapes"].

        Returns
        -------
        either the existing Zarr sub-group or a new one
        """
        store = parse_url(self.path, mode="r+").store
        root = zarr.group(store=store)
        assert element_type in ["images", "labels", "points", "polygons", "shapes"]
        element_type_group = root.require_group(element_type)
        return element_type_group.require_group(name)

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
        assert element_type in ["images", "labels", "points", "shapes"]
        # not need to create the group for labels as it is already handled by ome-zarr-py
        if element_type != "labels":
            elem_group = root.create_group(name=element_type) if element_type not in root else root[element_type]
        if overwrite:
            if element_type == "labels" and element_type in root:
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
            if not bypass and name in elem_group:
                raise ValueError(f"Element {name} already exists, use overwrite=True to overwrite it")

        if element_type != "labels":
            return elem_group
        return root

    def _locate_spatial_element(self, element: SpatialElement) -> tuple[str, str]:
        """
        Find the SpatialElement within the SpatialData object.

        Parameters
        ----------
        element
            The queried SpatialElement


        Returns
        -------
        name and type of the element

        Raises
        ------
        ValueError
            the element is not found or found multiple times in the SpatialData object
        """
        found: list[SpatialElement] = []
        found_element_type: list[str] = []
        found_element_name: list[str] = []
        for element_type in ["images", "labels", "points", "shapes"]:
            for element_name, element_value in getattr(self, element_type).items():
                if id(element_value) == id(element):
                    found.append(element_value)
                    found_element_type.append(element_type)
                    found_element_name.append(element_name)
        if len(found) == 0:
            raise ValueError("Element not found in the SpatialData object.")
        if len(found) > 1:
            raise ValueError(
                f"Element found multiple times in the SpatialData object."
                f"Found {len(found)} elements with names: {found_element_name},"
                f" and types: {found_element_type}"
            )
        assert len(found_element_name) == 1
        assert len(found_element_type) == 1
        return found_element_name[0], found_element_type[0]

    def contains_element(self, element: SpatialElement, raise_exception: bool = False) -> bool:
        """
        Check if the SpatialElement is contained in the SpatialData object.

        Parameters
        ----------
        element
            The SpatialElement to check
        raise_exception
            If True, raise an exception if the element is not found. If False, return False if the element is not found.

        Returns
        -------
        True if the element is found; False otherwise (if raise_exception is False).
        """
        try:
            self._locate_spatial_element(element)
            return True
        except ValueError as e:
            if raise_exception:
                raise e
            return False

    def _write_transformations_to_disk(self, element: SpatialElement) -> None:
        """
        Write transformations to disk for an element.

        Parameters
        ----------
        element
            The SpatialElement object for which the transformations to be written
        """
        from spatialdata.transformations.operations import get_transformation

        transformations = get_transformation(element, get_all=True)
        assert isinstance(transformations, dict)
        found_element_name, found_element_type = self._locate_spatial_element(element)

        if self.path is not None:
            group = self._get_group_for_element(name=found_element_name, element_type=found_element_type)
            axes = get_axes_names(element)
            if isinstance(element, (SpatialImage, MultiscaleSpatialImage)):
                from spatialdata._io._utils import (
                    overwrite_coordinate_transformations_raster,
                )

                overwrite_coordinate_transformations_raster(group=group, axes=axes, transformations=transformations)
            elif isinstance(element, (DaskDataFrame, GeoDataFrame, AnnData)):
                from spatialdata._io._utils import (
                    overwrite_coordinate_transformations_non_raster,
                )

                overwrite_coordinate_transformations_non_raster(group=group, axes=axes, transformations=transformations)
            else:
                raise ValueError("Unknown element type")

    @deprecation_alias(filter_table="filter_tables")
    def filter_by_coordinate_system(
        self, coordinate_system: str | list[str], filter_tables: bool = True
    ) -> SpatialData:
        """
        Filter the SpatialData by one (or a list of) coordinate system.

        This returns a SpatialData object with the elements containing a transformation mapping to the specified
        coordinate system(s).

        Parameters
        ----------
        coordinate_system
            The coordinate system(s) to filter by.
        filter_table
            If True (default), the table will be filtered to only contain regions
            of an element belonging to the specified coordinate system(s).

        Returns
        -------
        The filtered SpatialData.
        """
        # TODO: decide whether to add parameter to filter only specific table.
        from spatialdata._core.query.relational_query import _filter_table_by_coordinate_system
        from spatialdata.transformations.operations import get_transformation

        elements: dict[str, dict[str, SpatialElement]] = {}
        element_paths_in_coordinate_system = []
        if isinstance(coordinate_system, str):
            coordinate_system = [coordinate_system]
        for element_type, element_name, element in self._gen_elements():
            if element_type != "tables":
                transformations = get_transformation(element, get_all=True)
                assert isinstance(transformations, dict)
                for cs in coordinate_system:
                    if cs in transformations:
                        if element_type not in elements:
                            elements[element_type] = {}
                        elements[element_type][element_name] = element
                        element_paths_in_coordinate_system.append(element_name)

        # TODO: check whether full table dict should be returned or only those which annotate elements. Also check
        # filtering with tables having potentially different keys.
        if filter_tables:
            tables = {}
            for table_name, table in self._tables.items():
                tables[table_name] = _filter_table_by_coordinate_system(table, element_paths_in_coordinate_system)
        else:
            tables = self._tables

        return SpatialData(**elements, tables=tables)

    def rename_coordinate_systems(self, rename_dict: dict[str, str]) -> None:
        """
        Rename coordinate systems.

        Parameters
        ----------
        rename_dict
            A dictionary mapping old coordinate system names to new coordinate system names.

        Notes
        -----
        The method does not allow to rename a coordinate system into an existing one, unless the existing one is also
        renamed in the same call.
        """
        from spatialdata.transformations.operations import get_transformation, set_transformation

        # check that the rename_dict is valid
        old_names = self.coordinate_systems
        new_names = list(set(old_names).difference(set(rename_dict.keys())))
        for old_cs, new_cs in rename_dict.items():
            if old_cs not in old_names:
                raise ValueError(f"Coordinate system {old_cs} does not exist.")
            if new_cs in new_names:
                raise ValueError(
                    "It is not allowed to rename a coordinate system if the new name already exists and "
                    "if it is not renamed in the same call."
                )
            new_names.append(new_cs)

        # rename the coordinate systems
        for element in self._gen_elements_values():
            # get the transformations
            transformations = get_transformation(element, get_all=True)
            assert isinstance(transformations, dict)

            # appends a random suffix to the coordinate system name to avoid collisions
            suffixes_to_replace = set()
            for old_cs, new_cs in rename_dict.items():
                if old_cs in transformations:
                    random_suffix = hashlib.sha1(os.urandom(128)).hexdigest()[:8]
                    transformations[new_cs + random_suffix] = transformations.pop(old_cs)
                    suffixes_to_replace.add(new_cs + random_suffix)

            # remove the random suffixes
            new_transformations = {}
            for cs_with_suffix in transformations:
                if cs_with_suffix in suffixes_to_replace:
                    cs = cs_with_suffix[:-8]
                    new_transformations[cs] = transformations[cs_with_suffix]
                    suffixes_to_replace.remove(cs_with_suffix)
                else:
                    new_transformations[cs_with_suffix] = transformations[cs_with_suffix]

            # set the new transformations
            set_transformation(element=element, transformation=new_transformations, set_all=True)

    def transform_element_to_coordinate_system(
        self, element: SpatialElement, target_coordinate_system: str
    ) -> SpatialElement:
        """
        Transform an element to a given coordinate system.

        Parameters
        ----------
        element
            The element to transform.
        target_coordinate_system
            The target coordinate system.

        Returns
        -------
        The transformed element.
        """
        from spatialdata import transform
        from spatialdata.transformations import Identity
        from spatialdata.transformations.operations import (
            get_transformation_between_coordinate_systems,
            remove_transformation,
            set_transformation,
        )

        t = get_transformation_between_coordinate_systems(self, element, target_coordinate_system)
        transformed = transform(element, t, maintain_positioning=False)
        remove_transformation(transformed, remove_all=True)
        set_transformation(transformed, Identity(), target_coordinate_system)

        return transformed

    def transform_to_coordinate_system(
        self,
        target_coordinate_system: str,
    ) -> SpatialData:
        """
        Transform the SpatialData to a given coordinate system.

        Parameters
        ----------
        target_coordinate_system
            The target coordinate system.

        Returns
        -------
        The transformed SpatialData.
        """
        sdata = self.filter_by_coordinate_system(target_coordinate_system, filter_table=False)
        elements: dict[str, dict[str, SpatialElement]] = {}
        for element_type, element_name, element in sdata._gen_elements():
            if element_type != "tables":
                transformed = sdata.transform_element_to_coordinate_system(element, target_coordinate_system)
                if element_type not in elements:
                    elements[element_type] = {}
                elements[element_type][element_name] = transformed
        return SpatialData(**elements, table=sdata.tables)

    def add_image(
        self,
        name: str,
        image: SpatialImage | MultiscaleSpatialImage,
        storage_options: JSONDict | list[JSONDict] | None = None,
        overwrite: bool = False,
    ) -> None:
        """
        Add an image to the SpatialData object.

        Parameters
        ----------
        name
            Key to the element inside the SpatialData object.
        image
            The image to add, the object needs to pass validation
            (see :class:`~spatialdata.Image2DModel` and :class:`~spatialdata.Image3DModel`).
        storage_options
            Storage options for the Zarr storage.
            See https://zarr.readthedocs.io/en/stable/api/storage.html for more details.
        overwrite
            If True, overwrite the element if it already exists.

        Notes
        -----
        If the SpatialData object is backed by a Zarr storage, the image will be written to the Zarr storage.
        """
        if self.is_backed():
            files = get_backing_files(image)
            assert self.path is not None
            target_path = os.path.realpath(os.path.join(self.path, "images", name))
            if target_path in files:
                raise ValueError(
                    "Cannot add the image to the SpatialData object because it would overwrite an element that it is"
                    "using for backing. See more here: https://github.com/scverse/spatialdata/pull/138"
                )
            self._add_image_in_memory(name=name, image=image, overwrite=overwrite)
            # old code to support overwriting the backing file
            # with tempfile.TemporaryDirectory() as tmpdir:
            #     store = parse_url(Path(tmpdir) / "data.zarr", mode="w").store
            #     root = zarr.group(store=store)
            #     write_image(
            #         image=self.images[name],
            #         group=root,
            #         name=name,
            #         storage_options=storage_options,
            #     )
            #     src_element_path = Path(store.path) / name
            #     assert isinstance(self.path, str)
            #     tgt_element_path = Path(self.path) / "images" / name
            #     if os.path.isdir(tgt_element_path) and overwrite:
            #         element_store = parse_url(tgt_element_path, mode="w").store
            #         _ = zarr.group(store=element_store, overwrite=True)
            #         element_store.close()
            #     pathlib.Path(tgt_element_path).mkdir(parents=True, exist_ok=True)
            #     for file in os.listdir(str(src_element_path)):
            #         src_file = src_element_path / file
            #         tgt_file = tgt_element_path / file
            #         os.rename(src_file, tgt_file)
            # from spatialdata._io.read import _read_multiscale
            #
            # # reload the image from the Zarr storage so that now the element is lazy loaded, and most importantly,
            # # from the correct storage
            # image = _read_multiscale(str(tgt_element_path), raster_type="image")
            # self._add_image_in_memory(name=name, image=image, overwrite=True)
            elem_group = self._init_add_element(name=name, element_type="images", overwrite=overwrite)
            write_image(
                image=self.images[name],
                group=elem_group,
                name=name,
                storage_options=storage_options,
            )
            from spatialdata._io.io_raster import _read_multiscale

            # reload the image from the Zarr storage so that now the element is lazy loaded, and most importantly,
            # from the correct storage
            assert elem_group.path == "images"
            path = Path(elem_group.store.path) / "images" / name
            image = _read_multiscale(path, raster_type="image")
            self._add_image_in_memory(name=name, image=image, overwrite=True)
        else:
            self._add_image_in_memory(name=name, image=image, overwrite=overwrite)

    def add_labels(
        self,
        name: str,
        labels: SpatialImage | MultiscaleSpatialImage,
        storage_options: JSONDict | list[JSONDict] | None = None,
        overwrite: bool = False,
    ) -> None:
        """
        Add labels to the SpatialData object.

        Parameters
        ----------
        name
            Key to the element inside the SpatialData object.
        labels
            The labels (masks) to add, the object needs to pass validation
            (see :class:`~spatialdata.Labels2DModel` and :class:`~spatialdata.Labels3DModel`).
        storage_options
            Storage options for the Zarr storage.
            See https://zarr.readthedocs.io/en/stable/api/storage.html for more details.
        overwrite
            If True, overwrite the element if it already exists.

        Notes
        -----
        If the SpatialData object is backed by a Zarr storage, the image will be written to the Zarr storage.
        """
        if self.is_backed():
            files = get_backing_files(labels)
            assert self.path is not None
            target_path = os.path.realpath(os.path.join(self.path, "labels", name))
            if target_path in files:
                raise ValueError(
                    "Cannot add the image to the SpatialData object because it would overwrite an element that it is"
                    "using for backing. We are considering changing this behavior to allow the overwriting of "
                    "elements used for backing. If you would like to support this use case please leave a comment on "
                    "https://github.com/scverse/spatialdata/pull/138"
                )
            self._add_labels_in_memory(name=name, labels=labels, overwrite=overwrite)
            # old code to support overwriting the backing file
            # with tempfile.TemporaryDirectory() as tmpdir:
            #     store = parse_url(Path(tmpdir) / "data.zarr", mode="w").store
            #     root = zarr.group(store=store)
            #     write_labels(
            #         labels=self.labels[name],
            #         group=root,
            #         name=name,
            #         storage_options=storage_options,
            #     )
            #     src_element_path = Path(store.path) / "labels" / name
            #     assert isinstance(self.path, str)
            #     tgt_element_path = Path(self.path) / "labels" / name
            #     if os.path.isdir(tgt_element_path) and overwrite:
            #         element_store = parse_url(tgt_element_path, mode="w").store
            #         _ = zarr.group(store=element_store, overwrite=True)
            #         element_store.close()
            #     pathlib.Path(tgt_element_path).mkdir(parents=True, exist_ok=True)
            #     for file in os.listdir(str(src_element_path)):
            #         src_file = src_element_path / file
            #         tgt_file = tgt_element_path / file
            #         os.rename(src_file, tgt_file)
            # from spatialdata._io.read import _read_multiscale
            #
            # # reload the labels from the Zarr storage so that now the element is lazy loaded, and most importantly,
            # # from the correct storage
            # labels = _read_multiscale(str(tgt_element_path), raster_type="labels")
            # self._add_labels_in_memory(name=name, labels=labels, overwrite=True)
            elem_group = self._init_add_element(name=name, element_type="labels", overwrite=overwrite)
            write_labels(
                labels=self.labels[name],
                group=elem_group,
                name=name,
                storage_options=storage_options,
            )
            # reload the labels from the Zarr storage so that now the element is lazy loaded, and most importantly,
            # from the correct storage
            from spatialdata._io.io_raster import _read_multiscale

            # just a check to make sure that things go as expected
            assert elem_group.path == ""
            path = Path(elem_group.store.path) / "labels" / name
            labels = _read_multiscale(path, raster_type="labels")
            self._add_labels_in_memory(name=name, labels=labels, overwrite=True)
        else:
            self._add_labels_in_memory(name=name, labels=labels, overwrite=overwrite)

    def add_points(
        self,
        name: str,
        points: DaskDataFrame,
        overwrite: bool = False,
    ) -> None:
        """
        Add points to the SpatialData object.

        Parameters
        ----------
        name
            Key to the element inside the SpatialData object.
        points
            The points to add, the object needs to pass validation (see :class:`spatialdata.PointsModel`).
        storage_options
            Storage options for the Zarr storage.
            See https://zarr.readthedocs.io/en/stable/api/storage.html for more details.
        overwrite
            If True, overwrite the element if it already exists.

        Notes
        -----
        If the SpatialData object is backed by a Zarr storage, the image will be written to the Zarr storage.
        """
        if self.is_backed():
            files = get_backing_files(points)
            assert self.path is not None
            target_path = os.path.realpath(os.path.join(self.path, "points", name, "points.parquet"))
            if target_path in files:
                raise ValueError(
                    "Cannot add the image to the SpatialData object because it would overwrite an element that it is"
                    "using for backing. We are considering changing this behavior to allow the overwriting of "
                    "elements used for backing. If you would like to support this use case please leave a comment on "
                    "https://github.com/scverse/spatialdata/pull/138"
                )
            self._add_points_in_memory(name=name, points=points, overwrite=overwrite)
            # old code to support overwriting the backing file
            # with tempfile.TemporaryDirectory() as tmpdir:
            #     store = parse_url(Path(tmpdir) / "data.zarr", mode="w").store
            #     root = zarr.group(store=store)
            #     write_points(
            #         points=self.points[name],
            #         group=root,
            #         name=name,
            #     )
            #     src_element_path = Path(store.path) / name
            #     assert isinstance(self.path, str)
            #     tgt_element_path = Path(self.path) / "points" / name
            #     if os.path.isdir(tgt_element_path) and overwrite:
            #         element_store = parse_url(tgt_element_path, mode="w").store
            #         _ = zarr.group(store=element_store, overwrite=True)
            #         element_store.close()
            #     pathlib.Path(tgt_element_path).mkdir(parents=True, exist_ok=True)
            #     for file in os.listdir(str(src_element_path)):
            #         src_file = src_element_path / file
            #         tgt_file = tgt_element_path / file
            #         os.rename(src_file, tgt_file)
            # from spatialdata._io.read import _read_points
            #
            # # reload the points from the Zarr storage so that now the element is lazy loaded, and most importantly,
            # # from the correct storage
            # points = _read_points(str(tgt_element_path))
            # self._add_points_in_memory(name=name, points=points, overwrite=True)
            elem_group = self._init_add_element(name=name, element_type="points", overwrite=overwrite)
            write_points(
                points=self.points[name],
                group=elem_group,
                name=name,
            )
            # reload the points from the Zarr storage so that now the element is lazy loaded, and most importantly,
            # from the correct storage
            from spatialdata._io.io_points import _read_points

            assert elem_group.path == "points"

            path = Path(elem_group.store.path) / "points" / name
            points = _read_points(path)
            self._add_points_in_memory(name=name, points=points, overwrite=True)
        else:
            self._add_points_in_memory(name=name, points=points, overwrite=overwrite)

    def add_shapes(
        self,
        name: str,
        shapes: GeoDataFrame,
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
        self._add_shapes_in_memory(name=name, shapes=shapes, overwrite=overwrite)
        if self.is_backed():
            elem_group = self._init_add_element(name=name, element_type="shapes", overwrite=overwrite)
            write_shapes(
                shapes=self.shapes[name],
                group=elem_group,
                name=name,
            )
            # no reloading of the file storage since the AnnData is not lazy loaded

    def write(
        self,
        file_path: str | Path,
        storage_options: JSONDict | list[JSONDict] | None = None,
        overwrite: bool = False,
        consolidate_metadata: bool = True,
    ) -> None:
        """Write the SpatialData object to Zarr."""
        if isinstance(file_path, str):
            file_path = Path(file_path)
        assert isinstance(file_path, Path)

        if self.is_backed() and self.path != file_path:
            logger.info(f"The Zarr file used for backing will now change from {self.path} to {file_path}")

        # old code to support overwriting the backing file
        # target_path = None
        # tmp_zarr_file = None
        if os.path.exists(file_path):
            if parse_url(file_path, mode="r") is None:
                raise ValueError(
                    "The target file path specified already exists, and it has been detected to not be "
                    "a Zarr store. Overwriting non-Zarr stores is not supported to prevent accidental "
                    "data loss."
                )
            if not overwrite and self.path != str(file_path):
                raise ValueError("The Zarr store already exists. Use `overwrite=True` to overwrite the store.")
            raise ValueError(
                "The file path specified is the same as the one used for backing. "
                "Overwriting the backing file is not supported to prevent accidental data loss."
                "We are discussing how to support this use case in the future, if you would like us to "
                "support it please leave a comment on https://github.com/scverse/spatialdata/pull/138"
            )
            # old code to support overwriting the backing file
            # else:
            #     target_path = tempfile.TemporaryDirectory()
            #     tmp_zarr_file = Path(target_path.name) / "data.zarr"

        # old code to support overwriting the backing file
        # if target_path is None:
        #     store = parse_url(file_path, mode="w").store
        # else:
        #     store = parse_url(tmp_zarr_file, mode="w").store
        # store = parse_url(file_path, mode="w").store
        # root = zarr.group(store=store)
        store = parse_url(file_path, mode="w").store

        root = zarr.group(store=store, overwrite=overwrite)
        store.close()

        # old code to support overwriting the backing file
        # if target_path is None:
        #     self.path = str(file_path)
        # else:
        #     self.path = str(tmp_zarr_file)
        self.path = str(file_path)
        try:
            if len(self.images):
                root.create_group(name="images")
                # add_image_in_memory will delete and replace the same key in self.images,
                # so we need to make a copy of the keys. Same for the other elements
                keys = self.images.keys()
                from spatialdata._io.io_raster import _read_multiscale

                for name in keys:
                    elem_group = self._init_add_element(name=name, element_type="images", overwrite=overwrite)
                    write_image(
                        image=self.images[name],
                        group=elem_group,
                        name=name,
                        storage_options=storage_options,
                    )

                    # reload the image from the Zarr storage so that now the element is lazy loaded,
                    # and most importantly, from the correct storage
                    element_path = Path(self.path) / "images" / name
                    image = _read_multiscale(element_path, raster_type="image")
                    self._add_image_in_memory(name=name, image=image, overwrite=True)

            if len(self.labels):
                root.create_group(name="labels")
                # keys = list(self.labels.keys())
                keys = self.labels.keys()
                from spatialdata._io.io_raster import _read_multiscale

                for name in keys:
                    elem_group = self._init_add_element(name=name, element_type="labels", overwrite=overwrite)
                    write_labels(
                        labels=self.labels[name],
                        group=elem_group,
                        name=name,
                        storage_options=storage_options,
                    )

                    # reload the labels from the Zarr storage so that now the element is lazy loaded,
                    #  and most importantly, from the correct storage
                    element_path = Path(self.path) / "labels" / name
                    labels = _read_multiscale(element_path, raster_type="labels")
                    self._add_labels_in_memory(name=name, labels=labels, overwrite=True)

            if len(self.points):
                root.create_group(name="points")
                # keys = list(self.points.keys())
                keys = self.points.keys()
                from spatialdata._io.io_points import _read_points

                for name in keys:
                    elem_group = self._init_add_element(name=name, element_type="points", overwrite=overwrite)
                    write_points(
                        points=self.points[name],
                        group=elem_group,
                        name=name,
                    )
                    element_path = Path(self.path) / "points" / name

                    # reload the points from the Zarr storage so that the element is lazy loaded,
                    # and most importantly, from the correct storage
                    points = _read_points(element_path)
                    self._add_points_in_memory(name=name, points=points, overwrite=True)

            if len(self.shapes):
                root.create_group(name="shapes")
                # keys = list(self.shapes.keys())
                keys = self.shapes.keys()
                for name in keys:
                    elem_group = self._init_add_element(name=name, element_type="shapes", overwrite=overwrite)
                    write_shapes(
                        shapes=self.shapes[name],
                        group=elem_group,
                        name=name,
                    )
                    # no reloading of the file storage since the AnnData is not lazy loaded

            if len(self._tables):
                elem_group = root.create_group(name="tables")
                for key in self._tables:
                    write_table(table=self._tables[key], group=elem_group, name=key)

        except Exception as e:  # noqa: B902
            self.path = None
            raise e

        if consolidate_metadata:
            # consolidate metadata to more easily support remote reading
            # bug in zarr, 'zmetadata' is written instead of '.zmetadata'
            # see discussion https://github.com/zarr-developers/zarr-python/issues/1121
            zarr.consolidate_metadata(store, metadata_key=".zmetadata")

        # old code to support overwriting the backing file
        # if target_path is not None:
        #     if os.path.isdir(file_path):
        #         assert overwrite is True
        #         store = parse_url(file_path, mode="w").store
        #         _ = zarr.group(store=store, overwrite=overwrite)
        #         store.close()
        #     for file in os.listdir(str(tmp_zarr_file)):
        #         assert isinstance(tmp_zarr_file, Path)
        #         src_file = tmp_zarr_file / file
        #         tgt_file = file_path / file
        #         os.rename(src_file, tgt_file)
        #     target_path.cleanup()
        #
        #     self.path = str(file_path)
        #     # elements that need to be reloaded are: images, labels, points
        #     # non-backed elements don't need to be reloaded: table, shapes, polygons
        #
        #     from spatialdata._io.read import _read_multiscale, _read_points
        #
        #     for element_type in ["images", "labels", "points"]:
        #         names = list(self.__getattribute__(element_type).keys())
        #         for name in names:
        #             path = file_path / element_type / name
        #             if element_type in ["images", "labels"]:
        #                 raster_type = element_type if element_type == "labels" else "image"
        #                 element = _read_multiscale(str(path), raster_type=raster_type)  # type: ignore[arg-type]
        #             elif element_type == "points":
        #                 element = _read_points(str(path))
        #             else:
        #                 raise ValueError(f"Unknown element type {element_type}")
        #             self.__getattribute__(element_type)[name] = element
        assert isinstance(self.path, str)

    @property
    def table(self) -> None | AnnData:
        """
        Return the table.

        Returns
        -------
        The table.
        """
        # TODO: decide version for deprecation
        warnings.warn(
            "Table accessor will be deprecated with SpatialData version X.X, use sdata.tables instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Isinstance will still return table if anndata has 0 rows.
        if isinstance(self._tables.get("table"), AnnData):
            return self._tables["table"]
        return None

    @property
    def tables(self) -> dict[str, AnnData]:
        """
        Return tables dictionary.

        Returns
        -------
        dict[str, AnnData]
            Either the empty dictionary or a dictionary with as values the strings representing the table names and
            as values the AnnData tables themselves.
        """
        return self._tables

    @table.setter
    def table(self, table: AnnData) -> None:
        warnings.warn(
            "Table setter will be deprecated with SpatialData version X.X, use tables instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        TableModel().validate(table)
        if self._tables.get("table") is not None:
            raise ValueError("The table already exists. Use del sdata.tables['table'] to remove it first.")
        self._tables["table"] = table
        self._store_tables(table_name="table")

    @table.deleter
    def table(self) -> None:
        """Delete the table."""
        warnings.warn(
            "del sdata.table will be deprecated with SpatialData version X.X, use del sdata.tables['table'] instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if self._tables.get("table"):
            self._tables["table"] = None
            if self.is_backed():
                store = parse_url(self.path, mode="r+").store
                root = zarr.group(store=store)
                del root["tables/table"]
        else:
            # More informative than the error in the zarr library.
            raise KeyError("table with name 'table' not present in the SpatialData object.")

    def add_tables(
        self, table: None | AnnData = None, table_name: str = None, table_mapping: None | dict[str, AnnData] = None
    ):
        self._add_tables(table=table, table_name=table_name, table_mapping=table_mapping)

    def _store_tables(self, table_name: None | str = None, table_mapping: None | dict[str, AnnData] = None) -> None:
        if self.is_backed():
            store = parse_url(self.path, mode="r+").store
            root = zarr.group(store=store)
            elem_group = root.require_group(name="tables")
            if table_name:
                write_table(table=self._tables[table_name], group=elem_group, name=table_name)
            else:
                for table_name in table_mapping:
                    write_table(table=self._tables[table_name], group=elem_group, name=table_name)

    def _add_tables(
        self, table: None | AnnData = None, table_name: str = None, table_mapping: None | dict[str, AnnData] = None
    ) -> None:
        if table:
            if table_name:
                TableModel().validate(table)
                if self._tables.get(table_name) is not None:
                    raise ValueError("The table already exists. Use del sdata.tables[<table_name>] to remove it first.")
                self._tables[table_name] = table
                self._store_tables(table_name=table_name)
            else:
                raise ValueError("Please provide a string value for the parameter table_name.")
        elif table_mapping:
            for table in table_mapping.values():
                self.validate_table_in_spatialdata(table)
            self._tables.update(table_mapping)
            self._store_tables(table_mapping=table_mapping)

    @staticmethod
    def read(file_path: str, selection: tuple[str] | None = None) -> SpatialData:
        """
        Read a SpatialData object from a Zarr storage (on-disk or remote).

        Parameters
        ----------
        file_path
            The path or URL to the Zarr storage.
        selection
            The elements to read (images, labels, points, shapes, table). If None, all elements are read.

        Returns
        -------
        The SpatialData object.
        """
        from spatialdata import read_zarr

        return read_zarr(file_path, selection=selection)

    @property
    def images(self) -> dict[str, SpatialImage | MultiscaleSpatialImage]:
        """Return images as a Dict of name to image data."""
        return self._images

    @property
    def labels(self) -> dict[str, SpatialImage | MultiscaleSpatialImage]:
        """Return labels as a Dict of name to label data."""
        return self._labels

    @property
    def points(self) -> dict[str, DaskDataFrame]:
        """Return points as a Dict of name to point data."""
        return self._points

    @property
    def shapes(self) -> dict[str, GeoDataFrame]:
        """Return shapes as a Dict of name to shape data."""
        return self._shapes

    @property
    def coordinate_systems(self) -> list[str]:
        from spatialdata.transformations.operations import get_transformation

        all_cs = set()
        gen = self._gen_elements_values()
        for obj in gen:
            transformations = get_transformation(obj, get_all=True)
            assert isinstance(transformations, dict)
            for cs in transformations:
                all_cs.add(cs)
        return list(all_cs)

    def _non_empty_elements(self) -> list[str]:
        """Get the names of the elements that are not empty.

        Returns
        -------
        non_empty_elements
            The names of the elements that are not empty.
        """
        all_elements = ["images", "labels", "points", "shapes", "tables"]
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
        """
        Generate a string representation of the SpatialData object.

        Returns
        -------
            The string representation of the SpatialData object.
        """

        def rreplace(s: str, old: str, new: str, occurrence: int) -> str:
            li = s.rsplit(old, occurrence)
            return new.join(li)

        def h(s: str) -> str:
            return hashlib.md5(repr(s).encode()).hexdigest()

        descr = "SpatialData object with:"

        non_empty_elements = self._non_empty_elements()
        last_element_index = len(non_empty_elements) - 1
        for attr_index, attr in enumerate(non_empty_elements):
            last_attr = attr_index == last_element_index
            attribute = getattr(self, attr)

            descr += f"\n{h('level0')}{attr.capitalize()}"

            unsorted_elements = attribute.items()
            sorted_elements = sorted(unsorted_elements, key=lambda x: _natural_keys(x[0]))
            for k, v in sorted_elements:
                descr += f"{h('empty_line')}"
                descr_class = v.__class__.__name__
                if attr == "shapes":
                    descr += f"{h(attr + 'level1.1')}{k!r}: {descr_class} " f"shape: {v.shape} (2D shapes)"
                elif attr == "points":
                    length: int | None = None
                    if len(v.dask.layers) == 1:
                        name, layer = v.dask.layers.items().__iter__().__next__()
                        if "read-parquet" in name:
                            t = layer.creation_info["args"]
                            assert isinstance(t, tuple)
                            assert len(t) == 1
                            parquet_file = t[0]
                            table = read_parquet(parquet_file)
                            length = len(table)
                        else:
                            # length = len(v)
                            length = None
                    else:
                        length = None

                    n = len(get_axes_names(v))
                    dim_string = f"({n}D points)"

                    assert len(v.shape) == 2
                    if length is not None:
                        shape_str = f"({length}, {v.shape[1]})"
                    else:
                        shape_str = (
                            "("
                            + ", ".join([str(dim) if not isinstance(dim, Delayed) else "<Delayed>" for dim in v.shape])
                            + ")"
                        )
                    descr += f"{h(attr + 'level1.1')}{k!r}: {descr_class} " f"with shape: {shape_str} {dim_string}"
                elif attr == "tables":
                    descr += f"{h(attr + 'level1.1')}{k!r}: {descr_class} {v.shape}"
                else:
                    if isinstance(v, SpatialImage):
                        descr += f"{h(attr + 'level1.1')}{k!r}: {descr_class}[{''.join(v.dims)}] {v.shape}"
                    elif isinstance(v, MultiscaleSpatialImage):
                        shapes = []
                        dims: str | None = None
                        for pyramid_level in v:
                            dataset_names = list(v[pyramid_level].keys())
                            assert len(dataset_names) == 1
                            dataset_name = dataset_names[0]
                            vv = v[pyramid_level][dataset_name]
                            shape = vv.shape
                            if dims is None:
                                dims = "".join(vv.dims)
                            shapes.append(shape)
                        descr += f"{h(attr + 'level1.1')}{k!r}: {descr_class}[{dims}] " f"{', '.join(map(str, shapes))}"
                    else:
                        raise TypeError(f"Unknown type {type(v)}")
            if last_attr is True:
                descr = descr.replace(h("empty_line"), "\n  ")
            else:
                descr = descr.replace(h("empty_line"), "\n│ ")

        descr = rreplace(descr, h("level0"), "└── ", 1)
        descr = descr.replace(h("level0"), "├── ")

        for attr in ["images", "labels", "points", "tables", "shapes"]:
            descr = rreplace(descr, h(attr + "level1.1"), "    └── ", 1)
            descr = descr.replace(h(attr + "level1.1"), "    ├── ")

        from spatialdata.transformations.operations import get_transformation

        descr += "\nwith coordinate systems:\n"
        coordinate_systems = self.coordinate_systems.copy()
        coordinate_systems.sort(key=_natural_keys)
        for i, cs in enumerate(coordinate_systems):
            descr += f"▸ {cs!r}"
            gen = self._gen_elements()
            elements_in_cs: dict[str, list[str]] = {}
            for k, name, obj in gen:
                if not isinstance(obj, AnnData):
                    transformations = get_transformation(obj, get_all=True)
                    assert isinstance(transformations, dict)
                    target_css = transformations.keys()
                    if cs in target_css:
                        if k not in elements_in_cs:
                            elements_in_cs[k] = []
                        elements_in_cs[k].append(name)
            for element_names in elements_in_cs.values():
                element_names.sort(key=_natural_keys)
            if len(elements_in_cs) > 0:
                elements = ", ".join(
                    [
                        f"{element_name} ({element_type.capitalize()})"
                        for element_type, element_names in elements_in_cs.items()
                        for element_name in element_names
                    ]
                )
                descr += f", with elements:\n        {elements}"
            if i < len(coordinate_systems) - 1:
                descr += "\n"
        return descr

    def _gen_elements_values(self) -> Generator[SpatialElement, None, None]:
        for element_type in ["images", "labels", "points", "shapes"]:
            d = getattr(SpatialData, element_type).fget(self)
            yield from d.values()

    def _gen_elements(self) -> Generator[tuple[str, str, SpatialElement], None, None]:
        for element_type in ["images", "labels", "points", "shapes", "tables"]:
            d = getattr(SpatialData, element_type).fget(self)
            for k, v in d.items():
                yield element_type, k, v

    def _find_element(self, element_name: str) -> tuple[str, str, SpatialElement]:
        for element_type, element_name_, element in self._gen_elements():
            if element_name_ == element_name:
                return element_type, element_name_, element
        else:
            raise KeyError(f"Could not find element with name {element_name!r}")

    @classmethod
    @deprecation_alias(table="tables")
    def init_from_elements(
        cls, elements: dict[str, SpatialElement], tables: AnnData | dict[str, AnnData] | None = None
    ) -> SpatialData:
        """
        Create a SpatialData object from a dict of named elements and an optional table.

        Parameters
        ----------
        elements
            A dict of named elements.
        tables
            An optional table or dictionary of tables

        Returns
        -------
        The SpatialData object.
        """
        elements_dict: dict[str, SpatialElement] = {}
        for name, element in elements.items():
            model = get_model(element)
            if model in [Image2DModel, Image3DModel]:
                element_type = "images"
            elif model in [Labels2DModel, Labels3DModel]:
                element_type = "labels"
            elif model == PointsModel:
                element_type = "points"
            else:
                assert model == ShapesModel
                element_type = "shapes"
            elements_dict.setdefault(element_type, {})[name] = element
        return cls(**elements_dict, tables=tables)

    def __getitem__(self, item: str) -> SpatialElement:
        """
        Return the element with the given name.

        Parameters
        ----------
        item
            The name of the element to return.

        Returns
        -------
        The element.
        """
        _, _, element = self._find_element(item)
        return element

    def __setitem__(self, key: str, value: SpatialElement | AnnData) -> None:
        """
        Add the element to the SpatialData object.

        Parameters
        ----------
        key
            The name of the element.
        value
            The element.
        """
        schema = get_model(value)
        if schema in (Image2DModel, Image3DModel):
            self.add_image(key, value)
        elif schema in (Labels2DModel, Labels3DModel):
            self.add_labels(key, value)
        elif schema == PointsModel:
            self.add_points(key, value)
        elif schema == ShapesModel:
            self.add_shapes(key, value)
        elif schema == TableModel:
            self.add_tables(table_name=key, table=value)
        else:
            raise TypeError(f"Unknown element type with schema{schema!r}")


class QueryManager:
    """Perform queries on SpatialData objects."""

    def __init__(self, sdata: SpatialData):
        self._sdata = sdata

    def bounding_box(
        self,
        axes: tuple[str, ...],
        min_coordinate: ArrayLike,
        max_coordinate: ArrayLike,
        target_coordinate_system: str,
        filter_table: bool = True,
    ) -> SpatialData:
        """
        Perform a bounding box query on the SpatialData object.

        Parameters
        ----------
        axes
            The axes `min_coordinate` and `max_coordinate` refer to.
        min_coordinate
            The minimum coordinates of the bounding box.
        max_coordinate
            The maximum coordinates of the bounding box.
        target_coordinate_system
            The coordinate system the bounding box is defined in.
        filter_table
            If `True`, the table is filtered to only contain rows that are annotating regions
            contained within the bounding box.

        Returns
        -------
        The SpatialData object containing the requested data.
        Elements with no valid data are omitted.
        """
        from spatialdata._core.query.spatial_query import bounding_box_query

        return bounding_box_query(  # type: ignore[return-value]
            self._sdata,
            axes=axes,
            min_coordinate=min_coordinate,
            max_coordinate=max_coordinate,
            target_coordinate_system=target_coordinate_system,
            filter_table=filter_table,
        )

    def __call__(self, request: BaseSpatialRequest, **kwargs) -> SpatialData:  # type: ignore[no-untyped-def]
        from spatialdata._core.query.spatial_query import BoundingBoxRequest

        if not isinstance(request, BoundingBoxRequest):
            raise TypeError("unknown request type")
        # TODO: request doesn't contain filter_table. If the user doesn't specify this in kwargs, it will be set
        #  to it's default value. This could be a bit unintuitive and
        #  we may want to change make things more explicit.
        return self.bounding_box(**request.to_dict(), **kwargs)
