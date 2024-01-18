from __future__ import annotations

import hashlib
import os
from collections.abc import Generator
from itertools import chain
from pathlib import Path
from typing import TYPE_CHECKING, Any

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

from spatialdata._core._elements import Images, Labels, Points, Shapes
from spatialdata._logging import logger
from spatialdata._types import ArrayLike, Raster_T
from spatialdata._utils import _error_message_add_element
from spatialdata.models import (
    Image2DModel,
    Image3DModel,
    Labels2DModel,
    Labels3DModel,
    PointsModel,
    ShapesModel,
    TableModel,
    get_model,
)
from spatialdata.models._utils import SpatialElement, get_axes_names

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

    def __init__(
        self,
        images: dict[str, Raster_T] | None = None,
        labels: dict[str, Raster_T] | None = None,
        points: dict[str, DaskDataFrame] | None = None,
        shapes: dict[str, GeoDataFrame] | None = None,
        table: AnnData | None = None,
    ) -> None:
        self._path: Path | None = None

        self._shared_keys: set[str | None] = set()
        self._images: Images = Images(shared_keys=self._shared_keys)
        self._labels: Labels = Labels(shared_keys=self._shared_keys)
        self._points: Points = Points(shared_keys=self._shared_keys)
        self._shapes: Shapes = Shapes(shared_keys=self._shared_keys)
        self._table: AnnData | None = None

        self._validate_unique_element_names(
            list(chain.from_iterable([e.keys() for e in [images, labels, points, shapes] if e is not None]))
        )

        if images is not None:
            for k, v in images.items():
                self.images[k] = v

        if labels is not None:
            for k, v in labels.items():
                self.labels[k] = v

        if shapes is not None:
            for k, v in shapes.items():
                self.shapes[k] = v

        if points is not None:
            for k, v in points.items():
                self.points[k] = v

        if table is not None:
            Table_s.validate(table)
            self._table = table

        self._query = QueryManager(self)

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
            "table": None,
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
                if d["table"] is not None:
                    raise ValueError("Only one table can be present in the dataset.")
                d["table"] = e
            else:
                raise ValueError(f"Unknown schema {schema}")
        return SpatialData(**d)  # type: ignore[arg-type]

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
        This function calls :func:`spatialdata.aggregate` with the convenience that `values` and `by` can be string
        without having to specify the `values_sdata` and `by_sdata`, which in that case will be replaced by `self`.

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
            raise KeyError(
                f"Element names must be unique. The following element names are used multiple times: {duplicates}"
            )

    def is_backed(self) -> bool:
        """Check if the data is backed by a Zarr storage or if it is in-memory."""
        return self._path is not None

    @property
    def path(self) -> Path | None:
        """Path to the Zarr storage."""
        return self._path

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
        either the existing Zarr sub-group or a new one.
        """
        store = parse_url(self.path, mode="r+").store
        root = zarr.group(store=store)
        assert element_type in ["images", "labels", "points", "polygons", "shapes"]
        element_type_group = root.require_group(element_type)
        return element_type_group.require_group(name)

    def _init_add_element(self, name: str, element_type: str, overwrite: bool) -> zarr.Group:
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

    def locate_element(self, element: SpatialElement) -> list[str]:
        """
        Locate a SpatialElement within the SpatialData object and returns its Zarr paths relative to the root.

        Parameters
        ----------
        element
            The queried SpatialElement

        Returns
        -------
        A list of Zarr paths of the element relative to the root (multiple copies of the same element are allowed).
        The list is empty if the element is not present.
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
            return []
        return [f"{found_element_type[i]}/{found_element_name[i]}" for i in range(len(found))]

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
        located = self.locate_element(element)
        if len(located) == 0:
            raise ValueError(
                "Cannot save the transformation to the element as it has not been found in the SpatialData object"
            )
        if self.path is not None:
            for path in located:
                found_element_type, found_element_name = path.split("/")
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

                    overwrite_coordinate_transformations_non_raster(
                        group=group, axes=axes, transformations=transformations
                    )
                else:
                    raise ValueError("Unknown element type")

    def filter_by_coordinate_system(self, coordinate_system: str | list[str], filter_table: bool = True) -> SpatialData:
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
        from spatialdata._core.query.relational_query import _filter_table_by_coordinate_system
        from spatialdata.transformations.operations import get_transformation

        elements: dict[str, dict[str, SpatialElement]] = {}
        element_paths_in_coordinate_system = []
        if isinstance(coordinate_system, str):
            coordinate_system = [coordinate_system]
        for element_type, element_name, element in self._gen_elements():
            transformations = get_transformation(element, get_all=True)
            assert isinstance(transformations, dict)
            for cs in coordinate_system:
                if cs in transformations:
                    if element_type not in elements:
                        elements[element_type] = {}
                    elements[element_type][element_name] = element
                    element_paths_in_coordinate_system.append(element_name)

        if filter_table:
            table = _filter_table_by_coordinate_system(self.table, element_paths_in_coordinate_system)
        else:
            table = self.table

        return SpatialData(**elements, table=table)

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
            transformed = sdata.transform_element_to_coordinate_system(element, target_coordinate_system)
            if element_type not in elements:
                elements[element_type] = {}
            elements[element_type][element_name] = transformed
        return SpatialData(**elements, table=sdata.table)

    def write(
        self,
        file_path: str | Path,
        storage_options: JSONDict | list[JSONDict] | None = None,
        overwrite: bool = False,
        consolidate_metadata: bool = True,
    ) -> None:
        from spatialdata._io import write_image, write_labels, write_points, write_shapes, write_table
        from spatialdata._io._utils import get_dask_backing_files

        """Write the SpatialData object to Zarr."""
        if isinstance(file_path, str):
            file_path = Path(file_path)
        assert isinstance(file_path, Path)

        if self.is_backed() and str(self.path) != str(file_path):
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
            if not overwrite:
                raise ValueError("The Zarr store already exists. Use `overwrite=True` to overwrite the store.")
            if self.is_backed() and str(self.path) == str(file_path):
                raise ValueError(
                    "The file path specified is the same as the one used for backing. "
                    "Overwriting the backing file is not supported to prevent accidental data loss."
                    "We are discussing how to support this use case in the future, if you would like us to "
                    "support it please leave a comment on https://github.com/scverse/spatialdata/pull/138"
                )
            if any(Path(fp).resolve().is_relative_to(file_path.resolve()) for fp in get_dask_backing_files(self)):
                raise ValueError(
                    "The file path specified is a parent directory of one or more files used for backing for one or "
                    "more elements in the SpatialData object. You can either load every element of the SpatialData "
                    "object in memory, or save the current spatialdata object to a different path."
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
        self._path = Path(file_path)
        try:
            if len(self.images):
                root.create_group(name="images")
                # add_image_in_memory will delete and replace the same key in self.images,
                # so we need to make a copy of the keys. Same for the other elements
                keys = self.images.keys()

                for name in keys:
                    elem_group = self._init_add_element(name=name, element_type="images", overwrite=overwrite)
                    write_image(
                        image=self.images[name],
                        group=elem_group,
                        name=name,
                        storage_options=storage_options,
                    )

                    # TODO(giovp): fix or remove
                    # reload the image from the Zarr storage so that now the element is lazy loaded,
                    # and most importantly, from the correct storage
                    # element_path = Path(self.path) / "images" / name
                    # _read_multiscale(element_path, raster_type="image")

            if len(self.labels):
                root.create_group(name="labels")
                # keys = list(self.labels.keys())
                keys = self.labels.keys()

                for name in keys:
                    elem_group = self._init_add_element(name=name, element_type="labels", overwrite=overwrite)
                    write_labels(
                        labels=self.labels[name],
                        group=elem_group,
                        name=name,
                        storage_options=storage_options,
                    )

                    # TODO(giovp): fix or remove
                    # reload the labels from the Zarr storage so that now the element is lazy loaded,
                    #  and most importantly, from the correct storage
                    # element_path = Path(self.path) / "labels" / name
                    # _read_multiscale(element_path, raster_type="labels")

            if len(self.points):
                root.create_group(name="points")
                # keys = list(self.points.keys())
                keys = self.points.keys()

                for name in keys:
                    elem_group = self._init_add_element(name=name, element_type="points", overwrite=overwrite)
                    write_points(
                        points=self.points[name],
                        group=elem_group,
                        name=name,
                    )
                    # TODO(giovp): fix or remove
                    # element_path = Path(self.path) / "points" / name

                    # # reload the points from the Zarr storage so that the element is lazy loaded,
                    # # and most importantly, from the correct storage
                    # _read_points(element_path)

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

            if self.table is not None:
                elem_group = root.create_group(name="table")
                write_table(table=self.table, group=elem_group, name="table")

        except Exception as e:  # noqa: B902
            self._path = None
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
        assert isinstance(self.path, Path)

    @property
    def table(self) -> AnnData:
        """
        Return the table.

        Returns
        -------
        The table.
        """
        return self._table

    @table.setter
    def table(self, table: AnnData) -> None:
        """
        Set the table of a SpatialData object in a object that doesn't contain a table.

        Parameters
        ----------
        table
            The table to set.

        Notes
        -----
        If a table is already present, it needs to be removed first.
        The table needs to pass validation (see :class:`~spatialdata.TableModel`).
        If the SpatialData object is backed by a Zarr storage, the table will be written to the Zarr storage.
        """
        from spatialdata._io.io_table import write_table

        TableModel().validate(table)
        if self.table is not None:
            raise ValueError("The table already exists. Use del sdata.table to remove it first.")
        self._table = table
        if self.is_backed():
            store = parse_url(self.path, mode="r+").store
            root = zarr.group(store=store)
            elem_group = root.require_group(name="table")
            write_table(table=self.table, group=elem_group, name="table")

    @table.deleter
    def table(self) -> None:
        """Delete the table."""
        self._table = None
        if self.is_backed():
            store = parse_url(self.path, mode="r+").store
            root = zarr.group(store=store)
            del root["table/table"]

    @staticmethod
    def read(file_path: Path | str, selection: tuple[str] | None = None) -> SpatialData:
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

    def add_image(
        self,
        name: str,
        image: SpatialImage | MultiscaleSpatialImage,
        storage_options: JSONDict | list[JSONDict] | None = None,
        overwrite: bool = False,
    ) -> None:
        _error_message_add_element()

    def add_labels(
        self,
        name: str,
        labels: SpatialImage | MultiscaleSpatialImage,
        storage_options: JSONDict | list[JSONDict] | None = None,
        overwrite: bool = False,
    ) -> None:
        _error_message_add_element()

    def add_points(
        self,
        name: str,
        points: DaskDataFrame,
        overwrite: bool = False,
    ) -> None:
        _error_message_add_element()

    def add_shapes(
        self,
        name: str,
        shapes: GeoDataFrame,
        overwrite: bool = False,
    ) -> None:
        _error_message_add_element()

    @property
    def images(self) -> Images:
        """Return images as a Dict of name to image data."""
        return self._images

    @images.setter
    def images(self, images: dict[str, Raster_T]) -> None:
        """Set images."""
        self._shared_keys = self._shared_keys - set(self._images.keys())
        self._images = Images(shared_keys=self._shared_keys)
        for k, v in images.items():
            self._images[k] = v

    @property
    def labels(self) -> Labels:
        """Return labels as a Dict of name to label data."""
        return self._labels

    @labels.setter
    def labels(self, labels: dict[str, Raster_T]) -> None:
        """Set labels."""
        self._shared_keys = self._shared_keys - set(self._labels.keys())
        self._labels = Labels(shared_keys=self._shared_keys)
        for k, v in labels.items():
            self._labels[k] = v

    @property
    def points(self) -> Points:
        """Return points as a Dict of name to point data."""
        return self._points

    @points.setter
    def points(self, points: dict[str, DaskDataFrame]) -> None:
        """Set points."""
        self._shared_keys = self._shared_keys - set(self._points.keys())
        self._points = Points(shared_keys=self._shared_keys)
        for k, v in points.items():
            self._points[k] = v

    @property
    def shapes(self) -> Shapes:
        """Return shapes as a Dict of name to shape data."""
        return self._shapes

    @shapes.setter
    def shapes(self, shapes: dict[str, GeoDataFrame]) -> None:
        """Set shapes."""
        self._shared_keys = self._shared_keys - set(self._shapes.keys())
        self._shapes = Shapes(shared_keys=self._shared_keys)
        for k, v in shapes.items():
            self._shapes[k] = v

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
        all_elements = ["images", "labels", "points", "shapes", "table"]
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
        from spatialdata._utils import _natural_keys

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
            if isinstance(attribute, AnnData):
                descr += f"{h('empty_line')}"
                descr_class = attribute.__class__.__name__
                descr += f"{h('level1.0')}{attribute!r}: {descr_class} {attribute.shape}"
                descr = rreplace(descr, h("level1.0"), "    └── ", 1)
            else:
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
                                + ", ".join(
                                    [str(dim) if not isinstance(dim, Delayed) else "<Delayed>" for dim in v.shape]
                                )
                                + ")"
                            )
                        descr += f"{h(attr + 'level1.1')}{k!r}: {descr_class} " f"with shape: {shape_str} {dim_string}"
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
                            descr += (
                                f"{h(attr + 'level1.1')}{k!r}: {descr_class}[{dims}] " f"{', '.join(map(str, shapes))}"
                            )
                        else:
                            raise TypeError(f"Unknown type {type(v)}")
            if last_attr is True:
                descr = descr.replace(h("empty_line"), "\n  ")
            else:
                descr = descr.replace(h("empty_line"), "\n│ ")

        descr = rreplace(descr, h("level0"), "└── ", 1)
        descr = descr.replace(h("level0"), "├── ")

        for attr in ["images", "labels", "points", "table", "shapes"]:
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
        for element_type in ["images", "labels", "points", "shapes"]:
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
    def init_from_elements(cls, elements: dict[str, SpatialElement], table: AnnData | None = None) -> SpatialData:
        """
        Create a SpatialData object from a dict of named elements and an optional table.

        Parameters
        ----------
        elements
            A dict of named elements.
        table
            An optional table.

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
        return cls(**elements_dict, table=table)

    def subset(self, element_names: list[str], filter_table: bool = True) -> SpatialData:
        """
        Subset the SpatialData object.

        Parameters
        ----------
        element_names
            The names of the element_names to subset.
        filter_table
            If True (default), the table is filtered to only contain rows that are annotating regions
            contained within the element_names.

        Returns
        -------
        The subsetted SpatialData object.
        """
        from spatialdata._core.query.relational_query import _filter_table_by_elements

        elements_dict: dict[str, SpatialElement] = {}
        for element_type, element_name, element in self._gen_elements():
            if element_name in element_names:
                elements_dict.setdefault(element_type, {})[element_name] = element
        table = _filter_table_by_elements(self.table, elements_dict=elements_dict) if filter_table else self.table
        if len(table) == 0:
            table = None
        return SpatialData(**elements_dict, table=table)

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
            self.images[key] = value
        elif schema in (Labels2DModel, Labels3DModel):
            self.labels[key] = value
        elif schema == PointsModel:
            self.points[key] = value
        elif schema == ShapesModel:
            self.shapes[key] = value
        elif schema == TableModel:
            raise TypeError("Use the table property to set the table (e.g. sdata.table = value).")
        else:
            raise TypeError(f"Unknown element type with schema: {schema!r}.")


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
