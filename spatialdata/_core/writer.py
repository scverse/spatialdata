# the code in this file originates from the ome-ngff-tables-prototype repo, we will gradually distribute it's content
# inside the "building block" spatial data classes
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd
import zarr
from anndata import AnnData
from anndata.experimental import write_elem
from ome_zarr.format import Format
from ome_zarr.io import parse_url
from ome_zarr.types import JSONDict
from ome_zarr.writer import (
    _get_valid_axes,
    _validate_datasets,
    write_image,
    write_labels,
)

from spatialdata.format import SpatialDataFormat


def _write_metadata(
    group: zarr.Group,
    group_type: str,
    shape: Tuple[int, ...],
    attr: Optional[Mapping[str, Optional[str]]] = MappingProxyType({"attr": "X", "key": None}),
    fmt: Format = SpatialDataFormat(),
    axes: Union[str, List[str], List[Dict[str, str]]] = None,
    coordinate_transformations: List[List[Dict[str, Any]]] = None,
    **metadata: Union[str, JSONDict, List[JSONDict]],
):
    """Write metdata to a group."""
    dims = len(shape)
    axes = _get_valid_axes(dims, axes, fmt)

    datasets: List[dict] = []
    datasets.append({"path": attr})

    if coordinate_transformations is None:
        shape = [shape]
        coordinate_transformations = fmt.generate_coordinate_transformations(shape)

    fmt.validate_coordinate_transformations(dims, 1, coordinate_transformations)
    if coordinate_transformations is not None:
        for dataset, transform in zip(datasets, coordinate_transformations):
            dataset["coordinateTransformations"] = transform

    if axes is not None:
        axes = _get_valid_axes(axes=axes, fmt=fmt)
        if axes is not None:
            ndim = len(axes)

    multiscales = [
        dict(
            version=fmt.version,
            datasets=_validate_datasets(datasets, ndim, fmt),
            **metadata,
        )
    ]
    if axes is not None:
        multiscales[0]["axes"] = axes

    group.attrs["@type"] = group_type
    group.attrs["multiscales"] = multiscales


def write_points(
    group: zarr.Group,
    points: AnnData,
    group_name: str = "points",
    group_type: str = "ngff:points",
    fmt: Format = SpatialDataFormat(),
    axes: Union[str, List[str], List[Dict[str, str]]] = None,
    coordinate_transformations: List[List[Dict[str, Any]]] = None,
    **metadata: Union[str, JSONDict, List[JSONDict]],
):
    # TODO: validate
    write_elem(group, group_name, points)
    points_group = group[group_name]
    _write_metadata(
        points_group,
        group_type=group_type,
        shape=points.shape,
        attr={"attr": "X", "key": None},
        fmt=fmt,
        axes=axes,
        coordinate_transformations=coordinate_transformations,
        **metadata,
    )


def write_shapes(
    group: zarr.Group,
    shapes: AnnData,
    shapes_parameters: Mapping[str, str],
    group_name: str = "shapes",
    group_type: str = "ngff:shapes",
    fmt: Format = SpatialDataFormat(),
    axes: Union[str, List[str], List[Dict[str, str]]] = None,
    coordinate_transformations: List[List[Dict[str, Any]]] = None,
    **metadata: Union[str, JSONDict, List[JSONDict]],
):
    # TODO: validate
    write_elem(group, group_name, shapes)
    shapes_group = group[group_name]
    fmt.validate_shapes_parameters(shapes_parameters, coordinate_transformations)
    shapes_group.attrs["shapes_parameters"] = shapes_parameters
    _write_metadata(
        shapes_group,
        group_type=group_type,
        shape=shapes.shape,
        attr={"attr": "X", "key": None},
        fmt=fmt,
        axes=axes,
        coordinate_transformations=coordinate_transformations,
        **metadata,
    )


def write_table_regions(
    group: zarr.Group,
    adata: AnnData,
    table_group_name: str = "regions_table",
    group_type: str = "ngff:regions_table",
    region: Union[str, List[str]] = "features",
    region_key: Optional[str] = None,
    instance_key: Optional[str] = None,
):
    write_elem(group, table_group_name, adata)
    table_group = group[table_group_name]
    table_group.attrs["@type"] = group_type
    table_group.attrs["region"] = region
    table_group.attrs["region_key"] = region_key
    table_group.attrs["instance_key"] = instance_key


def write_points_dataset(
    file_path: str,
    image: Optional[np.ndarray] = None,
    image_chunks: Union[Tuple[Any, ...], int] = None,
    image_axes: Union[str, List[str]] = None,
    points: Optional[pd.DataFrame] = None,
    points_dense_columns: Optional[List[str]] = None,
    points_var: Optional[pd.DataFrame] = None,
    points_group_name: str = "points",
):
    """Write a dataset with image and points to ome-zarr

    Parameters
    ----------
    file_path : str
        path to save the zarr file
    image : Optional[np.ndarray]
        image array to save. if None, no image is saved.
        Default value is None.
    image_chunks : Union[Tuple[Any, ...], int]
        Chunking for the image data. See ome-zarr-py for details.
    image_axes : Union[str, List[str]]
        The labels for the image axes. See ome-zarr-py for details.
    points : Optional[pd.DataFrame]
        Table of points to store.
    points_dense_columns : List[str]
        THe names of the columns in points to store in the dense array.
        Must be the names of the coordinate columns.
    points_var : Optional[pd.DataFrame]
        The annotations for the columns of the dense matrix X that
        get stored in AnnData.var. Must be ordered to match
        dense_columns.
    points_group_name : str
        The name of the group to store the points in.
        Default value is 'points'
    """
    # create the zarr root
    store = parse_url(file_path, mode="w").store
    root = zarr.group(store=store)

    if image is not None:
        # if necessary add the image
        write_image(image=image, group=root, chunks=image_chunks, axes=image_axes)

    if points is not None:
        points_group = root.create_group(points_group_name)
        points_anndata = df_to_anndata(df=points, dense_columns=points_dense_columns, var=points_var)
        write_points(points_group, points_anndata)

        # add a flag to the group attrs to denote this is points data
        points_group.attrs["@type"] = "ngff:points"


def write_spatial_anndata(
    file_path: str,
    # image group
    image: Optional[np.ndarray] = None,
    image_chunks: Union[Tuple[Any, ...], int] = None,
    image_axes: Union[str, List[str]] = None,
    image_translation: Optional[np.array] = None,
    image_scale_factors: Optional[np.array] = None,
    # label group
    label_image: Optional[np.ndarray] = None,
    label_name: str = "label_image",
    # table group
    tables_adata: Optional[AnnData] = None,
    tables_region: Optional[Union[str, List[str]]] = None,
    tables_region_key: Optional[str] = None,
    tables_instance_key: Optional[str] = None,
    # shape group
    circles_adata: Optional[AnnData] = None,
    polygons_adata: Optional[AnnData] = None,
    # points group
    points_adata: Optional[AnnData] = None,
):
    """Write a spatial anndata object to ome-zarr
    Parameters
    ----------
    file_path : str
        path to save the zarr file
    image : Optional[np.ndarray]
        image array to save. if None, no image is saved.
        Default value is None.
    image_chunks : Union[Tuple[Any, ...], int]
        Chunking for the image data. See ome-zarr-py for details.
    image_axes : Union[str, List[str]]
        The labels for the image axes. See ome-zarr-py for details.
    image_translation: Optional[np.array]
        Translation values, the length is the number of xyz axes present (e.g. 2 for xy images)
    image_scale_factors: Optional[np.array]
        Scaling factors for each axis, the length is the number of xyz axes present (e.g. 2 for xy images)
    label_name : Union[str, List[str]]
        The name of the label image. See ome-zarr-py for details.
    label_image : Union[str, List[str]]
        The label image (i.e. segmentation mask). See ome-zarr-py for details.
    tables_adata:
        The :class:`anndata.AnnData` table with gene expression and annotations.
    tables_region
        The :class:`anndata.AnnData` region table that maps to one (or more) label image.
    tables_region_key
        The key in :attr:`AnnData.obs` that stores unique pointers to one (or more) label image.
    tables_instance_key
        The key in :attr:`AnnData.obs` that stores unique pointers to regions.
    circles_adata
        The :class:`anndata.AnnData` circle table that store coordinates of circles and metadata.
    polygons_adata
        The :class:`anndata.AnnData` polygons table that store coordinates of polygons and metadata.
    points_adata
        The :class:`anndata.AnnData` point table that store coordinates of points and metadata.

    Returns
    -------
    Nothing, save the file in Zarr store.
    """
    # create the zarr root
    store = parse_url(file_path, mode="w").store
    root = zarr.group(store=store)

    if image is not None:
        from ome_zarr.format import CurrentFormat, Format
        from ome_zarr.scale import Scaler

        downscale = 2
        max_layer = min(4, min(int(np.log2(image.shape[0])), int(np.log2(image.shape[1]))))
        scaler = Scaler(downscale=downscale, max_layer=max_layer)
        fmt: Format = CurrentFormat()
        coordinate_transformations = None
        if image_translation is not None or image_scale_factors is not None:
            translation_values = []
            for c in image_axes:
                if c == "x":
                    translation_values.append(image_translation[0])
                elif c == "y":
                    translation_values.append(image_translation[1])
                else:
                    translation_values.append(0.0)

            # Even if it is not efficient, lest apply the scaler to dummy data to be sure that the transformations are
            # correct
            from ome_zarr.writer import _create_mip

            mip, axes = _create_mip(image, fmt, scaler, image_axes)
            shapes = [data.shape for data in mip]
            pyramid_coordinate_transformations = fmt.generate_coordinate_transformations(shapes)
            coordinate_transformations = []
            for p in pyramid_coordinate_transformations:
                assert p[0]["type"] == "scale"
                pyramid_scale = p[0]["scale"]
                if image_scale_factors is None:
                    image_scale_factors = np.array([1.0, 1.0])
                # there is something wrong somewhere
                # matplotlib shows that the scaling factor is fine, so better create an small artificial dataset and
                # see where is the error
                # hack_factor = 1.55
                hack_factor = 1
                new_scale = (np.array(pyramid_scale) * np.flip(image_scale_factors) * hack_factor).tolist()
                p[0]["scale"] = new_scale
                translation = [{"type": "translation", "translation": translation_values}]
                transformation_series = p + translation
                coordinate_transformations.append(transformation_series)
                image = np.transpose(image)

        write_image(
            image=image,
            group=root,
            chunks=image_chunks,
            axes=image_axes,
            coordinate_transformations=coordinate_transformations,
            scaler=scaler,
        )

    if label_image is not None:
        # i.e. segmentation raster masks
        # the function write labels will create the group labels, so we pass the root
        write_labels(label_image, group=root, name=label_name)

    if tables_adata is not None:
        # e.g. expression table
        tables_group = root.create_group(name="tables")
        write_table_regions(
            group=tables_group,
            adata=tables_adata,
            region=tables_region,
            region_key=tables_region_key,
            instance_key=tables_instance_key,
        )
    if circles_adata is not None:
        # was it called circles? I didn't take a pic of the whiteboard
        circles_group = root.create_group(name="circles")
        write_table_circles(
            group=circles_group,
            adata=circles_adata,
        )
    if polygons_adata is not None:
        polygons_group = root.create_group(name="polygons")
        write_table_polygons(
            group=polygons_group,
            adata=polygons_adata,
        )
    if points_adata is not None:
        points_group = root.create_group(name="points")
        write_points(
            group=points_group,
            adata=points_adata,
        )


def write_table_polygons(
    group: zarr.Group,
    adata: AnnData,
    table_group_name: str = "polygons_table",
    group_type: str = "ngff:polygons_table",
):
    write_elem(group, table_group_name, adata)
    table_group = group[table_group_name]
    table_group.attrs["@type"] = group_type
