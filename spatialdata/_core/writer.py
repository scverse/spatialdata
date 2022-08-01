from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import zarr
from anndata import AnnData
from anndata.experimental import write_elem as write_adata
from ome_zarr.format import CurrentFormat, Format
from ome_zarr.scale import Scaler
from ome_zarr.types import JSONDict
from ome_zarr.writer import _get_valid_axes, _validate_datasets
from ome_zarr.writer import write_image as write_image_ngff

from spatialdata._types import ArrayLike
from spatialdata.format import SpatialDataFormat

__all__ = ["write_points", "write_shapes", "write_tables", "write_image", "write_labels"]


def _write_metadata(
    group: zarr.Group,
    group_type: str,
    shape: Tuple[int, ...],
    attr: Optional[Mapping[str, Optional[str]]] = MappingProxyType({"attr": "X", "key": None}),
    fmt: Format = SpatialDataFormat(),
    axes: Optional[Union[str, List[str], List[Dict[str, str]]]] = None,
    coordinate_transformations: Optional[List[List[Dict[str, Any]]]] = None,
    **metadata: Union[str, JSONDict, List[JSONDict]],
) -> None:
    """Write metdata to a group."""
    dims = len(shape)
    axes = _get_valid_axes(dims, axes, fmt)

    datasets: List[Dict[str, Any]] = []
    datasets.append({"path": attr})

    if coordinate_transformations is None:
        shape = [shape]  # type: ignore[assignment]
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
    axes: Optional[Union[str, List[str], List[Dict[str, str]]]] = None,
    coordinate_transformations: Optional[List[List[Dict[str, Any]]]] = None,
    **metadata: Union[str, JSONDict, List[JSONDict]],
) -> None:
    # TODO: validate
    write_adata(group, group_name, points)
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
    axes: Optional[Union[str, List[str], List[Dict[str, str]]]] = None,
    coordinate_transformations: Optional[List[List[Dict[str, Any]]]] = None,
    **metadata: Union[str, JSONDict, List[JSONDict]],
) -> None:
    fmt.validate_shapes_parameters(shapes_parameters)
    write_adata(group, group_name, shapes)
    shapes_group = group[group_name]
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


def write_tables(
    group: zarr.Group,
    tables: AnnData,
    fmt: Format = SpatialDataFormat(),
    group_name: str = "regions_table",
    group_type: str = "ngff:regions_table",
    region: Union[str, List[str]] = "features",
    region_key: Optional[str] = None,
    instance_key: Optional[str] = None,
) -> None:
    fmt.validate_tables(tables, region_key, instance_key)
    write_adata(group, group_name, tables)
    tables_group = group[group_name]
    tables_group.attrs["@type"] = group_type
    tables_group.attrs["region"] = region
    tables_group.attrs["region_key"] = region_key
    tables_group.attrs["instance_key"] = instance_key


def write_image(
    image: ArrayLike,
    group: zarr.Group,
    scaler: Scaler = Scaler(),
    chunks: Optional[Union[Tuple[Any, ...], int]] = None,
    fmt: Format = CurrentFormat(),
    axes: Optional[Union[str, List[str], List[Dict[str, str]]]] = None,
    coordinate_transformations: Optional[List[List[Dict[str, Any]]]] = None,
    storage_options: Optional[Union[JSONDict, List[JSONDict]]] = None,
    **metadata: Union[str, JSONDict, List[JSONDict]],
) -> None:
    # TODO: ergonomics.
    write_image_ngff(
        image=image,
        group=group,
        scaler=scaler,
        chunks=chunks,
        fmt=fmt,
        axes=axes,
        coordinate_transformations=coordinate_transformations,
        storage_options=storage_options,
        **metadata,
    )


def write_labels(
    labels: ArrayLike,
    group: zarr.Group,
    name: str,
    scaler: Scaler = Scaler(),
    chunks: Optional[Union[Tuple[Any, ...], int]] = None,
    fmt: Format = CurrentFormat(),
    axes: Optional[Union[str, List[str], List[Dict[str, str]]]] = None,
    coordinate_transformations: Optional[List[List[Dict[str, Any]]]] = None,
    storage_options: Optional[Union[JSONDict, List[JSONDict]]] = None,
    label_metadata: Optional[JSONDict] = None,
    **metadata: JSONDict,
) -> None:
    # TODO: ergonomics.
    write_image_ngff(
        labels=labels,
        group=group,
        name=name,
        scaler=scaler,
        chunks=chunks,
        fmt=fmt,
        axes=axes,
        coordinate_transformations=coordinate_transformations,
        storage_options=storage_options,
        label_metadata=label_metadata,
        **metadata,
    )


# def write_spatial_anndata(
#     file_path: str,
#     # image group
#     image: Optional[np.ndarray] = None,
#     image_chunks: Union[Tuple[Any, ...], int] = None,
#     image_axes: Union[str, List[str]] = None,
#     image_translation: Optional[np.array] = None,
#     image_scale_factors: Optional[np.array] = None,
#     # label group
#     label_image: Optional[np.ndarray] = None,
#     label_name: str = "label_image",
#     # table group
#     tables_adata: Optional[AnnData] = None,
#     tables_region: Optional[Union[str, List[str]]] = None,
#     tables_region_key: Optional[str] = None,
#     tables_instance_key: Optional[str] = None,
#     # shape group
#     circles_adata: Optional[AnnData] = None,
#     polygons_adata: Optional[AnnData] = None,
#     # points group
#     points_adata: Optional[AnnData] = None,
# ):
#     """Write a spatial anndata object to ome-zarr
#     Parameters
#     ----------
#     file_path : str
#         path to save the zarr file
#     image : Optional[np.ndarray]
#         image array to save. if None, no image is saved.
#         Default value is None.
#     image_chunks : Union[Tuple[Any, ...], int]
#         Chunking for the image data. See ome-zarr-py for details.
#     image_axes : Union[str, List[str]]
#         The labels for the image axes. See ome-zarr-py for details.
#     image_translation: Optional[np.array]
#         Translation values, the length is the number of xyz axes present (e.g. 2 for xy images)
#     image_scale_factors: Optional[np.array]
#         Scaling factors for each axis, the length is the number of xyz axes present (e.g. 2 for xy images)
#     label_name : Union[str, List[str]]
#         The name of the label image. See ome-zarr-py for details.
#     label_image : Union[str, List[str]]
#         The label image (i.e. segmentation mask). See ome-zarr-py for details.
#     tables_adata:
#         The :class:`anndata.AnnData` table with gene expression and annotations.
#     tables_region
#         The :class:`anndata.AnnData` region table that maps to one (or more) label image.
#     tables_region_key
#         The key in :attr:`AnnData.obs` that stores unique pointers to one (or more) label image.
#     tables_instance_key
#         The key in :attr:`AnnData.obs` that stores unique pointers to regions.
#     circles_adata
#         The :class:`anndata.AnnData` circle table that store coordinates of circles and metadata.
#     polygons_adata
#         The :class:`anndata.AnnData` polygons table that store coordinates of polygons and metadata.
#     points_adata
#         The :class:`anndata.AnnData` point table that store coordinates of points and metadata.

#     Returns
#     -------
#     Nothing, save the file in Zarr store.
#     """
#     # create the zarr root
#     store = parse_url(file_path, mode="w").store
#     root = zarr.group(store=store)

#     if image is not None:
#         from ome_zarr.format import CurrentFormat, Format
#         from ome_zarr.scale import Scaler

#         downscale = 2
#         max_layer = min(4, min(int(np.log2(image.shape[0])), int(np.log2(image.shape[1]))))
#         scaler = Scaler(downscale=downscale, max_layer=max_layer)
#         fmt: Format = CurrentFormat()
#         coordinate_transformations = None
#         if image_translation is not None or image_scale_factors is not None:
#             translation_values = []
#             for c in image_axes:
#                 if c == "x":
#                     translation_values.append(image_translation[0])
#                 elif c == "y":
#                     translation_values.append(image_translation[1])
#                 else:
#                     translation_values.append(0.0)

#             # Even if it is not efficient, lest apply the scaler to dummy data to be sure that the transformations are
#             # correct
#             from ome_zarr.writer import _create_mip

#             mip, axes = _create_mip(image, fmt, scaler, image_axes)
#             shapes = [data.shape for data in mip]
#             pyramid_coordinate_transformations = fmt.generate_coordinate_transformations(shapes)
#             coordinate_transformations = []
#             for p in pyramid_coordinate_transformations:
#                 assert p[0]["type"] == "scale"
#                 pyramid_scale = p[0]["scale"]
#                 if image_scale_factors is None:
#                     image_scale_factors = np.array([1.0, 1.0])
#                 # there is something wrong somewhere
#                 # matplotlib shows that the scaling factor is fine, so better create an small artificial dataset and
#                 # see where is the error
#                 # hack_factor = 1.55
#                 hack_factor = 1
#                 new_scale = (np.array(pyramid_scale) * np.flip(image_scale_factors) * hack_factor).tolist()
#                 p[0]["scale"] = new_scale
#                 translation = [{"type": "translation", "translation": translation_values}]
#                 transformation_series = p + translation
#                 coordinate_transformations.append(transformation_series)
#                 image = np.transpose(image)

#         write_image(
#             image=image,
#             group=root,
#             chunks=image_chunks,
#             axes=image_axes,
#             coordinate_transformations=coordinate_transformations,
#             scaler=scaler,
#         )

#     if label_image is not None:
#         # i.e. segmentation raster masks
#         # the function write labels will create the group labels, so we pass the root
#         write_labels(label_image, group=root, name=label_name)

#     if tables_adata is not None:
#         # e.g. expression table
#         tables_group = root.create_group(name="tables")
#         write_table_regions(
#             group=tables_group,
#             adata=tables_adata,
#             region=tables_region,
#             region_key=tables_region_key,
#             instance_key=tables_instance_key,
#         )
#     if circles_adata is not None:
#         # was it called circles? I didn't take a pic of the whiteboard
#         circles_group = root.create_group(name="circles")
#         write_table_circles(
#             group=circles_group,
#             adata=circles_adata,
#         )
#     if polygons_adata is not None:
#         polygons_group = root.create_group(name="polygons")
#         write_table_polygons(
#             group=polygons_group,
#             adata=polygons_adata,
#         )
#     if points_adata is not None:
#         points_group = root.create_group(name="points")
#         write_points(
#             group=points_group,
#             adata=points_adata,
#         )


def write_table_polygons(
    group: zarr.Group,
    adata: AnnData,
    table_group_name: str = "polygons_table",
    group_type: str = "ngff:polygons_table",
) -> None:
    write_adata(group, table_group_name, adata)
    table_group = group[table_group_name]
    table_group.attrs["@type"] = group_type
