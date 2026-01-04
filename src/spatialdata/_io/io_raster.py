from pathlib import Path
from typing import Any, Literal

import dask.array as da
import numpy as np
import zarr
from ome_zarr.format import Format
from ome_zarr.io import ZarrLocation
from ome_zarr.reader import Multiscales, Node, Reader
from ome_zarr.types import JSONDict
from ome_zarr.writer import _get_valid_axes
from ome_zarr.writer import write_image as write_image_ngff
from ome_zarr.writer import write_labels as write_labels_ngff
from ome_zarr.writer import write_multiscale as write_multiscale_ngff
from ome_zarr.writer import write_multiscale_labels as write_multiscale_labels_ngff
from xarray import DataArray, Dataset, DataTree

from spatialdata._io._utils import (
    _get_transformations_from_ngff_dict,
    overwrite_coordinate_transformations_raster,
)
from spatialdata._io.format import (
    CurrentRasterFormat,
    RasterFormatType,
    get_ome_zarr_format,
)
from spatialdata._utils import get_pyramid_levels
from spatialdata.models._utils import get_channel_names
from spatialdata.models.models import ATTRS_KEY
from spatialdata.transformations._utils import (
    _get_transformations,
    _get_transformations_xarray,
    _set_transformations,
    compute_coordinates,
)


def _read_multiscale(
    store: str | Path, raster_type: Literal["image", "labels"], reader_format: Format
) -> DataArray | DataTree:
    assert isinstance(store, str | Path)
    assert raster_type in ["image", "labels"]

    nodes: list[Node] = []
    image_loc = ZarrLocation(store, fmt=reader_format)
    if exists := image_loc.exists():
        image_reader = Reader(image_loc)()
        image_nodes = list(image_reader)
        nodes = _get_multiscale_nodes(image_nodes, nodes)
    else:
        raise OSError(
            f"Image location {image_loc} does not seem to exist. If it does, potentially the zarr.json (or .zattrs) "
            f"file inside is corrupted or not present or the image files themselves are corrupted."
        )
    if len(nodes) != 1:
        if not exists:
            raise ValueError(
                f"len(nodes) = {len(nodes)}, expected 1 and image location {image_loc} "
                "does not exist. Unable to read the NGFF file. Please report this bug "
                "and attach a minimal data example."
            )
        raise OSError(
            f"Image location {image_loc} exists, but len(nodes) = {len(nodes)}, expected 1. Element "
            f"{image_loc.basename()} is potentially corrupted. Please report this bug and attach a minimal data "
            f"example."
        )

    node = nodes[0]
    loaded_node = node.load(Multiscales)
    datasets, multiscales = (
        loaded_node.datasets,
        loaded_node.zarr.root_attrs["multiscales"],
    )
    # This works for all versions as in zarr v3 the level of the 'ome' key is taken as root_attrs.
    omero_metadata = loaded_node.zarr.root_attrs.get("omero")
    # TODO: check if below is still valid
    legacy_channels_metadata = node.load(Multiscales).zarr.root_attrs.get("channels_metadata", None)  # legacy v0.1
    assert len(multiscales) == 1
    # checking for multiscales[0]["coordinateTransformations"] would make fail
    # something that doesn't have coordinateTransformations in top level
    # which is true for the current version of the spec
    # and for instance in the xenium example
    encoded_ngff_transformations = multiscales[0]["coordinateTransformations"]
    transformations = _get_transformations_from_ngff_dict(encoded_ngff_transformations)
    # if image, read channels metadata
    channels: list[Any] | None = None
    if raster_type == "image":
        if legacy_channels_metadata is not None:
            channels = [d["label"] for d in legacy_channels_metadata["channels"]]
        if omero_metadata is not None:
            channels = [d["label"] for d in omero_metadata["channels"]]
    axes = [i["name"] for i in node.metadata["axes"]]
    if len(datasets) > 1:
        multiscale_image = {}
        for i, d in enumerate(datasets):
            data = node.load(Multiscales).array(resolution=d)
            multiscale_image[f"scale{i}"] = Dataset(
                {
                    "image": DataArray(
                        data,
                        name="image",
                        dims=axes,
                        coords={"c": channels} if channels is not None else {},
                    )
                }
            )
        msi = DataTree.from_dict(multiscale_image)
        _set_transformations(msi, transformations)
        return compute_coordinates(msi)

    data = node.load(Multiscales).array(resolution=datasets[0])
    si = DataArray(
        data,
        name="image",
        dims=axes,
        coords={"c": channels} if channels is not None else {},
    )
    _set_transformations(si, transformations)
    return compute_coordinates(si)


def _get_multiscale_nodes(image_nodes: list[Node], nodes: list[Node]) -> list[Node]:
    """Get nodes with Multiscales spec from a list of nodes.

    The nodes with the Multiscales spec are the nodes used for reading in image and label data. We only have to check
    the multiscales now, while before we also had to check the label spec. In the new ome-zarr-py though labels can have
    the Label spec, these do not contain the multiscales anymore used to read the data. They can contain label specific
    metadata though.

    Parameters
    ----------
    image_nodes
        List of nodes returned from the ome-zarr-py Reader.
    nodes
        List to append the nodes with the multiscales spec to.

    Returns
    -------
    List of nodes with the multiscales spec.
    """
    if len(image_nodes):
        for node in image_nodes:
            # Labels are now also Multiscales in newer version of ome-zarr-py
            if np.any([isinstance(spec, Multiscales) for spec in node.specs]):
                nodes.append(node)
    return nodes


def _write_raster(
    raster_type: Literal["image", "labels"],
    raster_data: DataArray | DataTree,
    group: zarr.Group,
    name: str,
    raster_format: RasterFormatType,
    storage_options: JSONDict | list[JSONDict] | None = None,
    label_metadata: JSONDict | None = None,
    **metadata: str | JSONDict | list[JSONDict],
) -> None:
    """Write raster data to disk.

    Parameters
    ----------
    raster_type
        Whether the raster data pertains to a image or labels 'SpatialElement`.
    raster_data
        The raster data to write.
    group
        The zarr group in the 'image' or 'labels' zarr group to write the raster data to.
    name: str
        The name of the raster element.
    raster_format
        The format used to write the raster data.
    storage_options
        Additional options for writing the raster data, like chunks and compression.
    label_metadata
        Label metadata which can only be defined when writing 'labels'.
    metadata
        Additional metadata for the raster element
    """
    if raster_type not in ["image", "labels"]:
        raise ValueError(f"{raster_type} is not a valid raster type. Must be 'image' or 'labels'.")
    # "name" and "label_metadata" are only used for labels. "name" is written in write_multiscale_ngff() but ignored in
    # write_image_ngff() (possibly an ome-zarr-py bug). We only use "name" to ensure correct group access in the
    # ome-zarr API.
    if raster_type == "labels":
        metadata["name"] = name
        metadata["label_metadata"] = label_metadata

    # convert channel names to channel metadata in omero
    if raster_type == "image":
        metadata["metadata"] = {"omero": {"channels": []}}
        channels = get_channel_names(raster_data)
        for c in channels:
            metadata["metadata"]["omero"]["channels"].append({"label": c})  # type: ignore[union-attr, index, call-overload]

    if isinstance(raster_data, DataArray):
        _write_raster_dataarray(
            raster_type,
            group,
            name,
            raster_data,
            raster_format,
            storage_options,
            **metadata,
        )
    elif isinstance(raster_data, DataTree):
        _write_raster_datatree(
            raster_type,
            group,
            name,
            raster_data,
            raster_format,
            storage_options,
            **metadata,
        )
    else:
        raise ValueError("Not a valid labels object")

    group = group["labels"][name] if raster_type == "labels" else group
    if ATTRS_KEY not in group.attrs:
        group.attrs[ATTRS_KEY] = {}
    attrs = group.attrs[ATTRS_KEY]
    attrs["version"] = raster_format.spatialdata_format_version
    # triggers the write operation
    group.attrs[ATTRS_KEY] = attrs


def _write_raster_dataarray(
    raster_type: Literal["image", "labels"],
    group: zarr.Group,
    element_name: str,
    raster_data: DataArray,
    raster_format: RasterFormatType,
    storage_options: JSONDict | list[JSONDict] | None = None,
    **metadata: str | JSONDict | list[JSONDict],
) -> None:
    """Write raster data of type DataArray to disk.

    Parameters
    ----------
    raster_type
        Whether the raster data pertains to a image or labels 'SpatialElement`.
    group
        The zarr group in the 'image' or 'labels' zarr group to write the raster data to.
    element_name
        The name of the raster element.
    raster_data
        The raster data to write.
    raster_format
        The format used to write the raster data.
    storage_options
        Additional options for writing the raster data, like chunks and compression.
    metadata
        Additional metadata for the raster element
    """
    write_single_scale_ngff = write_image_ngff if raster_type == "image" else write_labels_ngff

    data = raster_data.data
    transformations = _get_transformations(raster_data)
    if transformations is None:
        raise ValueError(f"{element_name} does not have any transformations and can therefore not be written.")
    input_axes: tuple[str, ...] = tuple(raster_data.dims)
    chunks = raster_data.chunks
    parsed_axes = _get_valid_axes(axes=list(input_axes), fmt=raster_format)
    if storage_options is not None:
        if "chunks" not in storage_options and isinstance(storage_options, dict):
            storage_options["chunks"] = chunks
    else:
        storage_options = {"chunks": chunks}
    # Scaler needs to be None since we are passing the data already downscaled for the multiscale case.
    # We need  this because the argument of write_image_ngff is called image while the argument of
    # write_labels_ngff is called label.
    metadata[raster_type] = data
    ome_zarr_format = get_ome_zarr_format(raster_format)
    write_single_scale_ngff(
        group=group,
        scaler=None,
        fmt=ome_zarr_format,
        axes=parsed_axes,
        coordinate_transformations=None,
        storage_options=storage_options,
        **metadata,
    )

    trans_group = group["labels"][element_name] if raster_type == "labels" else group
    overwrite_coordinate_transformations_raster(
        group=trans_group,
        transformations=transformations,
        axes=input_axes,
        raster_format=raster_format,
    )


def _write_raster_datatree(
    raster_type: Literal["image", "labels"],
    group: zarr.Group,
    element_name: str,
    raster_data: DataTree,
    raster_format: RasterFormatType,
    storage_options: JSONDict | list[JSONDict] | None = None,
    **metadata: str | JSONDict | list[JSONDict],
) -> None:
    """Write raster data of type DataTree to disk.

    Parameters
    ----------
    raster_type
        Whether the raster data pertains to a image or labels 'SpatialElement`.
    group
        The zarr group in the 'image' or 'labels' zarr group to write the raster data to.
    element_name
        The name of the raster element.
    raster_data
        The raster data to write.
    raster_format
        The format used to write the raster data.
    storage_options
        Additional options for writing the raster data, like chunks and compression.
    metadata
        Additional metadata for the raster element
    """
    write_multi_scale_ngff = write_multiscale_ngff if raster_type == "image" else write_multiscale_labels_ngff
    data = get_pyramid_levels(raster_data, attr="data")
    list_of_input_axes: list[Any] = get_pyramid_levels(raster_data, attr="dims")
    assert len(set(list_of_input_axes)) == 1
    input_axes = list_of_input_axes[0]
    # saving only the transformations of the first scale
    d = dict(raster_data["scale0"])
    assert len(d) == 1
    xdata = d.values().__iter__().__next__()
    transformations = _get_transformations_xarray(xdata)
    if transformations is None:
        raise ValueError(f"{element_name} does not have any transformations and can therefore not be written.")
    chunks = get_pyramid_levels(raster_data, "chunks")

    parsed_axes = _get_valid_axes(axes=list(input_axes), fmt=raster_format)
    storage_options = [{"chunks": chunk} for chunk in chunks]
    ome_zarr_format = get_ome_zarr_format(raster_format)
    dask_delayed = write_multi_scale_ngff(
        pyramid=data,
        group=group,
        fmt=ome_zarr_format,
        axes=parsed_axes,
        coordinate_transformations=None,
        storage_options=storage_options,
        **metadata,
        compute=False,
    )
    # Compute all pyramid levels at once to allow Dask to optimize the computational graph.
    # Optimize_graph is set to False for now as this causes permission denied errors when during atomic writes
    # os.replace is called. These can also be alleviated by using 'single-threaded' scheduler.
    da.compute(*dask_delayed, optimize_graph=False)

    trans_group = group["labels"][element_name] if raster_type == "labels" else group
    overwrite_coordinate_transformations_raster(
        group=trans_group,
        transformations=transformations,
        axes=tuple(input_axes),
        raster_format=raster_format,
    )


def write_image(
    image: DataArray | DataTree,
    group: zarr.Group,
    name: str,
    element_format: RasterFormatType = CurrentRasterFormat(),
    storage_options: JSONDict | list[JSONDict] | None = None,
    **metadata: str | JSONDict | list[JSONDict],
) -> None:
    _write_raster(
        raster_type="image",
        raster_data=image,
        group=group,
        name=name,
        raster_format=element_format,
        storage_options=storage_options,
        **metadata,
    )


def write_labels(
    labels: DataArray | DataTree,
    group: zarr.Group,
    name: str,
    element_format: RasterFormatType = CurrentRasterFormat(),
    storage_options: JSONDict | list[JSONDict] | None = None,
    label_metadata: JSONDict | None = None,
    **metadata: JSONDict,
) -> None:
    _write_raster(
        raster_type="labels",
        raster_data=labels,
        group=group,
        name=name,
        raster_format=element_format,
        storage_options=storage_options,
        label_metadata=label_metadata,
        **metadata,
    )
