from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import dask.array as da
import numpy as np
import zarr
from ome_zarr.format import Format
from ome_zarr.io import ZarrLocation
from ome_zarr.reader import Label, Multiscales, Node, Reader
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
    RasterFormats,
    RasterFormatV01,
    _parse_version,
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


def _read_multiscale(store: str | Path, raster_type: Literal["image", "labels"]) -> DataArray | DataTree:
    assert isinstance(store, str | Path)
    assert raster_type in ["image", "labels"]

    f = zarr.open(store, mode="r")
    version = _parse_version(f, expect_attrs_key=True)
    # old spatialdata datasets don't have format metadata for raster elements; this line ensure backwards compatibility,
    # interpreting the lack of such information as the presence of the format v01
    format = RasterFormatV01() if version is None else RasterFormats[version]
    f.store.close()

    nodes: list[Node] = []
    image_loc = ZarrLocation(store)
    if image_loc.exists():
        image_reader = Reader(image_loc)()
        image_nodes = list(image_reader)
        if len(image_nodes):
            for node in image_nodes:
                if np.any([isinstance(spec, Multiscales) for spec in node.specs]) and (
                    raster_type == "image"
                    and np.all([not isinstance(spec, Label) for spec in node.specs])
                    or raster_type == "labels"
                    and np.any([isinstance(spec, Label) for spec in node.specs])
                ):
                    nodes.append(node)
    if len(nodes) != 1:
        raise ValueError(
            f"len(nodes) = {len(nodes)}, expected 1. Unable to read the NGFF file. Please report this "
            f"bug and attach a minimal data example."
        )
    node = nodes[0]
    datasets = node.load(Multiscales).datasets
    multiscales = node.load(Multiscales).zarr.root_attrs["multiscales"]
    omero_metadata = node.load(Multiscales).zarr.root_attrs.get("omero", None)
    legacy_channels_metadata = node.load(Multiscales).zarr.root_attrs.get("channels_metadata", None)  # legacy v0.1
    assert len(multiscales) == 1
    # checking for multiscales[0]["coordinateTransformations"] would make fail
    # something that doesn't have coordinateTransformations in top level
    # which is true for the current version of the spec
    # and for instance in the xenium example
    encoded_ngff_transformations = multiscales[0]["coordinateTransformations"]
    transformations = _get_transformations_from_ngff_dict(encoded_ngff_transformations)
    # TODO: what to do with name? For now remove?
    # name = os.path.basename(node.metadata["name"])
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
            data = node.load(Multiscales).array(resolution=d, version=format.version)
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
    data = node.load(Multiscales).array(resolution=datasets[0], version=format.version)
    si = DataArray(
        data,
        name="image",
        dims=axes,
        coords={"c": channels} if channels is not None else {},
    )
    _set_transformations(si, transformations)
    return compute_coordinates(si)


def _get_group_for_writing_transformations(
    raster_type: Literal["image", "labels"], group: zarr.Group, name: str
) -> zarr.Group:
    """Get the appropriate zarr group for writing transformations.

    Parameters
    ----------
    raster_type
        Type of raster data, either "image" or "labels"
    group
        Parent zarr group
    name
        Name of the element

    Returns
    -------
    The zarr group where transformations should be written
    """
    if raster_type == "image":
        return group.require_group(name)
    return group["labels"][name]


def _apply_compression(
    storage_options: JSONDict | list[JSONDict], compressor: dict[Literal["lz4", "zstd"], int] | None
) -> JSONDict | list[JSONDict]:
    """Apply compression settings to storage options.

    Parameters
    ----------
    storage_options
        Storage options for zarr arrays
    compressor
        Compression settings as a dictionary with a single key-value pair

    Returns
    -------
    Updated storage options with compression settings
    """
    from zarr.codecs import Blosc

    if not compressor:
        return storage_options

    ((compression, compression_level),) = compressor.items()

    if isinstance(storage_options, dict):
        storage_options["compressor"] = Blosc(cname=compression, clevel=compression_level, shuffle=1)
    elif isinstance(storage_options, list):
        for option in storage_options:
            option["compressor"] = Blosc(cname=compression, clevel=compression_level, shuffle=1)

    return storage_options


def _write_data_array(
    raster_type: Literal["image", "labels"],
    raster_data: DataArray,
    group_data: zarr.Group,
    format: Format,
    storage_options: JSONDict | None,
    compressor: dict[Literal["lz4", "zstd"], int] | None,
    metadata: dict[str, Any],
    get_transformations_group: Callable[[], zarr.Group],
) -> None:
    """Write a DataArray to a zarr group.

    Parameters
    ----------
    raster_type
        Type of raster data, either "image" or "labels"
    raster_data
        The DataArray to write
    group_data
        The zarr group to write to
    format
        The spatialdata raster format to use for writing
    storage_options
        Storage options for zarr arrays (to be passed to ome-zarr)
    compressor
        Compression settings as a dictionary with a single key-value (compression, compression level) pair
    metadata
        Additional metadata
    get_transformations_group
        Function that returns the group for writing transformations
    """
    data = raster_data.data
    transformations = _get_transformations(raster_data)
    input_axes: tuple[str, ...] = tuple(raster_data.dims)
    chunks = raster_data.chunks
    parsed_axes = _get_valid_axes(axes=list(input_axes), fmt=format)

    # Set up storage options with chunks
    if storage_options is not None:
        if "chunks" not in storage_options and isinstance(storage_options, dict):
            storage_options["chunks"] = chunks
    else:
        storage_options = {"chunks": chunks}

    # Apply compression if specified
    storage_options = _apply_compression(storage_options, compressor)

    # Determine which write function to use
    write_single_scale_ngff = write_image_ngff if raster_type == "image" else write_labels_ngff

    # Scaler needs to be None since we are passing the data already downscaled for the multiscale case.
    # We need this because the argument of write_image_ngff is called image while the argument of
    # write_labels_ngff is called label.
    metadata[raster_type] = data

    # Write the data
    write_single_scale_ngff(
        group=group_data,
        scaler=None,
        fmt=format,
        axes=parsed_axes,
        coordinate_transformations=None,
        storage_options=storage_options,
        **metadata,
    )

    # Write transformations
    assert transformations is not None
    overwrite_coordinate_transformations_raster(
        group=get_transformations_group(), transformations=transformations, axes=input_axes
    )


def _write_data_tree(
    raster_type: Literal["image", "labels"],
    raster_data: DataTree,
    group_data: zarr.Group,
    format: Format,
    storage_options: JSONDict | list[JSONDict] | None,
    compressor: dict[Literal["lz4", "zstd"], int] | None,
    metadata: dict[str, Any],
    get_transformations_group: Callable[[], zarr.Group],
) -> None:
    """Write a DataTree to a zarr group.

    Parameters
    ----------
    raster_type
        Type of raster data, either "image" or "labels"
    raster_data
        The DataTree to write
    group_data
        The zarr group to write to
    format
        The SpatialData raster format to use for writing
    storage_options
        Storage options for zarr arrays (to be passed to ome-zarr)
    compressor
        Compression settings as a dictionary with a single key-value (compression, compression level) pair
    metadata
        Additional metadata
    get_transformations_group
        Function that returns the group for writing transformations
    """
    data = get_pyramid_levels(raster_data, attr="data")
    list_of_input_axes: list[Any] = get_pyramid_levels(raster_data, attr="dims")
    assert len(set(list_of_input_axes)) == 1
    input_axes = list_of_input_axes[0]

    # Saving only the transformations of the first scale
    d = dict(raster_data["scale0"])
    assert len(d) == 1
    xdata = d.values().__iter__().__next__()
    transformations = _get_transformations_xarray(xdata)
    assert transformations is not None
    assert len(transformations) > 0

    chunks = get_pyramid_levels(raster_data, "chunks")
    parsed_axes = _get_valid_axes(axes=list(input_axes), fmt=format)

    # Set up storage options with chunks
    if storage_options is None:
        storage_options = [{"chunks": chunk} for chunk in chunks]

    # Apply compression if specified
    storage_options = _apply_compression(storage_options, compressor)

    # Determine which write function to use
    write_multi_scale_ngff = write_multiscale_ngff if raster_type == "image" else write_multiscale_labels_ngff

    # Write the data
    dask_delayed = write_multi_scale_ngff(
        pyramid=data,
        group=group_data,
        fmt=format,
        axes=parsed_axes,
        coordinate_transformations=None,
        storage_options=storage_options,
        **metadata,
        compute=False,
    )

    # Compute all pyramid levels at once to allow Dask to optimize the computational graph.
    da.compute(*dask_delayed)

    # Write transformations
    assert transformations is not None
    overwrite_coordinate_transformations_raster(
        group=get_transformations_group(), transformations=transformations, axes=tuple(input_axes)
    )


def _write_raster(
    raster_type: Literal["image", "labels"],
    raster_data: DataArray | DataTree,
    group: zarr.Group,
    name: str,
    format: Format = CurrentRasterFormat(),
    storage_options: JSONDict | list[JSONDict] | None = None,
    compressor: dict[Literal["lz4", "zstd"], int] | None = None,
    label_metadata: JSONDict | None = None,
    **metadata: str | JSONDict | list[JSONDict],
) -> None:
    """Write raster data to a zarr group.

    This function handles writing both image and label data, in both single-scale (DataArray)
    and multi-scale (DataTree) formats.

    Parameters
    ----------
    raster_type
        Type of raster data, either "image" or "labels"
    raster_data
        The data to write, either a DataArray (single-scale) or DataTree (multi-scale)
    group
        The zarr group to write to
    name
        Name of the element
    format
        The raster format to use for writing
    storage_options
        Storage options for zarr arrays (to be passed to ome-zarr)
    compressor
        Compression settings as a dictionary with a single key-value (compression, compression level) pair
    label_metadata
        Metadata specific to labels
    **metadata
        Additional metadata
    """
    # Validate inputs
    if raster_type not in ["image", "labels"]:
        raise TypeError(f"Writing raster data is only supported for 'image' and 'labels'. Got: {raster_type}")

    # The argument "name" and "label_metadata" are only used for labels (to be precise, name is used in
    # write_multiscale_ngff() when writing metadata, but name is not used in write_image_ngff(). Maybe this is bug of
    # ome-zarr-py. In any case, we don't need that metadata and we use the argument name so that when we write labels
    # the correct group is created by the ome-zarr-py APIs. For images we do it manually in the function
    # _get_group_for_writing_data()
    if raster_type == "image" and label_metadata is not None:
        raise ValueError("If the rastertype is 'image', 'label_metadata' should be None.")

    if raster_type == "labels":
        metadata["name"] = name
        metadata["label_metadata"] = label_metadata

    # Prepare the group for writing data
    group_data = group.require_group(name) if raster_type == "image" else group

    # Create a function to get the transformations group
    get_transformations_group = lambda: _get_group_for_writing_transformations(raster_type, group, name)

    # Convert channel names to channel metadata in omero for images
    if raster_type == "image":
        metadata["metadata"] = {"omero": {"channels": []}}
        channels = get_channel_names(raster_data)
        for c in channels:
            metadata["metadata"]["omero"]["channels"].append({"label": c})  # type: ignore[union-attr, index, call-overload]

    # Write the data based on its type
    if isinstance(raster_data, DataArray):
        _write_data_array(
            raster_type=raster_type,
            raster_data=raster_data,
            group_data=group_data,
            format=format,
            storage_options=storage_options,
            compressor=compressor,
            metadata=metadata,
            get_transformations_group=get_transformations_group,
        )
    elif isinstance(raster_data, DataTree):
        _write_data_tree(
            raster_type=raster_type,
            raster_data=raster_data,
            group_data=group_data,
            format=format,
            storage_options=storage_options,
            compressor=compressor,
            metadata=metadata,
            get_transformations_group=get_transformations_group,
        )
    else:
        raise ValueError("Not a valid labels object")

    # Write format version metadata
    # As explained in a comment in format.py, since coordinate transformations are not part of NGFF yet, we need to have
    # our spatialdata extension also for raster type (eventually it will be dropped in favor of pure NGFF). Until then,
    # saving the NGFF version (i.e. 0.4) is not enough, and we need to also record which version of the spatialdata
    # format we are using for raster types
    group = get_transformations_group()
    if ATTRS_KEY not in group.attrs:
        group.attrs[ATTRS_KEY] = {}
    attrs = group.attrs[ATTRS_KEY]
    attrs["version"] = format.spatialdata_format_version
    # Triggers the write operation
    group.attrs[ATTRS_KEY] = attrs


def write_image(
    image: DataArray | DataTree,
    group: zarr.Group,
    name: str,
    format: Format = CurrentRasterFormat(),
    storage_options: JSONDict | list[JSONDict] | None = None,
    compressor: dict[Literal["lz4", "zstd"], int] | None = None,
    **metadata: str | JSONDict | list[JSONDict],
) -> None:
    _write_raster(
        raster_type="image",
        raster_data=image,
        group=group,
        name=name,
        format=format,
        storage_options=storage_options,
        compressor=compressor,
        **metadata,
    )


def write_labels(
    labels: DataArray | DataTree,
    group: zarr.Group,
    name: str,
    format: Format = CurrentRasterFormat(),
    storage_options: JSONDict | list[JSONDict] | None = None,
    label_metadata: JSONDict | None = None,
    compressor: dict[Literal["lz4", "zstd"], int] | None = None,
    **metadata: JSONDict,
) -> None:
    _write_raster(
        raster_type="labels",
        raster_data=labels,
        group=group,
        name=name,
        format=format,
        storage_options=storage_options,
        compressor=compressor,
        label_metadata=label_metadata,
        **metadata,
    )
