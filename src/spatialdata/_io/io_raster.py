from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, TypeGuard, cast

import dask.array as da
import numpy as np
import zarr
from ome_zarr.format import Format
from ome_zarr.io import parse_url
from ome_zarr.reader import Multiscales, Node, Reader
from ome_zarr.types import JSONDict
from ome_zarr.writer import _get_valid_axes
from ome_zarr.writer import write_image as write_image_ngff
from ome_zarr.writer import write_labels as write_labels_ngff
from ome_zarr.writer import write_multiscale as write_multiscale_ngff
from ome_zarr.writer import write_multiscale_labels as write_multiscale_labels_ngff
from xarray import DataArray, DataTree

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
from spatialdata.models.pyramids_utils import dask_arrays_to_datatree
from spatialdata.transformations._utils import (
    _get_transformations,
    _get_transformations_xarray,
    _set_transformations,
    compute_coordinates,
)


def _is_flat_int_sequence(value: object) -> TypeGuard[Sequence[int]]:
    # e.g. "", "auto" or b"auto"
    if isinstance(value, str | bytes):
        return False
    if not isinstance(value, Sequence):
        return False
    return all(isinstance(v, int) for v in value)


def _is_dask_chunk_grid(value: object) -> TypeGuard[Sequence[Sequence[int]]]:
    if isinstance(value, str | bytes):
        return False
    if not isinstance(value, Sequence):
        return False
    return len(value) > 0 and all(_is_flat_int_sequence(axis_chunks) for axis_chunks in value)


def _is_regular_dask_chunk_grid(chunk_grid: Sequence[Sequence[int]]) -> bool:
    """Check whether a Dask chunk grid is regular (zarr-compatible).

    A grid is regular when every axis has at most one unique chunk size among all but the last
    chunk, and the last chunk is not larger than the first.

    Parameters
    ----------
    chunk_grid
        Per-axis tuple of chunk sizes, for instance as returned by ``dask_array.chunks``.

    Examples
    --------
    Triggers ``continue`` on the first ``if`` (single or empty axis):

    >>> _is_regular_dask_chunk_grid([(4,)])   # single chunk → True
    True
    >>> _is_regular_dask_chunk_grid([()])     # empty axis → True
    True

    Triggers the first ``return False`` (non-uniform interior chunks):

    >>> _is_regular_dask_chunk_grid([(4, 4, 3, 4)])   # interior sizes differ → False
    False

    Triggers the second ``return False`` (last chunk larger than the first):

    >>> _is_regular_dask_chunk_grid([(4, 4, 4, 5)])   # last > first → False
    False

    Exits with ``return True``:

    >>> _is_regular_dask_chunk_grid([(4, 4, 4, 4)])   # all equal → True
    True
    >>> _is_regular_dask_chunk_grid([(4, 4, 4, 1)])   # last < first → True
    True

    Empty grid (loop never executes) → True:

    >>> _is_regular_dask_chunk_grid([])
    True

    Multi-axis: all axes regular → True; one axis irregular → False:

    >>> _is_regular_dask_chunk_grid([(4, 4, 4, 1), (3, 3, 2)])
    True
    >>> _is_regular_dask_chunk_grid([(4, 4, 4, 1), (4, 4, 3, 4)])
    False
    """
    # Match Dask's private _check_regular_chunks() logic without depending on its internal API.
    for axis_chunks in chunk_grid:
        if len(axis_chunks) <= 1:
            continue
        if len(set(axis_chunks[:-1])) > 1:
            return False
        if axis_chunks[-1] > axis_chunks[0]:
            return False
    return True


def _chunks_to_zarr_chunks(chunks: object) -> tuple[int, ...] | int | None:
    if isinstance(chunks, int):
        return chunks
    if _is_flat_int_sequence(chunks):
        return tuple(chunks)
    if _is_dask_chunk_grid(chunks):
        chunk_grid = tuple(tuple(axis_chunks) for axis_chunks in chunks)
        if _is_regular_dask_chunk_grid(chunk_grid):
            return tuple(axis_chunks[0] for axis_chunks in chunk_grid)
        return None
    return None


def _normalize_explicit_chunks(chunks: object) -> tuple[int, ...] | int:
    normalized = _chunks_to_zarr_chunks(chunks)
    if normalized is None:
        raise ValueError(
            'storage_options["chunks"] must resolve to a Zarr chunk shape or a regular Dask chunk grid. '
            "The current raster has irregular Dask chunks, which cannot be written to Zarr. "
            "To fix this, rechunk before writing, for example by passing regular chunks=... "
            "to Image2DModel.parse(...) / Labels2DModel.parse(...)."
        )
    return normalized


def _prepare_storage_options(
    storage_options: JSONDict | list[JSONDict] | None,
) -> JSONDict | list[JSONDict] | None:
    if storage_options is None:
        return None
    if isinstance(storage_options, dict):
        prepared = dict(storage_options)
        if "chunks" in prepared:
            prepared["chunks"] = _normalize_explicit_chunks(prepared["chunks"])
        return prepared

    prepared_options = [dict(options) for options in storage_options]
    for options in prepared_options:
        if "chunks" in options:
            options["chunks"] = _normalize_explicit_chunks(options["chunks"])
    return prepared_options


def _read_multiscale(
    store: str | Path | zarr.storage.ZipStore, raster_type: Literal["image", "labels"], reader_format: Format
) -> DataArray | DataTree:
    assert isinstance(store, str | Path | zarr.storage.ZipStore | zarr.Group)
    assert raster_type in ["image", "labels"]
    nodes: list[Node] = []
    # instantiate an internal subpath for zipstores
    internal_subpath = ""

    image_loc = parse_url(store, fmt=reader_format)

    if internal_subpath:
        image_loc.internal_subpath = internal_subpath
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
        arrays = [node.load(Multiscales).array(resolution=d) for d in datasets]
        msi = dask_arrays_to_datatree(arrays, dims=axes, channels=channels)
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
    raster_compressor: dict[Literal["lz4", "zstd"], int] | None = None,
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
    raster_compressor
        Compression settings as a len-1 dictionary with a single key-value {compression: compression level} pair
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
            raster_compressor=raster_compressor,
            **metadata,
        )
    elif isinstance(raster_data, DataTree):
        group = _write_raster_datatree(
            raster_type,
            group,
            name,
            raster_data,
            raster_format,
            storage_options,
            raster_compressor=raster_compressor,
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


def _build_v3_codec(
    compression: Literal["lz4", "zstd"],
    compression_level: int,
) -> Any:
    """Return the appropriate zarr v3 codec for the given compression type and level."""
    if compression == "zstd":
        from zarr.codecs import ZstdCodec

        return ZstdCodec(level=compression_level)
    # lz4: use the native zarr v3 BloscCodec
    from zarr.codecs import BloscCodec

    return BloscCodec(cname="lz4", clevel=compression_level)


def _apply_compression(
    storage_options: JSONDict | list[JSONDict],
    raster_compressor: dict[Literal["lz4", "zstd"], int] | None,
    zarr_format: Literal[2, 3] = 3,
) -> JSONDict | list[JSONDict]:
    """Apply compression settings to storage options.

    Parameters
    ----------
    storage_options
        Storage options for zarr arrays
    raster_compressor
        Compression settings as a dictionary with a single key-value pair
    zarr_format
        The zarr format version (2 or 3)

    Returns
    -------
    Updated storage options with compression settings
    """
    if not raster_compressor:
        return storage_options

    ((compression, compression_level),) = raster_compressor.items()

    if zarr_format == 2:
        from numcodecs import Blosc as BloscV2

        codec_v2 = BloscV2(cname=compression, clevel=compression_level, shuffle=1)

        def _update_dict(d: dict[str, Any]) -> None:
            d["compressor"] = codec_v2

        if isinstance(storage_options, dict):
            _update_dict(d=storage_options)
        elif isinstance(storage_options, list):
            for option in storage_options:
                _update_dict(d=option)
        elif storage_options is None:
            return {"compressor": codec_v2}
        else:
            raise ValueError(f"storage_options must be a dict or list, not {type(storage_options)}")
    else:
        # zarr v3: use native codec objects via the "compressors" (plural) key.
        # see  https://github.com/ome/ome-zarr-py/blob/v0.16.0/ome_zarr/writer.py#L754
        # ome-zarr-py ≥ 0.16.0 with dask ≥ 2026.3.0 forwards this key to zarr_array_kwargs.
        codec_v3 = _build_v3_codec(compression, compression_level)

        def _update_dict_v3(d: dict[str, Any]) -> None:
            d["compressors"] = [codec_v3]

        if isinstance(storage_options, dict):
            _update_dict_v3(d=storage_options)
        elif isinstance(storage_options, list):
            for option in storage_options:
                _update_dict_v3(d=option)
        elif storage_options is None:
            return {"compressors": [codec_v3]}
        else:
            raise ValueError(f"storage_options must be a dict or list, not {type(storage_options)}")

    return storage_options


def _write_raster_dataarray(
    raster_type: Literal["image", "labels"],
    group: zarr.Group,
    element_name: str,
    raster_data: DataArray,
    raster_format: RasterFormatType,
    storage_options: JSONDict | list[JSONDict] | None,
    raster_compressor: dict[Literal["lz4", "zstd"], int] | None,
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
    raster_compressor
        Compression settings as a len-1 dictionary with a single key-value {compression: compression level} pair
    metadata
        Additional metadata for the raster element
    """
    write_single_scale_ngff = write_image_ngff if raster_type == "image" else write_labels_ngff

    data = raster_data.data
    transformations = _get_transformations(raster_data)
    assert transformations is not None  # mypy: validate_element() in _write_element guarantees this
    input_axes: tuple[str, ...] = tuple(raster_data.dims)
    parsed_axes = _get_valid_axes(axes=list(input_axes), fmt=raster_format)
    storage_options = _prepare_storage_options(storage_options)
    # Apply compression if specified
    storage_options = _apply_compression(
        storage_options, raster_compressor, zarr_format=cast(Literal[2, 3], raster_format.zarr_format)
    )

    # Explicitly disable pyramid generation for single-scale rasters. Recent ome-zarr versions default
    # write_image()/write_labels() to scale_factors=(2, 4, 8, 16), which would otherwise write s0, s1, ...
    # even when the input is a plain DataArray.
    # We need this because the argument of write_image_ngff is called image while the argument of
    # write_labels_ngff is called label.
    metadata[raster_type] = data
    ome_zarr_format = get_ome_zarr_format(raster_format)
    write_single_scale_ngff(
        group=group,
        scale_factors=[],
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
    storage_options: JSONDict | list[JSONDict] | None,
    raster_compressor: dict[Literal["lz4", "zstd"], int] | None,
    **metadata: str | JSONDict | list[JSONDict],
) -> zarr.Group:
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
    raster_compressor
        Compression settings as a len-1 dictionary with a single key-value {compression: compression level} pair
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
    assert transformations is not None  # mypy: validate_element() in _write_element guarantees this

    parsed_axes = _get_valid_axes(axes=list(input_axes), fmt=raster_format)
    storage_options = _prepare_storage_options(storage_options)

    # Apply compression if specified
    storage_options = _apply_compression(storage_options, raster_compressor, zarr_format=raster_format.zarr_format)

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

    # Workaround for https://github.com/scverse/spatialdata/issues/1024.
    # ome-zarr-py bundles write_multiscales_metadata() as a dask.delayed task in the compute=False
    # code path (see https://github.com/ome/ome-zarr-py/issues/580). When da.compute() runs with
    # the 'processes' scheduler that task executes in a subprocess: the on-disk zarr.json is written
    # correctly, but the zarr.Group held in this process keeps its original in-memory GroupMetadata
    # and never sees the update. Re-opening the group forces a fresh read from the store.
    # This workaround should not be needed once https://github.com/ome/ome-zarr-py/issues/580 is fixed.
    group = zarr.open_group(store=group.store, path=group.path, mode="r+", use_consolidated=False)

    trans_group = group["labels"][element_name] if raster_type == "labels" else group
    overwrite_coordinate_transformations_raster(
        group=trans_group,
        transformations=transformations,
        axes=tuple(input_axes),
        raster_format=raster_format,
    )
    return group


def write_image(
    image: DataArray | DataTree,
    group: zarr.Group,
    name: str,
    element_format: RasterFormatType = CurrentRasterFormat(),
    storage_options: JSONDict | list[JSONDict] | None = None,
    raster_compressor: dict[Literal["lz4", "zstd"], int] | None = None,
    **metadata: str | JSONDict | list[JSONDict],
) -> None:
    _write_raster(
        raster_type="image",
        raster_data=image,
        group=group,
        name=name,
        raster_format=element_format,
        storage_options=storage_options,
        raster_compressor=raster_compressor,
        **metadata,
    )


def write_labels(
    labels: DataArray | DataTree,
    group: zarr.Group,
    name: str,
    element_format: RasterFormatType = CurrentRasterFormat(),
    storage_options: JSONDict | list[JSONDict] | None = None,
    label_metadata: JSONDict | None = None,
    raster_compressor: dict[Literal["lz4", "zstd"], int] | None = None,
    **metadata: JSONDict,
) -> None:
    _write_raster(
        raster_type="labels",
        raster_data=labels,
        group=group,
        name=name,
        raster_format=element_format,
        storage_options=storage_options,
        raster_compressor=raster_compressor,
        label_metadata=label_metadata,
        **metadata,
    )
