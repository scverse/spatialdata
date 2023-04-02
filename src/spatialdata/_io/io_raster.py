import os
from pathlib import Path
from typing import Any, Literal, Optional, Union

import numpy as np
import zarr
from multiscale_spatial_image import MultiscaleSpatialImage
from ome_zarr.format import Format
from ome_zarr.io import ZarrLocation
from ome_zarr.reader import Label, Multiscales, Node, Reader
from ome_zarr.types import JSONDict
from ome_zarr.writer import _get_valid_axes
from ome_zarr.writer import write_image as write_image_ngff
from ome_zarr.writer import write_labels as write_labels_ngff
from ome_zarr.writer import write_multiscale as write_multiscale_ngff
from ome_zarr.writer import write_multiscale_labels as write_multiscale_labels_ngff
from spatial_image import SpatialImage
from xarray import DataArray

from spatialdata._io import SpatialDataFormatV01
from spatialdata._io._utils import (
    _get_transformations_from_ngff_dict,
    _iter_multiscale,
    overwrite_coordinate_transformations_raster,
)
from spatialdata._io.format import CurrentRasterFormat
from spatialdata.transformations._utils import (
    _get_transformations,
    _get_transformations_xarray,
    _set_transformations,
    compute_coordinates,
)


def _read_multiscale(
    store: Union[str, Path], raster_type: Literal["image", "labels"], fmt: SpatialDataFormatV01 = CurrentRasterFormat()
) -> Union[SpatialImage, MultiscaleSpatialImage]:
    assert isinstance(store, (str, Path))
    assert raster_type in ["image", "labels"]
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
    channels_metadata = node.load(Multiscales).zarr.root_attrs.get("channels_metadata", None)
    assert len(multiscales) == 1
    # checking for multiscales[0]["coordinateTransformations"] would make fail
    # something that doesn't have coordinateTransformations in top level
    # which is true for the current version of the spec
    # and for instance in the xenium example
    encoded_ngff_transformations = multiscales[0]["coordinateTransformations"]
    transformations = _get_transformations_from_ngff_dict(encoded_ngff_transformations)
    name = os.path.basename(node.metadata["name"])
    # if image, read channels metadata
    channels: Optional[list[Any]] = None
    if raster_type == "image" and channels_metadata is not None:
        channels = fmt.channels_from_metadata(channels_metadata)
    axes = [i["name"] for i in node.metadata["axes"]]
    if len(datasets) > 1:
        multiscale_image = {}
        for i, d in enumerate(datasets):
            data = node.load(Multiscales).array(resolution=d, version=fmt.version)
            multiscale_image[f"scale{i}"] = DataArray(
                data,
                name=name,
                dims=axes,
                coords={"c": channels} if channels is not None else {},
            )
        msi = MultiscaleSpatialImage.from_dict(multiscale_image)
        _set_transformations(msi, transformations)
        return compute_coordinates(msi)
    data = node.load(Multiscales).array(resolution=datasets[0], version=fmt.version)
    si = SpatialImage(
        data,
        name=name,
        dims=axes,
        coords={"c": channels} if channels is not None else {},
    )
    _set_transformations(si, transformations)
    return compute_coordinates(si)


def _write_raster(
    raster_type: Literal["image", "labels"],
    raster_data: Union[SpatialImage, MultiscaleSpatialImage],
    group: zarr.Group,
    name: str,
    fmt: Format = CurrentRasterFormat(),
    storage_options: Optional[Union[JSONDict, list[JSONDict]]] = None,
    label_metadata: Optional[JSONDict] = None,
    channels_metadata: Optional[JSONDict] = None,
    **metadata: Union[str, JSONDict, list[JSONDict]],
) -> None:
    assert raster_type in ["image", "labels"]
    # the argument "name" and "label_metadata" are only used for labels (to be precise, name is used in
    # write_multiscale_ngff() when writing metadata, but name is not used in write_image_ngff(). Maybe this is bug of
    # ome-zarr-py. In any case, we don't need that metadata and we use the argument name so that when we write labels
    # the correct group is created by the ome-zarr-py APIs. For images we do it manually in the function
    # _get_group_for_writing_data()
    if raster_type == "image":
        assert label_metadata is None
    else:
        metadata["name"] = name
        metadata["label_metadata"] = label_metadata

    write_single_scale_ngff = write_image_ngff if raster_type == "image" else write_labels_ngff
    write_multi_scale_ngff = write_multiscale_ngff if raster_type == "image" else write_multiscale_labels_ngff

    group_data = group.require_group(name) if raster_type == "image" else group

    def _get_group_for_writing_transformations() -> zarr.Group:
        if raster_type == "image":
            return group.require_group(name)
        return group["labels"][name]

    # convert channel names to channel metadata
    if raster_type == "image":
        group_data.attrs["channels_metadata"] = fmt.channels_to_metadata(raster_data, channels_metadata)

    if isinstance(raster_data, SpatialImage):
        data = raster_data.data
        transformations = _get_transformations(raster_data)
        input_axes: tuple[str, ...] = tuple(raster_data.dims)
        chunks = raster_data.chunks
        parsed_axes = _get_valid_axes(axes=list(input_axes), fmt=fmt)
        if storage_options is not None:
            if "chunks" not in storage_options and isinstance(storage_options, dict):
                storage_options["chunks"] = chunks
        else:
            storage_options = {"chunks": chunks}
        # Scaler needs to be None since we are passing the data already downscaled for the multiscale case.
        # We need this because the argument of write_image_ngff is called image while the argument of
        # write_labels_ngff is called label.
        metadata[raster_type] = data
        write_single_scale_ngff(
            group=group_data,
            scaler=None,
            fmt=fmt,
            axes=parsed_axes,
            coordinate_transformations=None,
            storage_options=storage_options,
            **metadata,
        )
        assert transformations is not None
        overwrite_coordinate_transformations_raster(
            group=_get_group_for_writing_transformations(), transformations=transformations, axes=input_axes
        )
    elif isinstance(raster_data, MultiscaleSpatialImage):
        data = _iter_multiscale(raster_data, "data")
        list_of_input_axes: list[Any] = _iter_multiscale(raster_data, "dims")
        assert len(set(list_of_input_axes)) == 1
        input_axes = list_of_input_axes[0]
        # saving only the transformations of the first scale
        d = dict(raster_data["scale0"])
        assert len(d) == 1
        xdata = d.values().__iter__().__next__()
        transformations = _get_transformations_xarray(xdata)
        assert transformations is not None
        assert len(transformations) > 0
        chunks = _iter_multiscale(raster_data, "chunks")
        # coords = _iter_multiscale(raster_data, "coords")
        parsed_axes = _get_valid_axes(axes=list(input_axes), fmt=fmt)
        storage_options = [{"chunks": chunk} for chunk in chunks]
        write_multi_scale_ngff(
            pyramid=data,
            group=group_data,
            fmt=fmt,
            axes=parsed_axes,
            coordinate_transformations=None,
            storage_options=storage_options,
            **metadata,
        )
        assert transformations is not None
        overwrite_coordinate_transformations_raster(
            group=_get_group_for_writing_transformations(), transformations=transformations, axes=tuple(input_axes)
        )
    else:
        raise ValueError("Not a valid labels object")


def write_image(
    image: Union[SpatialImage, MultiscaleSpatialImage],
    group: zarr.Group,
    name: str,
    fmt: Format = CurrentRasterFormat(),
    storage_options: Optional[Union[JSONDict, list[JSONDict]]] = None,
    **metadata: Union[str, JSONDict, list[JSONDict]],
) -> None:
    _write_raster(
        raster_type="image",
        raster_data=image,
        group=group,
        name=name,
        fmt=fmt,
        storage_options=storage_options,
        **metadata,
    )


def write_labels(
    labels: Union[SpatialImage, MultiscaleSpatialImage],
    group: zarr.Group,
    name: str,
    fmt: Format = CurrentRasterFormat(),
    storage_options: Optional[Union[JSONDict, list[JSONDict]]] = None,
    label_metadata: Optional[JSONDict] = None,
    **metadata: JSONDict,
) -> None:
    _write_raster(
        raster_type="labels",
        raster_data=labels,
        group=group,
        name=name,
        fmt=fmt,
        storage_options=storage_options,
        label_metadata=label_metadata,
        **metadata,
    )
