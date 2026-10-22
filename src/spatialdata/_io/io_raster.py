# pyright: standard

from __future__ import annotations

from dataclasses import dataclass
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Final, Iterator, Literal, TypeGuard, assert_type, get_args

import itertools
import dask.array as da
import numpy as np
import ome_zarr as oz
import ome_zarr_models.v06.coordinate_transforms as ozm06ct
import xarray as xr
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
from xarray import DataArray, DataTree
from xarray.indexes import RangeIndex

from spatialdata._io._utils import (
    _get_transformations_from_ngff_dict,
    overwrite_channel_names,
    overwrite_coordinate_transformations_raster,
)
from spatialdata._io.exceptions import WritingToZarrV2DeprecationWarning
from spatialdata._io.format import (
    CurrentRasterFormat,
    RasterFormatType,
    get_ome_zarr_format,
)
from spatialdata._types import ELEMENT_TYPE, ELEMENT_TYPE_RASTER
from spatialdata._utils import get_pyramid_levels
from spatialdata.models.models import ATTRS_KEY
from spatialdata.models.pyramids_utils import dask_arrays_to_datatree
from spatialdata.transformations._utils import (
    _get_transformations,
    _get_transformations_xarray,
    _set_transformations,
    compute_coordinates,
)
from spatialdata.transformations.graph.edge import BaseTransformationEdge, IdentityEdge, ScaleEdge, TranslationEdge, parse_ngff_transf
from spatialdata.transformations.graph.vert import Axis, CoordSystem, CsVert, Edge, Edges, NgffCsPath, Provenance, Spaces


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

    >>> _is_regular_dask_chunk_grid([(4,)])  # single chunk → True
    True
    >>> _is_regular_dask_chunk_grid([()])  # empty axis → True
    True

    Triggers the first ``return False`` (non-uniform interior chunks):

    >>> _is_regular_dask_chunk_grid([(4, 4, 3, 4)])  # interior sizes differ → False
    False

    Triggers the second ``return False`` (last chunk larger than the first):

    >>> _is_regular_dask_chunk_grid([(4, 4, 4, 5)])  # last > first → False
    False

    Exits with ``return True``:

    >>> _is_regular_dask_chunk_grid([(4, 4, 4, 4)])  # all equal → True
    True
    >>> _is_regular_dask_chunk_grid([(4, 4, 4, 1)])  # last < first → True
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


@dataclass(frozen=True, eq=True)
class ElementKey:
    namespace: Final[PurePosixPath]
    name: Final[str]

    def __post_init__(self):
        assert not self.namespace.is_absolute()

    @classmethod
    def from_path(cls, path: PurePosixPath) -> ElementKey:
        return ElementKey(namespace=path.parent, name=path.name)

    def to_path(self) -> PurePosixPath:
        return self.namespace / self.name

    def is_sibling_of(self, other: ElementKey) -> bool:
        return self.namespace == other.namespace

    def is_relative_to(self, other: ElementKey) -> bool:
        return self.namespace.is_relative_to(other.namespace)


@dataclass(frozen=True, eq=True)
class MultiscalesKeys:
    raster: ElementKey
    labels: Mapping[PurePosixPath, ElementKey]


class FutureSpatialData:
    _graph: Spaces  # FIXME: use TransformationManager
    _images: dict[ElementKey, xr.DataTree]
    _labels: dict[ElementKey, xr.DataTree]
    _intrinsics: dict[ElementKey, CsVert]

    def __init__(self, graph: Spaces | None = None) -> None:
        self._graph = graph or Spaces()
        self._images = {}
        self._labels = {}
        self._intrinsics = {}
        super().__init__()

    def incorporate(self, other: FutureSpatialData):
        # FIXME: there might be more conflicts that need to be checked before merging
        # so we avoid a half-merged SpatialData
        for key in other._images.keys():
            if key in self._images:
                raise KeyError(f"Can't merge key {key}") # FIXME: custm error

        for key in other._labels.keys():
            if key in self._images:
                raise KeyError(f"Can't merge key {key}") # FIXME: custm error

        self._graph.incorporate(other._graph)
        self._images.update(other._images)
        self._labels.update(other._labels)
        self._intrinsics.update(other._intrinsics)

    def incorporate_graph(self, graph: Spaces):
        self._graph.incorporate(graph)

    def incorporate_ngff_graph(
        self,
        group_path: PurePosixPath,
        coord_systems: Sequence[ozm06ct.CoordinateSystem],
        transforms: Sequence[ozm06ct.AnyTransform],
    ) -> None:
        self._graph.incorporate_ngff_graph(group_path, coord_systems, transforms)

    def add_image(self, key: ElementKey, image: xr.DataTree, into: CsVert):
        if into not in self._graph.verts():
            raise KeyError(f"{CsVert} is not in the transformation graph")
        if key in self._labels:
            raise KeyError(f"Already a label")
        self._images[key] = image
        self._intrinsics[key] = into

    def get_image(self, key: ElementKey) -> xr.DataTree | None:
        return self._images[key]

    def add_label(self, key: ElementKey, label: xr.DataTree, into: CsVert):
        if into not in self._graph.verts():
            raise KeyError(f"{CsVert} is not in the transformation graph")
        if key in self._images:
            raise KeyError(f"Already an image")
        self._labels[key] = label
        self._intrinsics[key] = into

    def get_label(self, key: ElementKey) -> xr.DataTree | None:
        return self._labels[key]

    def get_intrinsic(self, key: ElementKey) -> CsVert:
        return  self._intrinsics[key]

    def add_cs(self, cs: CoordSystem, provenance: Provenance):
        self._graph.add_cs(cs=cs, provenance=provenance)

    def find_node(self, *, ngff_path: NgffCsPath) -> CsVert:
        return self._graph.find_node(ngff_path=ngff_path)

    def add_transformation(self, *, input: CsVert, output: CsVert, transf: BaseTransformationEdge, name: str | None):
        self._graph.add_transformation(input=input, output=output, transf=transf, name=name)

    def edges(self) -> Edges:
        return self._graph.edges()

    def ngff_label_dumper(self, *, label_key: ElementKey) -> MultiscalesDumper:
        return self._ngff_raster_dumper(MultiscalesKeys(raster=label_key, labels={}))

    def ngff_multiscales_dumper(self, keys: MultiscalesKeys) -> MultiscalesDumper:
        return self._ngff_raster_dumper(keys)

    def _ngff_raster_dumper(self, keys: MultiscalesKeys) -> MultiscalesDumper:
        assert keys.raster in self._images or (keys.raster in self._labels and len(keys.labels) == 0)

        intrinsic = self.get_intrinsic(key=keys.raster)
        multiscales_transforms: set[Edge] = self.edges().containing(cs=intrinsic).ngff_siblings().collect(set)

        label_dumpers: dict[PurePosixPath, MultiscalesDumper] = {}
        for label_rel_path, label_key in keys.labels.items():
            label_intrinsic = self.get_intrinsic(label_key)
            ngff_transforms_to_label = (
                self.edges()
                .directed(fro=intrinsic, to=label_intrinsic) #mutlsicale-local transfs can point down to labels
                .transf_is((IdentityEdge, ScaleEdge, TranslationEdge))
                .collect(set)
            )
            multiscales_transforms |= ngff_transforms_to_label
            label_dumpers[label_rel_path] = self.ngff_label_dumper(label_key=label_key)

        return MultiscalesDumper(
            intrinsic=intrinsic,
            key=keys.raster,
            labels=label_dumpers,
            sdata=self,
            transform_edges=list(multiscales_transforms),
        )

    def candidate_ngff_labels(
        self,
        *,
        image_key: ElementKey,
        gather_by_ngff_provenance: bool,
        gather_by_unique_connection: bool,
    ) -> Mapping[PurePosixPath, ElementKey]:
        image_intrinsic = self.get_intrinsic(image_key)
        out: dict[PurePosixPath, ElementKey] = {}

        for label_key  in self._labels.keys():
            label_intrinsic = self.get_intrinsic(label_key)
            edges = (
                self.edges()
                .directed(fro=image_intrinsic, to=label_intrinsic)
                .transf_is((IdentityEdge, ScaleEdge, TranslationEdge))
                .collect(set)
            )

            proceed = False
            if not proceed:
                proceed = gather_by_unique_connection and len(edges) == 1
            if not proceed:
                proceed = gather_by_ngff_provenance and any(e.points_down_ngff_hierarchy() for e in edges)
            if not proceed:
                continue

            if label_key.is_relative_to(image_key):
                label_rel_path = label_key.namespace.relative_to(image_key.namespace)
            else:
                label_rel_path = PurePosixPath(label_key.name)
                suffix = 2
                while label_rel_path in out:
                    label_rel_path = PurePosixPath(f"{label_key.name}__{suffix}")
                    suffix += 1
            out[label_rel_path] = label_key

        return out

    def scene_dumper(
        self,
        children: Mapping[PurePosixPath, MultiscalesKeys],
    ) -> SceneDumper:
        scene_coord_systems: set[CsVert] = set(self._graph.verts())

        ms_dumpers = {
            ms_path: self.ngff_multiscales_dumper(mskey)
            for ms_path, mskey in children.items()
        }
        ms_coord_systems = set(itertools.chain.from_iterable(
            ms.all_coord_systems() for ms in ms_dumpers.values()
        ))
        scene_coord_systems: set[CsVert] = set()
        scene_edges: set[Edge] = set()
        for edge in self.edges():
            if edge.origin in ms_coord_systems and edge.target not in ms_coord_systems:
                scene_cs = edge.target
            elif edge.origin not in ms_coord_systems and edge.target in ms_coord_systems:
                scene_cs = edge.origin
            else:
                continue
            scene_coord_systems.add(scene_cs)
            scene_edges.add(edge)

        return SceneDumper(
            multiscales=ms_dumpers,
            coord_systems=tuple(scene_coord_systems),
            transform_edges=tuple(scene_edges),
        )

    def default_scene_dumper(self) -> SceneDumper:
        multiscale_keys: dict[PurePosixPath, MultiscalesKeys] = {}
        for img_key, img in self._images.items():
            candidate_labels = self.candidate_ngff_labels(
                image_key=img_key, gather_by_ngff_provenance=True, gather_by_unique_connection=False,
            )
            # FIXME: raising for duplicate names doens't seem appropriate
            img_path = img_key.namespace
            if img_path in multiscale_keys:
                raise ValueError(f"Multiple images with conflicting paths: {img_path}") # FIXME: custom exception
            multiscale_keys[img_path] = MultiscalesKeys(
                raster=img_key,
                labels=candidate_labels
            )
        return self.scene_dumper(multiscale_keys)

def try_parse_ngff06_scene(
    group_path: PurePosixPath,
    scene: oz.OMEZarrScene,
) -> FutureSpatialData:
    sdata = FutureSpatialData()

    for img_path, img in scene.images.items():
        ms_sdata = try_parse_ngff06_multiscale(
            group_path=group_path / img_path.lstrip("/"),
            multiscale=img,
        )
        sdata.incorporate(ms_sdata)

    sdata.incorporate_ngff_graph(
        group_path=group_path,
        coord_systems=scene.coordinate_systems,
        transforms=scene.coordinate_transformations,
    )

    return sdata

def try_read_ngff06_multiscale(
    store: Path | zarr.Group,
) -> FutureSpatialData:
    multiscale = oz.OMEZarrMultiscale.from_ome_zarr(str(store) if isinstance(store, Path) else store)
    assert isinstance(multiscale, oz.OMEZarrMultiscale)  # disambiguate from OMEZarrLabel
    # FIXME: we should probably use something liek a zarr-interpretable URL isntead of a path
    return try_parse_ngff06_multiscale(multiscale=multiscale, group_path=PurePosixPath("."))

def try_parse_ngff06_multiscale(
    group_path: PurePosixPath,
    multiscale: oz.OMEZarrMultiscale | oz.OMEZarrLabels,
) -> FutureSpatialData:
    sdata = FutureSpatialData()

    if isinstance(multiscale, oz.OMEZarrMultiscale):
        for label_path, label in (multiscale.labels or {}).items():
            # spec says labels are under the "labels/" directory
            label_namespace = group_path / "labels" / label_path  # FIXME: is label_path safe?
            label_sdata = try_parse_ngff06_multiscale(
                group_path=label_namespace,
                multiscale=label,
            )
            sdata.incorporate(label_sdata)

    # mutliscales transforms can reference CSs in `labels/` , so we process those last
    sdata.incorporate_ngff_graph(
        group_path=group_path,
        coord_systems = multiscale.metadata.coordinateSystems,
        transforms = multiscale.metadata.coordinateTransformations or (),
    )

    channel_names: Sequence[str] | None = None
    if isinstance(multiscale, oz.OMEZarrMultiscale):
        omero = multiscale.omero
        channel_names = None if omero is None else [d.color for d in omero.channels]
    elif isinstance(multiscale, oz.OMEZarrLabels):
        # FIXME: ome zarr has special annotations for labels
        pass
    else:
        assert_never(multiscale)

    intrinsic_cs_path = NgffCsPath(group_path=group_path, name=multiscale.metadata.intrinsic_coordinate_system.name)
    intrinsic = sdata.find_node(ngff_path=intrinsic_cs_path)
    intrinsic_cs = intrinsic.coord_system
    data_tree = xr.DataTree()
    for scale_idx, (ds_md, ds) in enumerate(zip(multiscale.metadata.datasets, multiscale.images, strict=True)):
        # This coord system doesn't exist explicitly in the NGFF file, nor will
        # it exist in our graph of transformations; It is only created here
        # for the sake of creating the transformations that will be expressed
        # in levels of a xr.DataTree
        pixel_cs = CoordSystem.create_dummy(axes=intrinsic_cs.num_axes)
        pixel_cs_to_intrinsic_ngff = ozm06ct.Sequence(transformations=ds_md.coordinateTransformations)
        seq = parse_ngff_transf(input=pixel_cs, output=intrinsic_cs, model=pixel_cs_to_intrinsic_ngff)
        ds_shape = np.asarray(ds.data.shape)
        transformed_start = seq.transform_points(np.zeros_like(ds_shape)[np.newaxis, :])[0]
        transformed_stop = seq.transform_points((ds_shape - 1)[np.newaxis, :])[0]

        coords: xr.Coordinates = xr.Coordinates()
        for low, high, ax, extent in zip(transformed_start, transformed_stop, intrinsic_cs.axes, ds_shape, strict=True):
            if ax.type == "channel" and channel_names is not None:
                coords = coords.merge({ax.name: channel_names}).coords
            else:
                axis_index = xr.Coordinates.from_xindex(
                    RangeIndex.linspace(
                        start=low,
                        stop=high,
                        num=extent,
                        endpoint=True,
                        dim=ax.name,
                    )
                )
                coords = coords.merge(axis_index).coords

        # Note: the magic "image" and "scale<N> " strings mimic the current
        # behavior from `dask_arrays_to_datatree`
        data_tree[f"scale{scale_idx}"] = xr.Dataset(
            {
                "image": xr.DataArray(
                    ds.data,
                    name="image",
                    dims=intrinsic_cs.axes_names,
                    coords=coords,
                )
            },
        )

    sdata_key = ElementKey(namespace=group_path, name=multiscale.name)
    if isinstance(multiscale, oz.OMEZarrLabels):
        sdata.add_label(key=sdata_key, label=data_tree, into=intrinsic)
    elif isinstance(multiscale, oz.OMEZarrMultiscale):
        sdata.add_image(key=sdata_key, image=data_tree, into=intrinsic)
    else:
        assert_never(multiscale)

    return sdata

class MultiscalesDumper:
    name: Final[str]
    raster_data: Final[xr.DataTree]
    intrinsic: Final[CsVert]
    coord_systems: Final[set[CsVert]]
    transform_edges: Final[Sequence[Edge]]
    labels: Final[Mapping[PurePosixPath, MultiscalesDumper]]
    cs_to_path: Final[Mapping[CsVert, PurePosixPath]]

    def __init__(
        self,
        sdata: FutureSpatialData,
        key: ElementKey,
        intrinsic: CsVert,
        transform_edges: Sequence[Edge], # FIXME: set[Edge]?
        labels: Mapping[PurePosixPath, MultiscalesDumper],
    ):
        coord_systems = {intrinsic}

        _labels_coord_systems = set(itertools.chain(label.coord_systems for label in labels.values()))
        for e in transform_edges:
            for cs in (e.origin, e.target):
                if cs not in _labels_coord_systems:
                    coord_systems.add(cs)

        cs_to_path: dict[CsVert, PurePosixPath] = {cs: PurePosixPath("") for cs in coord_systems}
        intrinsic = sdata.get_intrinsic(key)
        cs_to_path[intrinsic] = PurePosixPath("")
        for label_path, label_dumper in labels.items():
            if label_path.is_absolute():
                raise ValueError(f"Label path {label_path} cannot be absolute")
            for label_cs, label_cs_path in label_dumper.cs_to_path.items():
                label_cs_path = "labels" / label_path / label_cs_path
                if label_cs in cs_to_path:
                    raise ValueError(f"Coordinate system {label_cs} from {label_cs_path} is not unique") # FIXME
                cs_to_path[label_cs] = label_cs_path

        for edge in transform_edges:
            if edge.origin!= intrinsic and edge.target != intrinsic:
                raise ValueError("Origin or target must be the intrinsic") # FIXME
            for vert in (edge.origin, edge.target):
                if vert not in cs_to_path:
                    raise ValueError(f"Reference to {vert} is dangling") # FIXME

        self.name = key.name
        raster_data = sdata.get_image(key) or sdata.get_label(key)
        if raster_data is None:
            raise KeyError(f"Could not find element with key {key}") # FIXME
        self.raster_data = raster_data
        self.intrinsic = intrinsic
        self.coord_systems = coord_systems
        self.transform_edges = transform_edges
        self.labels = labels
        self.cs_to_path = cs_to_path

        super().__init__()

    def contains(self, cs: CsVert) -> bool:
        if cs in self.coord_systems:
            return True
        for label in self.labels.values():
            if label.contains(cs):
                return True
        return False

    def all_coord_systems(self) -> Iterator[CsVert]:
        yield from self.coord_systems
        for label in self.labels.values():
            yield from label.all_coord_systems()

    def _make_ngff_cs_ident(self, csvert: CsVert) -> ozm06ct.CoordinateSystemIdentifier:
        path = self.cs_to_path[csvert]
        return ozm06ct.CoordinateSystemIdentifier(
            name=csvert.name,
            path=None if path == PurePosixPath(".") else path.as_posix(),
        )

    def dump(self) -> oz.OMEZarrMultiscale:
        image_data_tree: xr.DataTree = self.raster_data
        scale0 = image_data_tree["scale0"]["image"] # FIXME: using current silent convention
        assert not isinstance(scale0, xr.DataTree)
        assert isinstance(scale0.data, (da.Array, np.ndarray))

        omz_image = oz.OMEZarrImage(
            data=scale0.data,
            axes=[str(d) for d in scale0.dims], # FIXME: safer alternative?
            axes_units={
                ax.name: ax.unit
                for ax in self.intrinsic.coord_system.axes
                if ax.unit is not None
            },
            axes_types={ax.name: ax.type for ax in self.intrinsic.coord_system.axes},
            name=self.name,
        )

        omz_transforms: list[ozm06ct.AnyTransform] = [
            edge.data["transform"].to_ngff_transform(
                name=edge.data["name"],
                input=self._make_ngff_cs_ident(edge.origin),
                output=self._make_ngff_cs_ident(edge.target),
            )
            for edge in self.transform_edges
        ]

        dumped_labels: list[oz.OMEZarrLabels] = []
        for label in self.labels.values():
            dumped_label = label.dump()
            dumped_labels.append(
                oz.OMEZarrLabels(image=dumped_label.images[0]) # FIXME: use from_pyramid, no cs or transfers??
            )

        return oz.OMEZarrMultiscale(
            image=omz_image, # FIXME: use from_pyramid once it's available
            labels=dumped_labels,
            default_coordinate_system_name=self.intrinsic.provenance.name,
            coordinate_systems=[cs.to_ngff_cs() for cs in self.coord_systems.difference({self.intrinsic})],
            coordinate_transformations=tuple(omz_transforms),
        )

class SceneDumper:
    multiscales: Final[ Mapping[PurePosixPath, MultiscalesDumper] ]
    coord_systems: Final[ Sequence[CsVert] ]
    transform_edges: Final[ Sequence[Edge] ]
    cs_to_path: Final[ Mapping[CsVert, PurePosixPath] ]

    def __init__(
        self,
        *,
        multiscales: Mapping[PurePosixPath, MultiscalesDumper],
        coord_systems: Sequence[CsVert],
        transform_edges: Sequence[Edge],
    ) -> None:
        cs_to_path: dict[CsVert, PurePosixPath] = {cs: PurePosixPath("") for cs in coord_systems}
        for ms_path, ms_dumper in multiscales.items():
            if ms_path.is_absolute():
                raise ValueError(f"Raster name cannot be absolute: {ms_path}")
            for raster_cs, raster_cs_path in ms_dumper.cs_to_path.items():
                raster_cs_path = ms_path / raster_cs_path
                if raster_cs in cs_to_path:
                    raise ValueError(f"Coordinate system {raster_cs} from {raster_cs_path} is not unique")
                cs_to_path[raster_cs] = raster_cs_path

        for edge in transform_edges:
            for vert in (edge.origin, edge.target):
                if vert not in cs_to_path:
                    raise ValueError(f"Reference to {vert} is dangling") # FIXME

        self.multiscales = multiscales
        self.coord_systems = coord_systems
        self.transform_edges = transform_edges
        self.cs_to_path = cs_to_path
        super().__init__()

    def _make_ngff_cs_ident(self, csvert: CsVert) -> ozm06ct.CoordinateSystemIdentifier:
        path = self.cs_to_path[csvert]
        return ozm06ct.CoordinateSystemIdentifier(
            name=csvert.name,
            path=None if path == PurePosixPath(".") else path.as_posix(),
        )

    def dump(self) -> oz.OMEZarrScene:
        omz_transforms: list[ozm06ct.AnyTransform] = [
            edge.data["transform"].to_ngff_transform(
                name=edge.data["name"],
                input=self._make_ngff_cs_ident(edge.origin),
                output=self._make_ngff_cs_ident(edge.target),
            )
            for edge in self.transform_edges
        ]

        return oz.OMEZarrScene(
            images={path.as_posix(): r.dump() for path, r in self.multiscales.items()},
            coordinate_systems=[cs.to_ngff_cs() for cs in self.coord_systems],
            coordinate_transformations=omz_transforms,
        )

def _read_multiscale(
    store: str | Path, raster_type: ELEMENT_TYPE_RASTER, reader_format: Format
) -> DataArray | DataTree:
    assert isinstance(store, str | Path)
    assert raster_type in get_args(ELEMENT_TYPE_RASTER), ValueError(
        f"{raster_type} is not a valid raster type. Must be one of {get_args(ELEMENT_TYPE_RASTER)}."
    )

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
    if not isinstance(loaded_node, Multiscales):
        raise TypeError(
            f"Expected {image_loc.basename()} to hold a multiscales node, got {type(loaded_node).__name__}."
        )
    datasets, multiscales = (
        loaded_node.datasets,
        loaded_node.zarr.root_attrs["multiscales"],
    )
    # This works for all versions as in zarr v3 the level of the 'ome' key is taken as root_attrs.
    omero_metadata = loaded_node.zarr.root_attrs.get("omero")
    # TODO: check if below is still valid
    legacy_channels_metadata = loaded_node.zarr.root_attrs.get("channels_metadata", None)  # legacy v0.1
    assert len(multiscales) == 1
    # checking for multiscales[0]["coordinateTransformations"] would make fail
    # something that doesn't have coordinateTransformations in top level
    # which is true for the current version of the spec
    # and for instance in the xenium example
    encoded_ngff_transformations = multiscales[0]["coordinateTransformations"]
    transformations = _get_transformations_from_ngff_dict(encoded_ngff_transformations)
    # if image, read channels metadata
    channels: list[Any] | None = None
    if raster_type == ELEMENT_TYPE.IMAGE:
        if legacy_channels_metadata is not None:
            channels = [d["label"] for d in legacy_channels_metadata["channels"]]
        if omero_metadata is not None:
            channels = [d["label"] for d in omero_metadata["channels"]]
    axes = [i["name"] for i in node.metadata["axes"]]
    if len(datasets) > 1:
        arrays = [loaded_node.array(resolution=d) for d in datasets]
        msi = dask_arrays_to_datatree(arrays, dims=axes, channels=channels)
        _set_transformations(msi, transformations)
        return compute_coordinates(msi)

    data = loaded_node.array(resolution=datasets[0])
    si = DataArray(
        data,
        name=ELEMENT_TYPE.IMAGE,
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


def _get_raster_element_group(raster_type: ELEMENT_TYPE_RASTER, group: zarr.Group, element_name: str) -> zarr.Group:
    """Get the Zarr group holding a raster element that has just been written.

    Labels are nested one level deeper than images: ome-zarr writes them inside a "labels" group, so for them the
    group of the element is `group/labels/{element_name}`, while for images it is `group` itself.

    Parameters
    ----------
    raster_type
        Whether the element is an image or a labels element.
    group
        The Zarr group the element has been written to.
    element_name
        The name of the raster element.

    Returns
    -------
    The Zarr group of the raster element.
    """
    if raster_type != ELEMENT_TYPE.LABELS:
        return group
    labels_group = group["labels"]
    if not isinstance(labels_group, zarr.Group):
        raise TypeError(f"Expected a zarr group holding the labels, got {type(labels_group).__name__}.")
    element_group = labels_group[element_name]
    if not isinstance(element_group, zarr.Group):
        raise TypeError(
            f"Expected a zarr group holding the label {element_name!r}, got {type(element_group).__name__}."
        )
    return element_group


def _write_raster(
    raster_type: ELEMENT_TYPE_RASTER,
    raster_data: DataArray | DataTree,
    group: zarr.Group,
    name: str,
    raster_format: RasterFormatType,
    storage_options: JSONDict | list[JSONDict] | None = None,
    raster_compressor: dict[Literal["lz4", "zstd"], int] | None = None,
    label_metadata: JSONDict | None = None,
    **metadata: Any,
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
    assert raster_type in get_args(ELEMENT_TYPE_RASTER), ValueError(
        f"{raster_type} is not a valid raster type. Must be one of {get_args(ELEMENT_TYPE_RASTER)}."
    )
    # "name" and "label_metadata" are only used for labels. "name" is written in write_multiscale_ngff() but ignored in
    # write_image_ngff() (possibly an ome-zarr-py bug). We only use "name" to ensure correct group access in the
    # ome-zarr API.
    if raster_type == ELEMENT_TYPE.LABELS:
        metadata["name"] = name
        metadata["label_metadata"] = label_metadata

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

    group = _get_raster_element_group(raster_type, group, name)
    if raster_type == ELEMENT_TYPE.IMAGE:
        # ome-zarr-py >= 0.18 no longer writes the omero channel metadata, so we write it ourselves.
        overwrite_channel_names(group, raster_data)
    if ATTRS_KEY not in group.attrs:
        group.attrs[ATTRS_KEY] = {}
    stored_attrs = group.attrs[ATTRS_KEY]
    if not isinstance(stored_attrs, Mapping):
        raise TypeError(f"Expected {ATTRS_KEY} to be a JSON object, got {type(stored_attrs).__name__}.")
    attrs = dict(stored_attrs)
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
    storage_options: JSONDict | list[JSONDict] | None,
    raster_compressor: dict[Literal["lz4", "zstd"], int] | None,
    zarr_format: int = 3,
) -> JSONDict | list[JSONDict] | None:
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
    if zarr_format not in (2, 3):
        raise ValueError(f"Unsupported zarr format {zarr_format}; expected 2 or 3.")

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
    raster_type: ELEMENT_TYPE_RASTER,
    group: zarr.Group,
    element_name: str,
    raster_data: DataArray,
    raster_format: RasterFormatType,
    storage_options: JSONDict | list[JSONDict] | None,
    raster_compressor: dict[Literal["lz4", "zstd"], int] | None,
    **metadata: Any,
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
    write_single_scale_ngff = write_image_ngff if raster_type == ELEMENT_TYPE.IMAGE else write_labels_ngff

    data = raster_data.data
    transformations = _get_transformations(raster_data)
    assert transformations is not None  # mypy: validate_element() in _write_element guarantees this
    input_axes: tuple[str, ...] = tuple(str(dim) for dim in raster_data.dims)
    parsed_axes = _get_valid_axes(axes=list(input_axes), fmt=raster_format)
    storage_options = _prepare_storage_options(storage_options)
    # Apply compression if specified
    storage_options = _apply_compression(storage_options, raster_compressor, zarr_format=raster_format.zarr_format)

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

    trans_group = _get_raster_element_group(raster_type, group, element_name)
    overwrite_coordinate_transformations_raster(
        group=trans_group,
        transformations=transformations,
        axes=input_axes,
        raster_format=raster_format,
    )


def _write_raster_datatree(
    raster_type: ELEMENT_TYPE_RASTER,
    group: zarr.Group,
    element_name: str,
    raster_data: DataTree,
    raster_format: RasterFormatType,
    storage_options: JSONDict | list[JSONDict] | None,
    raster_compressor: dict[Literal["lz4", "zstd"], int] | None,
    **metadata: Any,
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
    write_multi_scale_ngff = (
        write_multiscale_ngff if raster_type == ELEMENT_TYPE.IMAGE else write_multiscale_labels_ngff
    )
    data = get_pyramid_levels(raster_data, attr="data")
    list_of_input_axes: list[Any] = get_pyramid_levels(raster_data, attr="dims")
    assert len(set(list_of_input_axes)) == 1
    input_axes = list_of_input_axes[0]
    # saving only the transformations of the first scale
    d = dict(raster_data["scale0"])
    assert len(d) == 1
    xdata = next(iter(d.values()))
    if not isinstance(xdata, DataArray):
        raise TypeError(f"Expected the first scale to hold a DataArray, got {type(xdata).__name__}.")
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

    trans_group = _get_raster_element_group(raster_type, group, element_name)
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
    **metadata: Any,
) -> None:
    if element_format.zarr_format == 2:
        warnings.warn(
            message=WritingToZarrV2DeprecationWarning.message, category=WritingToZarrV2DeprecationWarning, stacklevel=2
        )

    _write_raster(
        raster_type=ELEMENT_TYPE.IMAGE,
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
    if element_format.zarr_format == 2:
        warnings.warn(
            message=WritingToZarrV2DeprecationWarning.message, category=WritingToZarrV2DeprecationWarning, stacklevel=2
        )

    _write_raster(
        raster_type=ELEMENT_TYPE.LABELS,
        raster_data=labels,
        group=group,
        name=name,
        raster_format=element_format,
        storage_options=storage_options,
        raster_compressor=raster_compressor,
        label_metadata=label_metadata,
        **metadata,
    )
