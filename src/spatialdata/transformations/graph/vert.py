from __future__ import annotations

import uuid
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Callable, Final, Literal, NamedTuple, Type, TypedDict, assert_never

import networkx as nx
import ome_zarr_models.v06.coordinate_transforms as ozm06ct

from spatialdata.transformations.graph.edge import parse_ngff_transf

if TYPE_CHECKING:
    from spatialdata.transformations.graph.edge import BaseTransformationEdge

import ome_zarr_models.v06.coordinate_transforms as ozm06ct


class AxisParsingException(Exception):
    pass


@dataclass(frozen=True)
class Axis:
    """Representation of a coordinate system axis"""

    name: Final[str]
    type: Final[Literal["space", "channel"]]
    unit: Final[str | None] = None
    "unit of the axis. For a set of valid options see https://ngff.openmicroscopy.org/"
    long_name: Final[str | None] = None
    "a longer, human-friendly name for this axis"

    def cloned_with(self, *, unit: str | None) -> Axis:
        return Axis(name=self.name, type=self.type, unit=unit or self.unit, long_name=self.long_name)

    def __hash__(self) -> int:
        return hash((self.name, self.type, self.unit, self.long_name))

    def __repr__(self) -> str:
        return f"Axis(name={self.name}, type={self.type})"

    def __eq__(self, value: Axis, /) -> bool: # pyright: ignore[reportIncompatibleMethodOverride]
        if not isinstance(value, Axis): # pyright: ignore[reportUnnecessaryIsInstance]
            return False
        return (
            self.name == value.name
            and self.type == value.type
            and self.unit == value.unit
            and self.long_name == value.long_name
        )

    @classmethod
    def try_from_model(cls, model: ozm06ct.Axis) -> Axis:
        """
        Parse an `Axis` from an ome-zarr-models axis model.

        Raises
        ------
        AxisParsingException
            if `model` doesn't have a name, has a type other than "channel" or "space", or has a unit
            that isn't a string or None
        """

        name = model.name
        if name is None:
            raise AxisParsingException("Axis doesn't have a name")
        if model.type is None:
            raise AxisParsingException("Axis doesn't have a type")
        if model.type not in ("channel", "space"):
            raise AxisParsingException(f"Can't handle axis of type {model.type}")
        if not isinstance(model.unit, (str, type(None))):
            raise AxisParsingException("Can't handle axis unit")
        return Axis(
            name=name,
            type=model.type,  # type: ignore[arg-type]
            unit=model.unit,
            long_name=model.longName,
        )

    def to_model(self) -> ozm06ct.Axis:
        return ozm06ct.Axis(
            discrete=False,
            longName=self.long_name,
            name=self.name,
            type=self.type,
            unit=self.unit,
        )


class CoordSystemParsingException(Exception):
    pass


class DuplicateAxisNameError(Exception):
    def __init__(self, *, axis_name: str) -> None:
        self.axis_name = axis_name
        super().__init__(f"Axis name '{axis_name}' is used more than once")


@dataclass(frozen=True)
class CoordSystem:
    """
    A coordinate space that attaches semantics to coordinates via `axes`
    """

    axes: Final[tuple[Axis, ...]]

    def __post_init__(self) -> None:
        """
        Raises
        ------
        DuplicateAxisNameError
            if `axes` contains axes with duplicate names
        """
        seen_names: set[str] = set()
        for axis in self.axes:
            if axis.name in seen_names:
                raise DuplicateAxisNameError(axis_name=axis.name)
            seen_names.add(axis.name)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.axes})"

    def __hash__(self) -> int:
        return hash(self.axes)

    def __eq__(self, value: CoordSystem, /) -> bool: # pyright: ignore[reportIncompatibleMethodOverride]
        return self.axes == value.axes

    @classmethod
    def try_from_model(cls, model: ozm06ct.CoordinateSystem) -> CoordSystem:
        """
        Parse a `CoordSystem` from an ome-zarr-models coordinate system model.

        Raises
        ------
        AxisParsingException
            if any axis in `model.axes` fails to parse
        DuplicateAxisNameError
            if `model.axes` contains axes with duplicate names
        """
        return CoordSystem(
            axes=tuple(Axis.try_from_model(axis) for axis in model.axes),
        )

    @classmethod
    def try_from_model_or_default[T](cls, model: ozm06ct.CoordinateSystem | None, *, default: T) -> CoordSystem | T:
        if model is not None:
            return CoordSystem.try_from_model(model)
        return default

    def to_model(self, name: str) -> ozm06ct.CoordinateSystem:
        return ozm06ct.CoordinateSystem(
            name=name,
            axes=tuple(ax.to_model() for ax in self.axes),
        )

    @property
    def num_axes(self) -> int:
        return len(self.axes)

    @property
    def axes_names(self) -> tuple[str, ...]:
        """Get axes' names"""
        return tuple([ax.name for ax in self.axes])

    @property
    def axes_types(self) -> tuple[str, ...]:
        """Get axes' types"""
        return tuple([ax.type for ax in self.axes])

    def has_axis(self, name: str) -> bool:
        """
        Check the coordinate system has an axis of the given name.

        Parameters
        ----------
        name
            name of the axis.
        """
        return any(axis.name == name for axis in self.axes)

    def get_axis(self, name: str) -> Axis | None:
        """Get the axis by name"""
        for axis in self.axes:
            if axis.name == name:
                return axis
        return None

    def get_spatial_axes(self) -> Sequence[Axis]:
        return [axis for axis in self.axes if axis.type == "space"]

    @classmethod
    def create_dummy(cls, *, axes: int | Sequence[Axis]) -> CoordSystem:
        if isinstance(axes, int):
            axes = tuple(
                Axis(
                    name=f"axis_{ax_idx}",
                    type="space",  # FIXME
                )
                for ax_idx in range(axes)
            )
        else:
            axes = tuple(axes)

        return CoordSystem(
            axes=axes,
        )


@dataclass(frozen=True)
class NgffCsPath:
    group_path: Final[PurePosixPath]
    name: Final[str]

    def __post_init__(self) -> None:
        assert not self.group_path.is_absolute()

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(path={self.group_path}, name={self.name})"

    def lives_under(self, other: NgffCsPath) -> bool:
        if not self.group_path.is_relative_to(other.group_path):
            return False
        if self.group_path == other.group_path:
            return False
        return True

    def lives_over(self, other: NgffCsPath) -> bool:
        return other.lives_under(self)

    def lives_in_same_level_as(self, other: NgffCsPath) -> bool:
        return self.group_path == other.group_path


@dataclass
class UserCreated:
    name: str

type Provenance = NgffCsPath | UserCreated


class CsVert:
    id: uuid.UUID
    coord_system: CoordSystem
    provenance: Provenance

    def __init__(self, cs: CoordSystem, provenance: Provenance) -> None:
        self.id = uuid.uuid4()
        self.coord_system = cs
        self.provenance = provenance
        super().__init__()

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, value: CsVert, /) -> bool: # pyright: ignore [reportIncompatibleMethodOverride]
        if not isinstance(value, CsVert): # pyright: ignore [reportUnnecessaryIsInstance]
            return False
        return self.id == value.id

    def to_ngff_cs(self) -> ozm06ct.CoordinateSystem:
        return self.coord_system.to_model(name=self.name)

    @property
    def name(self) -> str:
        return self.provenance.name

    @property
    def namespace(self) -> PurePosixPath:
        if isinstance(self.provenance, NgffCsPath):
            return self.provenance.group_path
        if isinstance(self.provenance, UserCreated): # pyright: ignore [reportUnnecessaryIsInstance]
            return PurePosixPath("")
        assert_never(self.provenance)

    def to_ngff_cs_id(self, relative_to: PurePosixPath) -> ozm06ct.CoordinateSystemIdentifier:
        ns = self.namespace.relative_to(relative_to)
        return ozm06ct.CoordinateSystemIdentifier(
            name=self.provenance.name,
            path=None if ns == PurePosixPath("") else str(ns),
        )


class EdgeData(TypedDict): # NOTE: TypedDict because networkx uses dicts
    name: str | None
    transform: BaseTransformationEdge

def hash_edge_data(ed: EdgeData) -> int:
    return hash((ed["name"], ed["transform"]))


class Edge(NamedTuple):
    origin: CsVert
    target: CsVert
    data: EdgeData

    def __hash__(self) -> int:
        return hash((self.origin, self.target))

    def points_down_ngff_hierarchy(self) -> bool:
        if not isinstance(self.origin.provenance, NgffCsPath):
            return False
        if not isinstance(self.target.provenance, NgffCsPath):
            return False
        return self.origin.provenance.lives_over(self.target.provenance)

    def points_up_ngff_hierarchy(self) -> bool:
        if not isinstance(self.origin.provenance, NgffCsPath):
            return False
        if not isinstance(self.target.provenance, NgffCsPath):
            return False
        return self.target.provenance.lives_over(self.origin.provenance)

    def is_horizontal_in_ngff(self) -> bool:
        "Returns true is edge is from an NGFF hierachy and doesn't point up or down the tree"
        if not isinstance(self.origin.provenance, NgffCsPath):
            return False
        if not isinstance(self.target.provenance, NgffCsPath):
            return False
        return self.origin.provenance.lives_in_same_level_as(self.target.provenance)

    def to_model(self, relative_to: PurePosixPath) -> ozm06ct.AnyTransform:
        input = self.origin.to_ngff_cs_id(relative_to)
        output = self.target.to_ngff_cs_id(relative_to)
        return self.data["transform"].to_ngff_transform(
            name=self.data["name"],
            input=input,
            output=output,
        )

    def vert_that_isnt(self, vert: CsVert) -> CsVert:
        if self.origin == vert:
            return self.target
        if self.target == vert:
            return self.origin
        raise ValueError(f"Edge does not contain {vert}")

type VertFilter = Callable[[CsVert, Spaces], bool]

class Verts:
    def __init__(self, verts: Iterator[CsVert], graph: Spaces, filters: Sequence[VertFilter]) -> None:
        self.verts = verts
        self.graph = graph
        self.filters: Sequence[VertFilter] = filters
        super().__init__()

    def __iter__(self) -> Verts:
        return self

    def __next__(self) -> CsVert:
        for vert in self.verts:
            if all(f(vert, self.graph) for f in self.filters):
                return vert
        raise StopIteration()

    def filter(self, filter: Callable[[CsVert, Spaces], bool]) -> Verts:
        return Verts(verts=self.verts, graph=self.graph, filters=[*self.filters, filter])

    def ngff_siblings(self, *, group_path: PurePosixPath) -> Verts:
        def filter(vert: CsVert, graph: Spaces) -> bool:
            if not isinstance(vert.provenance, NgffCsPath):
                return False
            return vert.provenance.group_path == group_path

        return self.filter(filter)

    def connected_to(self, other: CsVert) -> Verts:
        def filter(vert: CsVert, graph: Spaces) -> bool:
            for _ in graph.get_edges_between(vert, other):
                return True
            return False
        return self.filter(filter)

    def has_single_neighbor(self) -> Verts:
        def filter(v: CsVert, g: Spaces) -> bool:
            return len(g.all_neighbors(v)) == 1

        return self.filter(filter)

    def collect[T](self, f: Callable[[Iterator[CsVert]], T]) -> T:
        return f(self)

    def any(self) -> bool:
        for _ in self.verts:
            return True
        return False

type EdgeFilter = Callable[[Edge, Spaces], bool]

class Edges:
    def __init__(
        self, edges: Iterator[Edge], graph: Spaces, filters: Sequence[EdgeFilter]
    ) -> None:
        self.edges = edges
        self.graph = graph
        self.filters = filters

    def __iter__(self) -> Edges:
        return self

    def __next__(self) -> Edge:
        for e in self.edges:
            if all(f(e, self.graph) for f  in self.filters):
                return e
        raise StopIteration()

    def filter(self, filter: Callable[[Edge, Spaces], bool]) -> Edges:
        return Edges(edges=self.edges, graph=self.graph, filters=[*self.filters, filter])

    def directed(self, *, fro: CsVert, to: CsVert) -> Edges:
        def filter(edge: Edge, g: Spaces) -> bool:
            return edge.origin == fro and edge.target == to
        return self.filter(filter)

    def transf_is(self, classes: tuple[Type[BaseTransformationEdge], ...]) -> Edges:
        def filter(edge: Edge, g: Spaces) -> bool:
            return isinstance(edge.data["transform"], classes)
        return self.filter(filter)

    def containing(self, cs: CsVert) -> Edges:
        def filter(edge: Edge, graph: Spaces) -> bool:
            return edge.origin == cs or edge.target == cs
        return self.filter(filter)

    def between(self, cs1: CsVert, cs2: CsVert) -> Edges:
        def filter(edge: Edge, graph: Spaces) -> bool:
            if (edge.origin, edge.target) == (cs1, cs2):
                return True
            if (edge.target, edge.origin) == (cs2, cs1):
                return True
            return False
        return self.filter(filter)

    def ngff_siblings(self) -> Edges:
        def filter(edge: Edge, graph: Spaces) -> bool:
            return edge.is_horizontal_in_ngff()
        return self.filter(filter=filter)

    def map[T](self, f: Callable[[Edge], T]) -> Iterator[T]:
        return map(f, self.edges)

    def collect[T](self, f: Callable[[Iterator[Edge]], T]) -> T:
        return f(self)

    def any(self) -> bool:
        for _ in self.edges:
            return True
        return False


class Spaces:
    def __init__(self) -> None:
        self._graph: nx.MultiDiGraph[CsVert, dict[str, None], EdgeData] = nx.MultiDiGraph()
        super().__init__()

    def incorporate_ngff_graph(
        self,
        group_path: PurePosixPath,
        coord_systems: Sequence[ozm06ct.CoordinateSystem],
        transforms: Sequence[ozm06ct.AnyTransform],
    ) -> None:
        for cs in coord_systems:
            parsed_cs = CoordSystem.try_from_model(cs)
            self.add_cs(cs=parsed_cs, provenance=NgffCsPath(group_path=group_path, name=cs.name))

        for transf in transforms:
            assert transf.input is not None  # as per the spec
            assert transf.output is not None  # as per the spec
            assert transf.input.name is not None  # as per the spec
            assert transf.output.name is not None  # as per the spec
            input_path = NgffCsPath(group_path / (transf.input.path or ""), transf.input.name)
            output_path = NgffCsPath(group_path / (transf.output.path or ""), transf.output.name)

            input = self.find_node(ngff_path=input_path)
            output = self.find_node(ngff_path=output_path)
            parsed = parse_ngff_transf(input=input.coord_system, output=output.coord_system, model=transf)
            self.add_transformation(name=transf.name, input=input, output=output, transf=parsed)

    def incorporate(self, other: Spaces):
        # FIXME: maybe there are some nodes or edges that cannot be incorporated.
        # We might need to checkk ALL OF THEM before partially updating self,
        # so we can succeed or fail in a transaction-like manner

        for vert in other.verts():
            self._graph.add_node(vert) # FIXME:
        for edge in other.edges():
            self.add_transformation(
                input=edge.origin, output=edge.target, transf=edge.data["transform"], name=edge.data["name"]
            )

    def all_neighbors(self, cs: CsVert) -> set[CsVert]:
        import itertools
        return set(itertools.chain(self._graph.successors(cs), self._graph.predecessors(cs)))

    def add_cs(self, *, cs: CoordSystem, provenance: Provenance) -> CsVert:
        node = CsVert(cs, provenance)
        self._graph.add_node(node)
        return node

    def add_transformation(self, *, input: CsVert, output: CsVert, transf: BaseTransformationEdge, name: str | None) -> None:
        assert input.coord_system == transf.input
        assert output.coord_system == transf.output
        _ = self._graph.add_edge(input, output, **EdgeData(name=name, transform=transf))

    def find_direct_edges(self, *, fro: CsVert, to: CsVert) -> Iterator[EdgeData]:
        for src_vert, tgt_vert, edge_data in self._graph.edges.data():
            if self._graph.nodes[src_vert] != fro:
                continue
            if self._graph.nodes[tgt_vert] != to:
                continue
            yield edge_data

    def get_edges_between(self, a: CsVert, b: CsVert) -> Iterator[Edge]:
        for src_vert, tgt_vert, edge_data in self._graph.edges.data():
            if (src_vert, tgt_vert) in ((a,b), (b,a)):
                yield Edge(data=edge_data, origin=src_vert, target=tgt_vert)

    def find_node(self, ngff_path: NgffCsPath) -> CsVert:
        for node in self._graph.nodes:
            provenance = node.provenance
            if not isinstance(provenance, NgffCsPath):
                continue
            if provenance == ngff_path:
                return node
        raise KeyError(f"could not find CS with key {ngff_path}")

    def verts(self) -> Verts:
        return Verts(filters=[], graph=self, verts=iter(self._graph.nodes))

    def edges(self) -> Edges:
        def edges():
            for src_node, tgt_node, edge_data in self._graph.edges.data():
                yield Edge(origin=src_node, target=tgt_node, data=edge_data)
        return Edges(edges=edges(), filters=[], graph=self)
