from __future__ import annotations

from collections.abc import Sequence
from typing import Final, Literal

import ome_zarr.classes.image as ozi
import pydantic as pyd


class AxisParsingException(Exception):
    pass


class Axis:
    """
    Representation of an axis, following the NGFF specification.

    Attributes
    ----------
    name
        name of the axis.
    type
        type of the axis. Should be in ["channel", "space"].
    unit
        unit of the axis. For a set of valid options see https://ngff.openmicroscopy.org/
    long_name:
        a longer, human-friendly name for this axis
    """

    name: Final[str]
    type: Final[Literal["space", "channel"]]
    unit: Final[str | None]
    long_name: Final[str | None]

    class LegacyModel(pyd.BaseModel):
        name: Literal["x", "y", "z", "c"]
        type: Literal["space", "channel"]

    def __init__(
        self, *, name: str, type: Literal["space", "channel"], unit: str | None = None, long_name: str | None = None
    ):
        self.name = name
        self.type = type
        self.unit = unit
        self.long_name = long_name

    def cloned_with(self, *, unit: str | None) -> Axis:
        return Axis(name=self.name, type=self.type, unit=unit or self.unit, long_name=self.long_name)

    def __hash__(self) -> int:
        return hash((self.name, self.type, self.unit, self.long_name))

    def __repr__(self) -> str:
        return f"NgffAxis(name={self.name}, type={self.type})"

    def __eq__(self, value: object, /) -> bool:
        if not isinstance(value, Axis):
            return False
        return (
            self.name == value.name
            and self.type == value.type
            and self.unit == value.unit
            and self.long_name == value.long_name
        )

    @classmethod
    def try_from_model(cls, model: ozi.Axis) -> Axis:
        name = model.name
        if name is None:
            raise AxisParsingException("Axis doesn't have a name")
        if model.type != "channel" and model.type != "space":
            raise AxisParsingException(f"Can't handle axis of type {model.type}")
        if not isinstance(model.unit, str):
            raise AxisParsingException("Can't handle axis unit")
        return Axis(
            name=name,
            type=model.type,
            unit=model.unit,
            long_name=model.longName,
        )

    @classmethod
    def try_from_dict(cls, d: pyd.JsonValue) -> Axis:
        model = ozi.Axis.model_validate(d)
        return Axis.try_from_model(model)

    def to_model(self) -> ozi.Axis:
        return ozi.Axis(
            discrete=False,
            longName=self.long_name,
            name=self.name,
            type=self.type,
            unit=self.unit,
        )


class CoordSystemParsingException(Exception):
    pass


class CoordSystem:
    """
    Representation of a coordinate system, following the NGFF specification.

    Parameters
    ----------
    name
        name of the coordinate system
    axes
        names of the axes of the coordinate system
    """

    name: Final[str]
    axes: Final[tuple[Axis, ...]]

    virtual: Final[bool]
    """A virtual coordinate system exists as an intermediate step between
    non-virtual coordinate systems and is usually ignored during serialization"""

    class LegacyAxes:
        pass

    def __init__(self, name: str, axes: Sequence[Axis], virtual: bool = False):
        self.name = name
        self.axes = tuple(axes)
        self.virtual = virtual
        if len(self.axes) != len({axis.name for axis in self.axes}):
            raise ValueError("Axes names must be unique")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name!r}, {self.axes})"

    def __hash__(self) -> int:
        return hash((self.name, self.axes, self.virtual))

    @classmethod
    def try_from_model(cls, model: ozi.CoordinateSystem) -> CoordSystem:
        axes: list[Axis] = []
        for axis in model.axes:
            if isinstance(parsed := Axis.try_from_model(axis), Exception):
                raise CoordSystemParsingException(parsed)  # FIXME
            axes.append(parsed)
        return CoordSystem(
            name=model.name,
            axes=axes,
        )

    @classmethod
    def try_from_model_or_default[T](cls, model: ozi.CoordinateSystem | None, *, default: T) -> CoordSystem | T:
        if model is not None:
            return CoordSystem.try_from_model(model)
        return default

    def to_model(self) -> ozi.CoordinateSystem | None:
        if self.virtual:
            return None
        return ozi.CoordinateSystem(
            name=self.name,
            axes=tuple(ax.to_model() for ax in self.axes),
        )

    def to_model_cs_ident(self) -> ozi.CoordinateSystemIdentifier | None:
        input = self.to_model()
        if input is None:
            return None
        return ozi.CoordinateSystemIdentifier(name=input.name)

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
