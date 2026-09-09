from __future__ import annotations

from collections.abc import Sequence
from typing import Final, Literal

import ome_zarr.classes.image as ozi
import ome_zarr_models.v06.coordinate_transforms as ozm06ct


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
        if model.type != "channel" and model.type != "space":
            raise AxisParsingException(f"Can't handle axis of type {model.type}")
        if not isinstance(model.unit, (str, type(None))):
            raise AxisParsingException("Can't handle axis unit")
        return Axis(
            name=name,
            type=model.type,
            unit=model.unit,
            long_name=model.longName,
        )


class CoordSystemParsingException(Exception):
    pass


class DuplicateAxisNameError(Exception):
    def __init__(self, *, axis_name: str) -> None:
        self.axis_name = axis_name
        super().__init__(f"Axis name '{axis_name}' is used more than once")


class CoordSystem:
    """
    Representation of a coordinate system.
    """

    name: Final[str]
    axes: Final[tuple[Axis, ...]]

    virtual: Final[bool]
    """A virtual coordinate system exists as an intermediate step between
    non-virtual coordinate systems and is usually ignored during serialization"""

    def __init__(self, name: str, axes: Sequence[Axis], virtual: bool = False):
        """
        Parameters
        ----------
        name
            name of the coordinate system
        axes
            axes of the coordinate system
        virtual
            vitual coordinate systems don't serialize to NGFF

        Raises
        ------
        DuplicateAxisNameError
            if `axes` contains axes with duplicate names
        """
        self.name = name
        self.axes = tuple(axes)
        self.virtual = virtual
        seen_names: set[str] = set()
        for axis in self.axes:
            if axis.name in seen_names:
                raise DuplicateAxisNameError(axis_name=axis.name)
            seen_names.add(axis.name)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.name!r}, {self.axes})"

    def __hash__(self) -> int:
        return hash((self.name, self.axes, self.virtual))

    @classmethod
    def try_from_model(cls, model: ozi.CoordinateSystem) -> CoordSystem:
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
            name=model.name,
            axes=[Axis.try_from_model(axis) for axis in model.axes],
        )

    @classmethod
    def try_from_model_or_default[T](cls, model: ozi.CoordinateSystem | None, *, default: T) -> CoordSystem | T:
        """
        Parse a `CoordSystem` from `model`, or return `default` if `model` is None.

        Raises
        ------
        AxisParsingException
            if `model` is not None and any of its axes fails to parse
        """
        if model is not None:
            return CoordSystem.try_from_model(model)
        return default

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
