from __future__ import annotations

from abc import ABC, abstractmethod
from numbers import Number
from typing import Union

import numpy as np

from spatialdata._core.core_utils import ValidAxis_t, validate_axis_name

# from spatialdata._core.ngff.ngff_coordinate_system import NgffCoordinateSystem
from spatialdata._types import ArrayLike

__all__ = [
    "BaseTransformation",
    "Identity",
    "MapAxis",
    "Translation",
    "Scale",
    "Affine",
    "Sequence",
]


class BaseTransformation(ABC):
    """Base class for all transformations."""

    def _validate_axes(self, axes: list[ValidAxis_t]) -> None:
        for ax in axes:
            validate_axis_name(ax)
        if len(axes) != len(set(axes)):
            raise ValueError("Axes must be unique.")

    @staticmethod
    def _empty_affine_matrix(input_axes: list[ValidAxis_t], output_axes: list[ValidAxis_t]) -> ArrayLike:
        m = np.zeros((len(output_axes) + 1, len(input_axes) + 1))
        m[-1, -1] = 1
        return m

    # def __init__(
    #     self,
    # ) -> None:
    #     pass
    #
    # def _indent(self, indent: int) -> str:
    #     return " " * indent * 4
    #
    # def _repr_transformation_signature(self, indent: int = 0) -> str:
    #     if self.input_coordinate_system is not None:
    #         domain = ", ".join(self.input_coordinate_system.axes_names)
    #     else:
    #         domain = ""
    #     if self.output_coordinate_system is not None:
    #         codomain = ", ".join(self.output_coordinate_system.axes_names)
    #     else:
    #         codomain = ""
    #     return f"{self._indent(indent)}{type(self).__name__} ({domain} -> {codomain})"
    #
    # @abstractmethod
    # def _repr_transformation_description(self, indent: int = 0) -> str:
    #     pass
    #
    # def _repr_indent(self, indent: int = 0) -> str:
    #     if isinstance(self, NgffIdentity):
    #         return f"{self._repr_transformation_signature(indent)}"
    #     else:
    #         return f"{self._repr_transformation_signature(indent)}\n{self._repr_transformation_description(indent + 1)}"
    #
    # def __repr__(self) -> str:
    #     return self._repr_indent(0)
    #
    # @classmethod
    # @abstractmethod
    # def _from_dict(cls, d: Transformation_t) -> NgffBaseTransformation:
    #     pass
    #
    # @classmethod
    # def from_dict(cls, d: Transformation_t) -> NgffBaseTransformation:
    #     pass
    #
    #     # d = MappingProxyType(d)
    #     type = d["type"]
    #     # MappingProxyType is readonly
    #     transformation = NGFF_TRANSFORMATIONS[type]._from_dict(d)
    #     if "input" in d:
    #         input_coordinate_system = d["input"]
    #         if isinstance(input_coordinate_system, dict):
    #             input_coordinate_system = NgffCoordinateSystem.from_dict(input_coordinate_system)
    #         transformation.input_coordinate_system = input_coordinate_system
    #     if "output" in d:
    #         output_coordinate_system = d["output"]
    #         if isinstance(output_coordinate_system, dict):
    #             output_coordinate_system = NgffCoordinateSystem.from_dict(output_coordinate_system)
    #         transformation.output_coordinate_system = output_coordinate_system
    #     return transformation
    #
    # @abstractmethod
    # def to_dict(self) -> Transformation_t:
    #     pass

    @abstractmethod
    def inverse(self) -> BaseTransformation:
        pass

    # @abstractmethod
    # def transform_points(self, points: ArrayLike) -> ArrayLike:
    #     pass

    @abstractmethod
    def to_affine_matrix(self, input_axes: list[ValidAxis_t], output_axes: list[ValidAxis_t]) -> ArrayLike:
        pass

    # order of the composition: self is applied first, then the transformation passed as argument
    def compose_with(self, transformations: Union[BaseTransformation, list[BaseTransformation]]) -> BaseTransformation:
        if isinstance(transformations, BaseTransformation):
            return Sequence([self, transformations])
        else:
            return Sequence([self, *transformations])

    # def __eq__(self, other: Any) -> bool:
    #     if not isinstance(other, BaseTransformation):
    #         raise NotImplementedError("Cannot compare BaseTransformation with other types")
    #     return self.to_dict() == other.to_dict()

    @staticmethod
    def _parse_list_into_array(array: Union[list[Number], ArrayLike]) -> ArrayLike:
        if isinstance(array, list):
            array = np.array(array)
        if array.dtype != float:
            array = array.astype(float)
        return array


class Identity(BaseTransformation):
    def to_affine_matrix(self, input_axes: list[ValidAxis_t], output_axes: list[ValidAxis_t]) -> ArrayLike:
        self._validate_axes(input_axes)
        self._validate_axes(output_axes)
        if not all([ax in output_axes for ax in input_axes]):
            raise ValueError("Input axes must be a subset of output axes.")
        m = self._empty_affine_matrix(input_axes, output_axes)
        for i_out, ax_out in enumerate(output_axes):
            for i_in, ax_in in enumerate(input_axes):
                if ax_in == ax_out:
                    m[i_out, i_in] = 1
        return m

    def inverse(self) -> BaseTransformation:
        return self


class MapAxis(BaseTransformation):
    def __init__(self, map_axis: dict[ValidAxis_t, ValidAxis_t]) -> None:
        assert isinstance(map_axis, dict)
        for des_ax, src_ax in map_axis.items():
            validate_axis_name(des_ax)
            validate_axis_name(src_ax)
        self.map_axis = map_axis

    def inverse(self) -> BaseTransformation:
        if len(self.map_axis.values()) != len(set(self.map_axis.values())):
            raise ValueError("Cannot invert a MapAxis transformation with non-injective map_axis.")
        return MapAxis({des_ax: src_ax for src_ax, des_ax in self.map_axis.items()})

    def to_affine_matrix(self, input_axes: list[ValidAxis_t], output_axes: list[ValidAxis_t]) -> ArrayLike:
        self._validate_axes(input_axes)
        self._validate_axes(output_axes)
        if not all([ax in output_axes for ax in input_axes]):
            raise ValueError("Input axes must be a subset of output axes.")
        for ax in self.map_axis.keys():
            if ax not in output_axes:
                raise ValueError(f"Axis {ax} not found in output axes.")
        for ax in self.map_axis.values():
            if ax not in input_axes:
                raise ValueError(f"Axis {ax} not found in input axes.")
        m = self._empty_affine_matrix(input_axes, output_axes)
        for i_out, ax_out in enumerate(output_axes):
            for i_in, ax_in in enumerate(input_axes):
                if ax_out in self.map_axis:
                    if self.map_axis[ax_out] == ax_in:
                        m[i_out, i_in] = 1
                elif ax_in == ax_out:
                    m[i_out, i_in] = 1
        return m


class Translation(BaseTransformation):
    def __init__(self, translation: Union[list[Number], ArrayLike], axes: list[ValidAxis_t]) -> None:
        self.translation = self._parse_list_into_array(translation)
        self._validate_axes(axes)
        self.axes = axes
        assert len(self.translation) == len(self.axes)

    def inverse(self) -> BaseTransformation:
        return Translation(-self.translation, self.axes)

    def to_affine_matrix(self, input_axes: list[ValidAxis_t], output_axes: list[ValidAxis_t]) -> ArrayLike:
        self._validate_axes(input_axes)
        self._validate_axes(output_axes)
        if not all([ax in output_axes for ax in input_axes]):
            raise ValueError("Input axes must be a subset of output axes.")
        m = self._empty_affine_matrix(input_axes, output_axes)
        for i_out, ax_out in enumerate(output_axes):
            for i_in, ax_in in enumerate(input_axes):
                if ax_in == ax_out:
                    m[i_out, i_in] = 1
                    if ax_out in self.axes:
                        m[i_out, -1] = self.translation[self.axes.index(ax_out)]
                elif ax_in == ax_out:
                    m[i_out, i_in] = 1
        return m


class Scale(BaseTransformation):
    def __init__(self, scale: Union[list[Number], ArrayLike], axes: list[ValidAxis_t]) -> None:
        self.scale = self._parse_list_into_array(scale)
        self._validate_axes(axes)
        self.axes = axes
        assert len(self.scale) == len(self.axes)

    def inverse(self) -> BaseTransformation:
        return Scale(1 / self.scale, self.axes)

    def to_affine_matrix(self, input_axes: list[ValidAxis_t], output_axes: list[ValidAxis_t]) -> ArrayLike:
        self._validate_axes(input_axes)
        self._validate_axes(output_axes)
        if not all([ax in output_axes for ax in input_axes]):
            raise ValueError("Input axes must be a subset of output axes.")
        m = self._empty_affine_matrix(input_axes, output_axes)
        for i_out, ax_out in enumerate(output_axes):
            for i_in, ax_in in enumerate(input_axes):
                if ax_in == ax_out:
                    if ax_out in self.axes:
                        scale_factor = self.scale[self.axes.index(ax_out)]
                    else:
                        scale_factor = 1
                    m[i_out, i_in] = scale_factor
                elif ax_in == ax_out:
                    m[i_out, i_in] = 1
        return m


class Affine(BaseTransformation):
    def __init__(
        self, matrix: Union[list[Number], ArrayLike], input_axes: list[ValidAxis_t], output_axes: list[ValidAxis_t]
    ) -> None:
        self._validate_axes(input_axes)
        self._validate_axes(output_axes)
        self.matrix = self._parse_list_into_array(matrix)
        assert self.matrix.dtype == float
        assert self.matrix.shape == (len(output_axes) + 1, len(input_axes) + 1)
        assert np.array_equal(self.matrix[-1, :-1], np.zeros(len(input_axes) - 1))
        assert self.matrix[-1, -1] == 1.0

    def inverse(self) -> BaseTransformation:
        raise NotImplementedError()

    def to_affine_matrix(self, input_axes: list[ValidAxis_t], output_axes: list[ValidAxis_t]) -> ArrayLike:
        raise NotImplementedError()


class Sequence(BaseTransformation):
    def __init__(self, transformations: list[BaseTransformation]) -> None:
        self.transformations = transformations

    def inverse(self) -> BaseTransformation:
        raise NotImplementedError()

    def to_affine_matrix(self, input_axes: list[ValidAxis_t], output_axes: list[ValidAxis_t]) -> ArrayLike:
        raise NotImplementedError()
