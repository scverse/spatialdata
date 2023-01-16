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
    def __init__(self, map_axis: dict[str, str]) -> None:
        self.map_axis = map_axis

    def inverse(self) -> BaseTransformation:
        raise NotImplementedError()

    def to_affine_matrix(self, input_axes: list[ValidAxis_t], output_axes: list[ValidAxis_t]) -> ArrayLike:
        raise NotImplementedError()


class Scale(BaseTransformation):
    def __init__(self, scale: Union[list[Number], ArrayLike], axes: list[ValidAxis_t]) -> None:
        self.scale = self._parse_list_into_array(scale)
        self._validate_axes(axes)
        assert len(self.scale) == len(axes)

    def inverse(self) -> BaseTransformation:
        raise NotImplementedError()

    def to_affine_matrix(self, input_axes: list[ValidAxis_t], output_axes: list[ValidAxis_t]) -> ArrayLike:
        raise NotImplementedError()


class Translation(BaseTransformation):
    def __init__(self, translation: Union[list[Number], ArrayLike], axes: list[ValidAxis_t]) -> None:
        self.translation = self._parse_list_into_array(translation)
        assert len(self.translation) == len(axes)

    def inverse(self) -> BaseTransformation:
        raise NotImplementedError()

    def to_affine_matrix(self, input_axes: list[ValidAxis_t], output_axes: list[ValidAxis_t]) -> ArrayLike:
        raise NotImplementedError()


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
