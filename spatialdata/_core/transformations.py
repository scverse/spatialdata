from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING, Union

import numpy as np
import xarray as xr
from xarray import DataArray

from spatialdata._core.ngff.ngff_transformations import (
    NgffAffine,
    NgffBaseTransformation,
    NgffIdentity,
    NgffMapAxis,
    NgffScale,
    NgffSequence,
    NgffTranslation,
)
from spatialdata._logging import logger

# from spatialdata._core.ngff.ngff_coordinate_system import NgffCoordinateSystem
from spatialdata._types import ArrayLike

if TYPE_CHECKING:
    from spatialdata._core.core_utils import SpatialElement, ValidAxis_t

__all__ = [
    "BaseTransformation",
    "Identity",
    "MapAxis",
    "Translation",
    "Scale",
    "Affine",
    "Sequence",
]

# I was using "from numbers import Number" but this led to mypy errors, so I switched to the following:
Number = Union[int, float]
TRANSFORMATIONS_MAP: dict[NgffBaseTransformation, type[BaseTransformation]] = {}


class BaseTransformation(ABC):
    """Base class for all transformations."""

    def _validate_axes(self, axes: tuple[ValidAxis_t, ...]) -> None:
        # to avoid circular imports
        from spatialdata._core.core_utils import validate_axis_name

        for ax in axes:
            validate_axis_name(ax)
        if len(axes) != len(set(axes)):
            raise ValueError("Axes must be unique.")

    @staticmethod
    def _empty_affine_matrix(input_axes: tuple[ValidAxis_t, ...], output_axes: tuple[ValidAxis_t, ...]) -> ArrayLike:
        m = np.zeros((len(output_axes) + 1, len(input_axes) + 1))
        m[-1, -1] = 1
        return m

    def _indent(self, indent: int) -> str:
        return " " * indent * 4

    @abstractmethod
    def _repr_transformation_description(self, indent: int = 0) -> str:
        pass

    def _repr_indent(self, indent: int = 0) -> str:
        s = f"{self._indent(indent)}{type(self).__name__} "
        s += f"{self._repr_transformation_description(indent + 1)}"
        return s

    def __repr__(self) -> str:
        return self._repr_indent(0)

    @classmethod
    @abstractmethod
    def _from_ngff(cls, d: NgffBaseTransformation) -> BaseTransformation:
        pass

    @classmethod
    def from_ngff(cls, d: NgffBaseTransformation) -> BaseTransformation:
        transformation = TRANSFORMATIONS_MAP[type(d)]._from_ngff(d)
        return transformation

    @abstractmethod
    def to_ngff(self) -> NgffBaseTransformation:
        pass

    @abstractmethod
    def inverse(self) -> BaseTransformation:
        pass

    # @abstractmethod
    # def transform_points(self, points: ArrayLike) -> ArrayLike:
    #     pass

    @abstractmethod
    def to_affine_matrix(self, input_axes: tuple[ValidAxis_t, ...], output_axes: tuple[ValidAxis_t, ...]) -> ArrayLike:
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

    # helper functions to transform coordinates; we use an internal representation based on xarray.DataArray
    #
    # warning: the function _transform_coordinates() will always expect points that are x, y or x, y, z and return
    # points that are x, y or x, y, z (it allows the case in which the number of dimensions changes) the function
    # to_affine_matrix() is public so it doesn't add this costraint, but this function is used only to transform
    # spatial elements, where we always have x, y, z
    @abstractmethod
    def _transform_coordinates(self, data: DataArray) -> DataArray:
        raise NotImplementedError

    # utils for the internal representation of coordinates using xarray
    @staticmethod
    def _xarray_coords_get_coords(data: DataArray) -> tuple[ValidAxis_t, ...]:
        axes = data.coords["dim"].data.tolist()
        assert isinstance(axes, list)
        return tuple(axes)

    @staticmethod
    def _xarray_coords_get_column(data: DataArray, axis: ValidAxis_t) -> DataArray:
        return data[:, data["dim"] == axis]

    @staticmethod
    def _xarray_coords_validate_axes(data: DataArray) -> None:
        axes = BaseTransformation._xarray_coords_get_coords(data)
        if axes not in [("x", "y"), ("x", "y", "z")]:
            raise ValueError(f"Invalid axes: {axes}")

    @staticmethod
    def _xarray_coords_filter_axes(data: DataArray, axes: Optional[tuple[ValidAxis_t, ...]] = None) -> DataArray:
        if axes is None:
            axes = ("x", "y", "z")
        return data[:, data["dim"].isin(axes)]

    @staticmethod
    def _xarray_coords_reorder_axes(data: DataArray) -> DataArray:
        axes = BaseTransformation._xarray_coords_get_coords(data)
        if "z" in axes:
            data = data.sel(dim=["x", "y", "z"])
        else:
            data = data.sel(dim=["x", "y"])
        BaseTransformation._xarray_coords_validate_axes(data)
        return data

    def _get_n_spatial_dims(self, axes: tuple[str, ...]) -> int:
        valid_axes = {("c", "z", "y", "x"): 3, ("c", "y", "x"): 2, ("z", "y", "x"): 3, ("y", "x"): 2}
        if axes not in valid_axes:
            raise ValueError(f"Invalid axes: {axes}")
        return valid_axes[axes]

    def transform(self, element: SpatialElement) -> SpatialElement:
        from spatialdata._core._transform_elements import _transform

        transformed = _transform(element, self)
        return transformed


class Identity(BaseTransformation):
    def to_affine_matrix(self, input_axes: tuple[ValidAxis_t, ...], output_axes: tuple[ValidAxis_t, ...]) -> ArrayLike:
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

    def _repr_transformation_description(self, indent: int = 0) -> str:
        return ""

    def _transform_coordinates(self, data: DataArray) -> DataArray:
        return data

    @classmethod
    def _from_ngff(cls, d: NgffBaseTransformation) -> BaseTransformation:
        pass

    def to_ngff(self) -> NgffBaseTransformation:
        pass


# Warning on MapAxis vs NgffMapAxis: MapAxis can add new axes that are not present in input. NgffMapAxis can't do
# this. It can only 1) permute the axis order, 2) eventually assiging the same axis to multiple output axes and 3)
# drop axes. When convering from MapAxis to NgffMapAxis this can be done by returing a Sequence of NgffAffine and
# NgffMapAxis, where the NgffAffine corrects the axes
class MapAxis(BaseTransformation):
    def __init__(self, map_axis: dict[ValidAxis_t, ValidAxis_t]) -> None:
        # to avoid circular imports
        from spatialdata._core.core_utils import validate_axis_name

        assert isinstance(map_axis, dict)
        for des_ax, src_ax in map_axis.items():
            validate_axis_name(des_ax)
            validate_axis_name(src_ax)
        self.map_axis = map_axis

    def inverse(self) -> BaseTransformation:
        if len(self.map_axis.values()) != len(set(self.map_axis.values())):
            raise ValueError("Cannot invert a MapAxis transformation with non-injective map_axis.")
        return MapAxis({des_ax: src_ax for src_ax, des_ax in self.map_axis.items()})

    def to_affine_matrix(self, input_axes: tuple[ValidAxis_t, ...], output_axes: tuple[ValidAxis_t, ...]) -> ArrayLike:
        self._validate_axes(input_axes)
        self._validate_axes(output_axes)
        # validation logic:
        # if an ax is in output_axes, then:
        #    if it is in self.keys, then the corresponding value must be in input_axes
        for ax in output_axes:
            if ax in self.map_axis:
                if self.map_axis[ax] not in input_axes:
                    raise ValueError("Output axis is mapped to an input axis that is not in input_axes.")
        # validation logic:
        # if an ax is in input_axes, then it is either in self.values or in output_axes
        for ax in input_axes:
            if ax not in self.map_axis.values() and ax not in output_axes:
                raise ValueError("Input axis is not mapped to an output axis and is not in output_axes.")
        m = self._empty_affine_matrix(input_axes, output_axes)
        for i_out, ax_out in enumerate(output_axes):
            for i_in, ax_in in enumerate(input_axes):
                if ax_out in self.map_axis:
                    if self.map_axis[ax_out] == ax_in:
                        m[i_out, i_in] = 1
                elif ax_in == ax_out:
                    m[i_out, i_in] = 1
        return m

    def _repr_transformation_description(self, indent: int = 0) -> str:
        s = "\n"
        for k, v in self.map_axis.items():
            s += f"{self._indent(indent)}{k} <- {v}\n"
        s = s[:-1]
        return s

    def _transform_coordinates(self, data: DataArray) -> DataArray:
        self._xarray_coords_validate_axes(data)
        data_input_axes = self._xarray_coords_get_coords(data)
        data_output_axes = _get_current_output_axes(self, data_input_axes)

        transformed = []
        for ax in data_output_axes:
            if ax in self.map_axis:
                column = self._xarray_coords_get_column(data, self.map_axis[ax])
            else:
                column = self._xarray_coords_get_column(data, ax)
            column.coords["dim"] = [ax]
            transformed.append(column)
        to_return = xr.concat(transformed, dim="dim")
        to_return = self._xarray_coords_reorder_axes(to_return)
        return to_return

    @classmethod
    def _from_ngff(cls, d: NgffBaseTransformation) -> BaseTransformation:
        pass

    def to_ngff(self) -> NgffBaseTransformation:
        pass


class Translation(BaseTransformation):
    def __init__(self, translation: Union[list[Number], ArrayLike], axes: tuple[ValidAxis_t, ...]) -> None:
        self.translation = self._parse_list_into_array(translation)
        self._validate_axes(axes)
        self.axes = axes
        assert len(self.translation) == len(self.axes)

    def inverse(self) -> BaseTransformation:
        return Translation(-self.translation, self.axes)

    def to_affine_matrix(self, input_axes: tuple[ValidAxis_t, ...], output_axes: tuple[ValidAxis_t, ...]) -> ArrayLike:
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
        return m

    def _repr_transformation_description(self, indent: int = 0) -> str:
        return f"({', '.join(self.axes)})\n{self._indent(indent)}{self.translation}"

    def _transform_coordinates(self, data: DataArray) -> DataArray:
        self._xarray_coords_validate_axes(data)
        translation = DataArray(self.translation, coords={"dim": self.axes})
        transformed = data + translation
        to_return = self._xarray_coords_reorder_axes(transformed)
        return to_return

    @classmethod
    def _from_ngff(cls, d: NgffBaseTransformation) -> BaseTransformation:
        pass

    def to_ngff(self) -> NgffBaseTransformation:
        pass


class Scale(BaseTransformation):
    def __init__(self, scale: Union[list[Number], ArrayLike], axes: tuple[ValidAxis_t, ...]) -> None:
        self.scale = self._parse_list_into_array(scale)
        self._validate_axes(axes)
        self.axes = axes
        assert len(self.scale) == len(self.axes)

    def inverse(self) -> BaseTransformation:
        return Scale(1 / self.scale, self.axes)

    def to_affine_matrix(self, input_axes: tuple[ValidAxis_t, ...], output_axes: tuple[ValidAxis_t, ...]) -> ArrayLike:
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
        return m

    def _repr_transformation_description(self, indent: int = 0) -> str:
        return f"({', '.join(self.axes)})\n{self._indent(indent)}{self.scale}"

    def _transform_coordinates(self, data: DataArray) -> DataArray:
        self._xarray_coords_validate_axes(data)
        scale = DataArray(self.scale, coords={"dim": self.axes})
        transformed = data * scale
        to_return = self._xarray_coords_reorder_axes(transformed)
        return to_return

    @classmethod
    def _from_ngff(cls, d: NgffBaseTransformation) -> BaseTransformation:
        pass

    def to_ngff(self) -> NgffBaseTransformation:
        pass


class Affine(BaseTransformation):
    def __init__(
        self,
        matrix: Union[list[Number], ArrayLike],
        input_axes: tuple[ValidAxis_t, ...],
        output_axes: tuple[ValidAxis_t, ...],
    ) -> None:
        self._validate_axes(input_axes)
        self._validate_axes(output_axes)
        self.input_axes = input_axes
        self.output_axes = output_axes
        self.matrix = self._parse_list_into_array(matrix)
        assert self.matrix.dtype == float
        if self.matrix.shape != (len(output_axes) + 1, len(input_axes) + 1):
            raise ValueError("Invalid shape for affine matrix.")
        if not np.array_equal(self.matrix[-1, :-1], np.zeros(len(input_axes))):
            raise ValueError("Affine matrix must be homogeneous.")
        assert self.matrix[-1, -1] == 1.0

    def inverse(self) -> BaseTransformation:
        inv = np.linalg.inv(self.matrix)
        return Affine(inv, self.output_axes, self.input_axes)

    def to_affine_matrix(self, input_axes: tuple[ValidAxis_t, ...], output_axes: tuple[ValidAxis_t, ...]) -> ArrayLike:
        self._validate_axes(input_axes)
        self._validate_axes(output_axes)
        # validation logic:
        # either an ax in input_axes is present in self.input_axes or it is not present in self.output_axes. That is:
        # if the ax in input_axes is mapped by the matrix to something, ok, otherwise it must not appear as the
        # output of the matrix
        for ax in input_axes:
            if ax not in self.input_axes and ax in self.output_axes:
                raise ValueError(
                    f"The axis {ax} is not an input of the affine transformation but it appears as output. Probably "
                    f"you want to remove it from the input_axes of the to_affine_matrix() call."
                )
        # asking a representation of the affine transformation that is not using the matrix
        if len(set(input_axes).intersection(self.input_axes)) == 0:
            logger.warning(
                "Asking a representation of the affine transformation that is not using the matrix: "
                f"self.input_axews = {self.input_axes}, self.output_axes = {self.output_axes}, "
                f"input_axes = {input_axes}, output_axes = {output_axes}"
            )
        m = self._empty_affine_matrix(input_axes, output_axes)
        for i_out, ax_out in enumerate(output_axes):
            for i_in, ax_in in enumerate(input_axes):
                if ax_out in self.output_axes:
                    j_out = self.output_axes.index(ax_out)
                    if ax_in in self.input_axes:
                        j_in = self.input_axes.index(ax_in)
                        m[i_out, i_in] = self.matrix[j_out, j_in]
                    m[i_out, -1] = self.matrix[j_out, -1]
                elif ax_in == ax_out:
                    m[i_out, i_in] = 1
        return m

    def _repr_transformation_description(self, indent: int = 0) -> str:
        s = f"({', '.join(self.input_axes)} -> {', '.join(self.output_axes)})\n"
        for row in self.matrix:
            s += f"{self._indent(indent)}{row}\n"
        s = s[:-1]
        return s

    def _transform_coordinates(self, data: DataArray) -> DataArray:
        self._xarray_coords_validate_axes(data)
        data_input_axes = self._xarray_coords_get_coords(data)
        data_output_axes = _get_current_output_axes(self, data_input_axes)
        matrix = self.to_affine_matrix(data_input_axes, data_output_axes)
        transformed = (matrix @ np.vstack((data.data.T, np.ones(len(data))))).T[:, :-1]
        to_return = DataArray(transformed, coords={"points": data.coords["points"], "dim": data_output_axes})
        self._xarray_coords_filter_axes(to_return)
        to_return = self._xarray_coords_reorder_axes(to_return)
        return to_return

    @classmethod
    def _from_ngff(cls, d: NgffBaseTransformation) -> BaseTransformation:
        pass

    def to_ngff(self) -> NgffBaseTransformation:
        pass


class Sequence(BaseTransformation):
    def __init__(self, transformations: list[BaseTransformation]) -> None:
        self.transformations = transformations

    def inverse(self) -> BaseTransformation:
        return Sequence([t.inverse() for t in self.transformations[::-1]])

    # this wrapper is used since we want to return just the affine matrix from to_affine_matrix(), but we need to
    # return two values for the recursive logic to work
    def _to_affine_matrix_wrapper(
        self, input_axes: tuple[ValidAxis_t, ...], output_axes: tuple[ValidAxis_t, ...], _nested_sequence: bool = False
    ) -> tuple[ArrayLike, tuple[ValidAxis_t, ...]]:
        self._validate_axes(input_axes)
        self._validate_axes(output_axes)
        if not all([ax in output_axes for ax in input_axes]):
            raise ValueError("Input axes must be a subset of output axes.")

        current_input_axes = input_axes
        current_output_axes = _get_current_output_axes(self.transformations[0], current_input_axes)
        m = self.transformations[0].to_affine_matrix(current_input_axes, current_output_axes)
        print(f"# 0: current_input_axes = {current_input_axes}, current_output_axes = {current_output_axes}")
        print(self.transformations[0])
        print()
        for i, t in enumerate(self.transformations[1:]):
            current_input_axes = current_output_axes
            # in the case of nested Sequence transformations, only the very last transformation in the outer sequence
            # will force the output to be the one specified by the user. To identify the original call from the
            # nested calls we use the _nested_sequence flag
            if i == len(self.transformations) - 2 and not _nested_sequence:
                current_output_axes = output_axes
            else:
                current_output_axes = _get_current_output_axes(t, current_input_axes)
            print(f"# {i + 1}: current_input_axes = {current_input_axes}, current_output_axes = {current_output_axes}")
            print(t)
            print()
            # lhs hand side
            if not isinstance(t, Sequence):
                lhs = t.to_affine_matrix(current_input_axes, current_output_axes)
            else:
                lhs, adjusted_current_output_axes = t._to_affine_matrix_wrapper(
                    current_input_axes, current_output_axes, _nested_sequence=True
                )
                current_output_axes = adjusted_current_output_axes
            try:
                m = lhs @ m
            except ValueError as e:
                # to debug
                raise e
        return m, current_output_axes

    def to_affine_matrix(
        self, input_axes: tuple[ValidAxis_t, ...], output_axes: tuple[ValidAxis_t, ...], _nested_sequence: bool = False
    ) -> ArrayLike:
        matrix, current_output_axes = self._to_affine_matrix_wrapper(input_axes, output_axes)
        assert current_output_axes == output_axes
        return matrix

    def _repr_transformation_description(self, indent: int = 0) -> str:
        s = "\n"
        for t in self.transformations:
            s += f"{t._repr_indent(indent=indent)}\n"
        s = s[:-1]
        return s

    def _transform_coordinates(self, data: DataArray) -> DataArray:
        for t in self.transformations:
            data = t._transform_coordinates(data)
        self._xarray_coords_validate_axes(data)
        return data

    @classmethod
    def _from_ngff(cls, d: NgffBaseTransformation) -> BaseTransformation:
        pass

    def to_ngff(self) -> NgffBaseTransformation:
        pass


def _get_current_output_axes(
    transformation: BaseTransformation, input_axes: tuple[ValidAxis_t, ...]
) -> tuple[ValidAxis_t, ...]:
    if (
        isinstance(transformation, Identity)
        or isinstance(transformation, Translation)
        or isinstance(transformation, Scale)
    ):
        return input_axes
    elif isinstance(transformation, MapAxis):
        map_axis_input_axes = set(transformation.map_axis.values())
        set(transformation.map_axis.keys())
        to_return = []
        for ax in input_axes:
            if ax not in map_axis_input_axes:
                assert ax not in to_return
                to_return.append(ax)
            else:
                mapped = [ax_out for ax_out, ax_in in transformation.map_axis.items() if ax_in == ax]
                assert all([ax_out not in to_return for ax_out in mapped])
                to_return.extend(mapped)
        return tuple(to_return)
    elif isinstance(transformation, Affine):
        to_return = []
        add_affine_output_axes = False
        for ax in input_axes:
            if ax not in transformation.input_axes:
                assert ax not in to_return
                to_return.append(ax)
            else:
                add_affine_output_axes = True
        if add_affine_output_axes:
            for ax in transformation.output_axes:
                if ax not in to_return:
                    to_return.append(ax)
                else:
                    raise ValueError(
                        f"Trying to query an invalid representation of an affine matrix: the ax {ax} is not "
                        f"an input axis of the affine matrix but it appears both as output as input of the "
                        f"matrix representation being queried"
                    )
        return tuple(to_return)
    elif isinstance(transformation, Sequence):
        return input_axes
    else:
        raise ValueError("Unknown transformation type.")


TRANSFORMATIONS_MAP[type[NgffIdentity]] = Identity
TRANSFORMATIONS_MAP[type[NgffMapAxis]] = MapAxis
TRANSFORMATIONS_MAP[type[NgffTranslation]] = Translation
TRANSFORMATIONS_MAP[type[NgffScale]] = Scale
TRANSFORMATIONS_MAP[type[NgffAffine]] = Affine
TRANSFORMATIONS_MAP[type[NgffSequence]] = Sequence
