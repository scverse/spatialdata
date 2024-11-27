from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Optional, Union
from warnings import warn

import numpy as np
import scipy
import xarray as xr
from xarray import DataArray

from spatialdata._types import ArrayLike
from spatialdata.transformations.ngff.ngff_coordinate_system import (
    NgffCoordinateSystem,
    _get_spatial_axes,
)
from spatialdata.transformations.ngff.ngff_transformations import (
    NgffAffine,
    NgffBaseTransformation,
    NgffIdentity,
    NgffMapAxis,
    NgffScale,
    NgffSequence,
    NgffTranslation,
)

if TYPE_CHECKING:
    from spatialdata._utils import Number
    from spatialdata.models import SpatialElement
    from spatialdata.models._utils import ValidAxis_t

TRANSFORMATIONS_MAP: dict[type[NgffBaseTransformation], type[BaseTransformation]] = {}


class BaseTransformation(ABC):
    """Base class for all transformations."""

    @staticmethod
    def validate_axes(axes: tuple[ValidAxis_t, ...]) -> None:
        """Validate the axes.

        This function is to allow to call validate_axes() from this file
        in multiple places while avoiding circular imports.
        """
        from spatialdata.models._utils import validate_axes

        validate_axes(axes)

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
    def _from_ngff(cls, t: NgffBaseTransformation) -> BaseTransformation:
        pass

    @classmethod
    def from_ngff(cls, t: NgffBaseTransformation) -> BaseTransformation:
        if type(t) not in TRANSFORMATIONS_MAP:
            raise ValueError(f"Conversion from {type(t)} to BaseTransformation is not supported")
        transformation = TRANSFORMATIONS_MAP[type(t)]._from_ngff(t)
        return transformation

    @abstractmethod
    def to_ngff(
        self,
        input_axes: tuple[ValidAxis_t, ...],
        output_axes: tuple[ValidAxis_t, ...],
        unit: str | None = None,
        output_coordinate_system_name: str | None = None,
    ) -> NgffBaseTransformation:
        pass

    def _get_default_coordinate_system(
        self,
        axes: tuple[ValidAxis_t, ...],
        unit: str | None = None,
        name: str | None = None,
        default_to_global: bool = False,
    ) -> NgffCoordinateSystem:
        from spatialdata.transformations.ngff._utils import (
            get_default_coordinate_system,
        )

        cs = get_default_coordinate_system(axes)
        if unit is not None:
            spatial_axes = _get_spatial_axes(cs)
            for ax in spatial_axes:
                cs.get_axis(ax).unit = unit
        if name is not None:
            cs.name = name
        elif default_to_global:
            from spatialdata.models._utils import DEFAULT_COORDINATE_SYSTEM

            cs.name = DEFAULT_COORDINATE_SYSTEM
        return cs

    @abstractmethod
    def inverse(self) -> BaseTransformation:
        pass

    # @abstractmethod
    # def transform_points(self, points: ArrayLike) -> ArrayLike:
    #     pass

    @abstractmethod
    def to_affine_matrix(self, input_axes: tuple[ValidAxis_t, ...], output_axes: tuple[ValidAxis_t, ...]) -> ArrayLike:
        pass

    def to_affine(self, input_axes: tuple[ValidAxis_t, ...], output_axes: tuple[ValidAxis_t, ...]) -> Affine:
        affine_matrix = self.to_affine_matrix(input_axes, output_axes)
        return Affine(affine_matrix, input_axes, output_axes)

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

    # helper functions to transform coordinates; we use an internal representation based on xarray.DataArray
    #
    # warning: the function _transform_coordinates() will always expect points that are x, y or x, y, z and return
    # points that are x, y or x, y, z (it allows the case in which the number of dimensions changes) the function
    # to_affine_matrix() is public so it doesn't add this costraint, but this function is used only to transform
    # SpatialElements, where we always have x, y, z
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
        data = data.sel(dim=["x", "y", "z"]) if "z" in axes else data.sel(dim=["x", "y"])
        BaseTransformation._xarray_coords_validate_axes(data)
        return data

    def _get_n_spatial_dims(self, axes: tuple[str, ...]) -> int:
        valid_axes = {("c", "z", "y", "x"): 3, ("c", "y", "x"): 2, ("z", "y", "x"): 3, ("y", "x"): 2}
        if axes not in valid_axes:
            raise ValueError(f"Invalid axes: {axes}")
        return valid_axes[axes]

    @abstractmethod
    def __eq__(self, other: Any) -> bool:
        pass


class Identity(BaseTransformation):
    def to_affine_matrix(self, input_axes: tuple[ValidAxis_t, ...], output_axes: tuple[ValidAxis_t, ...]) -> ArrayLike:
        self.validate_axes(input_axes)
        self.validate_axes(output_axes)
        if not all(ax in output_axes for ax in input_axes):
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
    def _from_ngff(cls, t: NgffBaseTransformation) -> BaseTransformation:
        assert isinstance(t, NgffIdentity)
        return Identity()

    def to_ngff(
        self,
        input_axes: tuple[ValidAxis_t, ...],
        output_axes: tuple[ValidAxis_t, ...],
        unit: str | None = None,
        output_coordinate_system_name: str | None = None,
    ) -> NgffBaseTransformation:
        input_cs = self._get_default_coordinate_system(axes=input_axes, unit=unit)
        output_cs = self._get_default_coordinate_system(
            axes=output_axes,
            unit=unit,
            name=output_coordinate_system_name,
            default_to_global=True,
        )
        ngff_transformation = NgffIdentity(input_coordinate_system=input_cs, output_coordinate_system=output_cs)
        return ngff_transformation

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Identity)


# Warning on MapAxis vs NgffMapAxis: MapAxis can add new axes that are not present in input. NgffMapAxis can't do
# this. It can only 1) permute the axis order, 2) eventually assiging the same axis to multiple output axes and 3)
# drop axes. When convering from MapAxis to NgffMapAxis this can be done by returing a Sequence of NgffAffine and
# NgffMapAxis, where the NgffAffine corrects the axes
class MapAxis(BaseTransformation):
    def __init__(self, map_axis: dict[ValidAxis_t, ValidAxis_t]) -> None:
        # to avoid circular imports
        from spatialdata.models import validate_axis_name

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
        self.validate_axes(input_axes)
        self.validate_axes(output_axes)
        # validation logic:
        # if an ax is in output_axes, then:
        #    if it is in self.keys, then the corresponding value must be in input_axes
        for ax in output_axes:
            if ax in self.map_axis and self.map_axis[ax] not in input_axes:
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
    def _from_ngff(cls, t: NgffBaseTransformation) -> BaseTransformation:
        assert isinstance(t, NgffMapAxis)
        return MapAxis(map_axis=t.map_axis)

    def to_ngff(
        self,
        input_axes: tuple[ValidAxis_t, ...],
        output_axes: tuple[ValidAxis_t, ...],
        unit: str | None = None,
        output_coordinate_system_name: str | None = None,
    ) -> NgffBaseTransformation:
        input_cs = self._get_default_coordinate_system(axes=input_axes, unit=unit)
        output_cs = self._get_default_coordinate_system(
            axes=output_axes,
            unit=unit,
            name=output_coordinate_system_name,
            default_to_global=True,
        )
        ngff_transformation = NgffMapAxis(
            input_coordinate_system=input_cs, output_coordinate_system=output_cs, map_axis=self.map_axis
        )
        return ngff_transformation

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, MapAxis) and self.map_axis == other.map_axis


class Translation(BaseTransformation):
    def __init__(self, translation: Union[list[Number], ArrayLike], axes: tuple[ValidAxis_t, ...]) -> None:
        from spatialdata._utils import _parse_list_into_array

        self.translation = _parse_list_into_array(translation)
        self.validate_axes(axes)
        self.axes = axes
        assert len(self.translation) == len(self.axes)

    def inverse(self) -> BaseTransformation:
        return Translation(-self.translation, self.axes)

    def to_affine_matrix(self, input_axes: tuple[ValidAxis_t, ...], output_axes: tuple[ValidAxis_t, ...]) -> ArrayLike:
        self.validate_axes(input_axes)
        self.validate_axes(output_axes)
        if not all(ax in output_axes for ax in input_axes):
            raise ValueError("Input axes must be a subset of output axes.")
        m = self._empty_affine_matrix(input_axes, output_axes)
        for i_out, ax_out in enumerate(output_axes):
            for i_in, ax_in in enumerate(input_axes):
                if ax_in == ax_out:
                    m[i_out, i_in] = 1
                    if ax_out in self.axes:
                        m[i_out, -1] = self.translation[self.axes.index(ax_out)]
        return m

    def to_translation_vector(self, axes: tuple[ValidAxis_t, ...]) -> ArrayLike:
        self.validate_axes(axes)
        v = []
        for ax in axes:
            if ax not in self.axes:
                v.append(0.0)
            else:
                i = self.axes.index(ax)
                v.append(self.translation[i])
        return np.array(v)

    def _repr_transformation_description(self, indent: int = 0) -> str:
        return f"({', '.join(self.axes)})\n{self._indent(indent)}{self.translation}"

    def _transform_coordinates(self, data: DataArray) -> DataArray:
        self._xarray_coords_validate_axes(data)
        output_axes = self._xarray_coords_get_coords(data)
        translation_adjusted = self.to_translation_vector(axes=output_axes)
        translation = DataArray(translation_adjusted, coords={"dim": list(output_axes)})
        transformed = data + translation
        to_return = self._xarray_coords_reorder_axes(transformed)
        return to_return

    @classmethod
    def _from_ngff(cls, t: NgffBaseTransformation) -> BaseTransformation:
        assert isinstance(t, NgffTranslation)
        assert t.input_coordinate_system is not None
        assert t.output_coordinate_system is not None
        input_axes = tuple(t.input_coordinate_system.axes_names)
        output_axes = tuple(t.output_coordinate_system.axes_names)
        assert input_axes == output_axes
        return Translation(translation=t.translation, axes=input_axes)

    def to_ngff(
        self,
        input_axes: tuple[ValidAxis_t, ...],
        output_axes: tuple[ValidAxis_t, ...],
        unit: str | None = None,
        output_coordinate_system_name: str | None = None,
    ) -> NgffBaseTransformation:
        input_cs = self._get_default_coordinate_system(axes=input_axes, unit=unit)
        output_cs = self._get_default_coordinate_system(
            axes=output_axes,
            unit=unit,
            name=output_coordinate_system_name,
            default_to_global=True,
        )
        new_translation_vector = self.to_translation_vector(axes=input_axes)
        ngff_transformation = NgffTranslation(
            input_coordinate_system=input_cs, output_coordinate_system=output_cs, translation=new_translation_vector
        )
        return ngff_transformation

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, Translation)
            and np.allclose(self.translation, other.translation)
            and self.axes == other.axes
        )


class Scale(BaseTransformation):
    def __init__(self, scale: Union[list[Number], ArrayLike], axes: tuple[ValidAxis_t, ...]) -> None:
        from spatialdata._utils import _parse_list_into_array

        self.scale = _parse_list_into_array(scale)
        self.validate_axes(axes)
        self.axes = axes
        assert len(self.scale) == len(self.axes)

    def inverse(self) -> BaseTransformation:
        return Scale(1 / self.scale, self.axes)

    def to_affine_matrix(self, input_axes: tuple[ValidAxis_t, ...], output_axes: tuple[ValidAxis_t, ...]) -> ArrayLike:
        self.validate_axes(input_axes)
        self.validate_axes(output_axes)
        if not all(ax in output_axes for ax in input_axes):
            raise ValueError("Input axes must be a subset of output axes.")
        m = self._empty_affine_matrix(input_axes, output_axes)
        for i_out, ax_out in enumerate(output_axes):
            for i_in, ax_in in enumerate(input_axes):
                if ax_in == ax_out:
                    scale_factor = self.scale[self.axes.index(ax_out)] if ax_out in self.axes else 1
                    m[i_out, i_in] = scale_factor
        return m

    def to_scale_vector(self, axes: tuple[ValidAxis_t, ...]) -> ArrayLike:
        self.validate_axes(axes)
        v = []
        for ax in axes:
            if ax not in self.axes:
                v.append(1.0)
            else:
                i = self.axes.index(ax)
                v.append(self.scale[i])
        return np.array(v)

    def _repr_transformation_description(self, indent: int = 0) -> str:
        return f"({', '.join(self.axes)})\n{self._indent(indent)}{self.scale}"

    def _transform_coordinates(self, data: DataArray) -> DataArray:
        self._xarray_coords_validate_axes(data)
        output_axes = self._xarray_coords_get_coords(data)
        scale_adjusted = self.to_scale_vector(axes=output_axes)
        scale = DataArray(scale_adjusted, coords={"dim": list(output_axes)})
        transformed = data * scale
        to_return = self._xarray_coords_reorder_axes(transformed)
        return to_return

    @classmethod
    def _from_ngff(cls, t: NgffBaseTransformation) -> BaseTransformation:
        assert isinstance(t, NgffScale)
        assert t.input_coordinate_system is not None
        assert t.output_coordinate_system is not None
        input_axes = tuple(t.input_coordinate_system.axes_names)
        output_axes = tuple(t.output_coordinate_system.axes_names)
        assert input_axes == output_axes
        return Scale(scale=t.scale, axes=input_axes)

    def to_ngff(
        self,
        input_axes: tuple[ValidAxis_t, ...],
        output_axes: tuple[ValidAxis_t, ...],
        unit: str | None = None,
        output_coordinate_system_name: str | None = None,
    ) -> NgffBaseTransformation:
        input_cs = self._get_default_coordinate_system(axes=input_axes, unit=unit)
        output_cs = self._get_default_coordinate_system(
            axes=output_axes, unit=unit, name=output_coordinate_system_name, default_to_global=True
        )
        new_scale_vector = self.to_scale_vector(input_axes)
        ngff_transformation = NgffScale(
            input_coordinate_system=input_cs, output_coordinate_system=output_cs, scale=new_scale_vector
        )
        return ngff_transformation

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Scale) and np.allclose(self.scale, other.scale) and self.axes == other.axes


class Affine(BaseTransformation):
    def __init__(
        self,
        matrix: Union[list[Number], ArrayLike],
        input_axes: tuple[ValidAxis_t, ...],
        output_axes: tuple[ValidAxis_t, ...],
    ) -> None:
        from spatialdata._utils import _parse_list_into_array

        self.validate_axes(input_axes)
        self.validate_axes(output_axes)
        self.input_axes = input_axes
        self.output_axes = output_axes
        self.matrix = _parse_list_into_array(matrix)
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
        self.validate_axes(input_axes)
        self.validate_axes(output_axes)
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
            warn(
                "Asking a representation of the affine transformation that is not using the matrix: "
                f"self.input_axews = {self.input_axes}, self.output_axes = {self.output_axes}, "
                f"input_axes = {input_axes}, output_axes = {output_axes}",
                UserWarning,
                stacklevel=2,
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
        to_return = DataArray(transformed, coords={"points": data.coords["points"], "dim": list(data_output_axes)})
        self._xarray_coords_filter_axes(to_return)
        to_return = self._xarray_coords_reorder_axes(to_return)
        return to_return

    @classmethod
    def _from_ngff(cls, t: NgffBaseTransformation) -> BaseTransformation:
        assert isinstance(t, NgffAffine)
        assert t.input_coordinate_system is not None
        assert t.output_coordinate_system is not None
        input_axes = tuple(t.input_coordinate_system.axes_names)
        output_axes = tuple(t.output_coordinate_system.axes_names)
        return Affine(matrix=t.affine, input_axes=input_axes, output_axes=output_axes)

    def to_ngff(
        self,
        input_axes: tuple[ValidAxis_t, ...],
        output_axes: tuple[ValidAxis_t, ...],
        unit: str | None = None,
        output_coordinate_system_name: str | None = None,
    ) -> NgffBaseTransformation:
        new_matrix = self.to_affine_matrix(input_axes, output_axes)
        input_cs = self._get_default_coordinate_system(axes=input_axes, unit=unit)
        output_cs = self._get_default_coordinate_system(
            axes=output_axes,
            unit=unit,
            name=output_coordinate_system_name,
            default_to_global=True,
        )
        ngff_transformation = NgffAffine(
            input_coordinate_system=input_cs, output_coordinate_system=output_cs, affine=new_matrix
        )
        return ngff_transformation

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Affine):
            return False
        return (
            np.allclose(self.matrix, other.matrix)
            and self.input_axes == other.input_axes
            and self.output_axes == other.output_axes
        )


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
        DEBUG_SEQUENCE = False
        self.validate_axes(input_axes)
        self.validate_axes(output_axes)
        if not all(ax in output_axes for ax in input_axes):
            raise ValueError("Input axes must be a subset of output axes.")

        current_input_axes = input_axes
        current_output_axes = _get_current_output_axes(self.transformations[0], current_input_axes)
        m = self.transformations[0].to_affine_matrix(current_input_axes, current_output_axes)
        if DEBUG_SEQUENCE:
            print(f"# 0: current_input_axes = {current_input_axes}, current_output_axes = {current_output_axes}")
            print(self.transformations[0])
            print()
        for i, t in enumerate(self.transformations[1:]):
            current_input_axes = current_output_axes
            current_output_axes = _get_current_output_axes(t, current_input_axes)
            if DEBUG_SEQUENCE:
                print(
                    f"# {i + 1}: current_input_axes = {current_input_axes}, current_output_axes = {current_output_axes}"
                )
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
            # # in the case of nested Sequence transformations, only the very last transformation in the outer sequence
            # # will force the output to be the one specified by the user. To identify the original call from the
            # # nested calls we use the _nested_sequence flag
            # if i == len(self.transformations) - 2 and not _nested_sequence:
            #     lhs = lhs[np.array([current_input_axes.index(ax) for ax in output_axes] + [-1]), :]
            #     current_output_axes = output_axes
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
        if current_output_axes != output_axes:
            reordered = []
            for ax in output_axes:
                if ax in current_output_axes:
                    i = current_output_axes.index(ax)
                    reordered.append(matrix[i, :])
                else:
                    reordered.append(np.zeros(matrix.shape[1]))
            reordered.append(matrix[-1, :])
            matrix = np.array(reordered)
            # assert set(current_output_axes) == set(output_axes)
            # we need to reorder the axes
            # reorder = [current_output_axes.index(ax) for ax in output_axes]
            # matrix = matrix[reorder + [-1], :]
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
    def _from_ngff(cls, t: NgffBaseTransformation) -> BaseTransformation:
        assert isinstance(t, NgffSequence)
        return Sequence(transformations=[BaseTransformation.from_ngff(t) for t in t.transformations])

    def to_ngff(
        self,
        input_axes: tuple[ValidAxis_t, ...],
        output_axes: tuple[ValidAxis_t, ...],
        unit: str | None = None,
        output_coordinate_system_name: str | None = None,
    ) -> NgffBaseTransformation:
        input_cs = self._get_default_coordinate_system(axes=input_axes, unit=unit)
        output_cs = self._get_default_coordinate_system(
            axes=output_axes,
            unit=unit,
            name=output_coordinate_system_name,
            default_to_global=True,
        )
        converted_transformations = []
        latest_input_axes = input_axes
        for t in self.transformations:
            latest_output_axes = _get_current_output_axes(t, latest_input_axes)
            converted_transformations.append(
                t.to_ngff(
                    input_axes=latest_input_axes,
                    output_axes=latest_output_axes,
                    # unit=unit,
                    # output_coordinate_system_name=output_coordinate_system_name,
                )
            )
            latest_input_axes = latest_output_axes
        ngff_transformation = NgffSequence(
            input_coordinate_system=input_cs,
            output_coordinate_system=output_cs,
            transformations=converted_transformations,
        )
        return ngff_transformation

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, Sequence):
            return False
        return self.transformations == other.transformations


def _get_current_output_axes(
    transformation: BaseTransformation, input_axes: tuple[ValidAxis_t, ...]
) -> tuple[ValidAxis_t, ...]:
    if isinstance(transformation, Identity | Translation | Scale):
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
                assert all(ax_out not in to_return for ax_out in mapped)
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
        for t in transformation.transformations:
            input_axes = _get_current_output_axes(t, input_axes)
        return input_axes
    else:
        raise ValueError("Unknown transformation type.")


def _get_affine_for_element(element: SpatialElement, transformation: BaseTransformation) -> Affine:
    from spatialdata.models import get_axes_names

    input_axes = get_axes_names(element)
    output_axes = _get_current_output_axes(transformation, input_axes)
    matrix = transformation.to_affine_matrix(input_axes=input_axes, output_axes=output_axes)
    return Affine(matrix, input_axes=input_axes, output_axes=output_axes)


def _decompose_affine_into_linear_and_translation(affine: Affine) -> tuple[Affine, Translation]:
    matrix = affine.matrix
    translation_part = matrix[:-1, -1]

    linear_part = np.zeros_like(matrix)
    linear_part[:-1, :-1] = matrix[:-1, :-1]
    linear_part[-1, -1] = 1

    linear_transformation = Affine(linear_part, input_axes=affine.input_axes, output_axes=affine.output_axes)
    translation_transformation = Translation(translation_part, axes=affine.output_axes)
    return linear_transformation, translation_transformation


def _compose_affine_from_linear_and_translation(
    linear: ArrayLike, translation: ArrayLike, input_axes: tuple[ValidAxis_t, ...], output_axes: tuple[ValidAxis_t, ...]
) -> Affine:
    matrix = np.zeros((linear.shape[0] + 1, linear.shape[1] + 1))
    matrix[:-1, :-1] = linear
    matrix[:-1, -1] = translation
    matrix[-1, -1] = 1
    return Affine(matrix, input_axes=input_axes, output_axes=output_axes)


def _decompose_transformation(
    transformation: BaseTransformation, input_axes: tuple[ValidAxis_t, ...], simple_decomposition: bool = True
) -> Sequence:
    """
    Decompose a given 2D transformation into a sequence of predetermined types of transformations.

    Parameters
    ----------
    transformation
        The transformation to decompose. It is assumed to be of a type that can be represented as a single affine
        transformation. It should leave the input axes unmodified, and it should not transform the c channel, if this
        is present.
    input_axes
        The axes of the data the transformation is to be applied to
    simple_decomposition
        If true, decomposes a transformation into it's linear part (affine without translation) and translation part,
        otherwise decomposes it into a sequence of reflection, rotation, shear, scale, translation.

    Returns
    -------
    sequence
        Returns a sequence of transformations (class :class:`~spatialdata.transformations.Sequence`) which operates only
        on the spatial part (no c channel). The output sequence will contain either 2 either 5 transformations in the
        following order (the first is applied first).
        Case `simple_decomposition = True`.

            1. Linear part (affine): linear part of the affine transformation, represented as a
            :class:`~spatialdata.transformations.Affine` transformation.
            2. Translation. Represented as a :class:`~spatialdata.transformations.Translation` transformation.

        Case `simple_decomposition = False`.

            1. Reflection. Represented as :class:`~spatialdata.transformations.Scale` transformation with elements in
                {1, -1}.
            2. Rotation. Represented as an :class:`~spatialdata.transformations.Affine` transformation which in its
                matrix form presents itself as an homogeneous affine matrix with no translation part and determinant 1.
                Please look at the source code of this function if you need to recover the angle theta.
            3. Shear. Represented as an :class:`~spatialdata.transformations.Affine` transformation which in its matrix
                form presents itself as an homogeneous affine matrix with no translation part. The matrix is upper
                triangular with diagonal elements all equal to 1.
            4. Scale. Represented as a :class:`~spatialdata.transformations.Scale` transformation with positive
            elements.
            5. Translation. Represented as a :class:`~spatialdata.transformations.Translation` transformation.

        Note that some of these transformations may be identity transformations.
    """
    output_axes = _get_current_output_axes(transformation=transformation, input_axes=input_axes)
    if input_axes != output_axes:
        raise ValueError("The transformation should leave the input axes unmodified.")
    if "z" in input_axes:
        raise ValueError("The transformation should not transform the z axis.")
    affine = transformation.to_affine(input_axes=input_axes, output_axes=output_axes)
    matrix = affine.matrix
    if "c" in input_axes:
        c_index = input_axes.index("c")
        if (
            matrix[c_index, c_index] != 1
            or np.linalg.norm(matrix[c_index, :]) != 1
            or np.linalg.norm(matrix[:, c_index]) != 1
        ):
            raise ValueError("The transformation should not transform the c channel.")
        axes = input_axes[:c_index] + input_axes[c_index + 1 :]
        m = np.delete(matrix, c_index, 0)
        m = np.delete(m, c_index, 1)
    else:
        axes = input_axes
        m = matrix

    translation_part = m[:-1, -1]
    linear_part = m[:-1, :-1]

    if simple_decomposition:
        translation = Translation(translation_part, axes=axes)
        linear = _compose_affine_from_linear_and_translation(
            linear=linear_part,
            translation=np.zeros(linear_part.shape[0]),
            input_axes=axes,
            output_axes=axes,
        )
        sequence = Sequence([linear, translation])
    else:
        # qr factorization
        a = linear_part
        r, q = scipy.linalg.rq(a)

        theta = np.arctan2(q[1, 0], q[0, 0])
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

        scale_matrix = np.diag(np.abs(np.diag(r)))
        shear_matrix = np.linalg.inv(scale_matrix) @ r
        assert np.allclose(scale_matrix @ shear_matrix, r)
        d = np.diag(np.diag(shear_matrix))

        qq = rotation_matrix.T @ q
        # check that qq is a diagonal matrix with diagonal values in {-1, 1}
        assert np.allclose(np.diag(qq) ** 2, np.ones(qq.shape[0]))
        assert np.isclose(np.sum(np.abs(qq.ravel())), qq.shape[0])
        assert np.allclose(rotation_matrix @ qq, q)

        adjusted_shear_matrix = shear_matrix @ d
        adjusted_rotation_matrix = d @ rotation_matrix @ d
        assert np.allclose(
            adjusted_rotation_matrix @ adjusted_rotation_matrix.T, np.eye(adjusted_rotation_matrix.shape[0])
        )
        adjusted_qq = d @ qq

        aaa = scale_matrix @ shear_matrix @ d @ d @ rotation_matrix @ d @ d @ qq
        assert np.allclose(a, aaa)
        aa = scale_matrix @ adjusted_shear_matrix @ adjusted_rotation_matrix @ adjusted_qq
        assert np.allclose(a, aa)

        scale = Scale(np.diag(scale_matrix), axes=axes)
        shear = _compose_affine_from_linear_and_translation(
            linear=adjusted_shear_matrix,
            translation=np.zeros(shear_matrix.shape[0]),
            input_axes=axes,
            output_axes=axes,
        )
        rotation = _compose_affine_from_linear_and_translation(
            linear=adjusted_rotation_matrix,
            translation=np.zeros(rotation_matrix.shape[0]),
            input_axes=axes,
            output_axes=axes,
        )
        inversion = Scale(np.diag(adjusted_qq), axes=axes)
        translation = Translation(translation_part, axes=axes)
        sequence = Sequence([inversion, rotation, shear, scale, translation])
    check_m = sequence.to_affine_matrix(input_axes=input_axes, output_axes=input_axes)
    assert np.allclose(check_m, matrix)
    return sequence


TRANSFORMATIONS_MAP[NgffIdentity] = Identity
TRANSFORMATIONS_MAP[NgffMapAxis] = MapAxis
TRANSFORMATIONS_MAP[NgffTranslation] = Translation
TRANSFORMATIONS_MAP[NgffScale] = Scale
TRANSFORMATIONS_MAP[NgffAffine] = Affine
TRANSFORMATIONS_MAP[NgffSequence] = Sequence
