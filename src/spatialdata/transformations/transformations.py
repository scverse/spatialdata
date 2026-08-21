from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any
from warnings import warn

import numpy as np
import scipy
import xarray as xr
from xarray import DataArray

from spatialdata._types import ArrayLike
from spatialdata.transformations.ngff.ngff_coordinate_system import NgffCoordinateSystem, _get_spatial_axes
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
        from spatialdata.transformations.ngff._utils import get_default_coordinate_system

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
        """
        Return the inverse of the transformation.

        Returns
        -------
        BaseTransformation
            A new transformation that is the inverse of this one, such that applying
            both in sequence yields the identity transformation.
        """
        pass

    # @abstractmethod
    # def transform_points(self, points: ArrayLike) -> ArrayLike:
    #     pass

    @abstractmethod
    def to_affine_matrix(self, input_axes: tuple[ValidAxis_t, ...], output_axes: tuple[ValidAxis_t, ...]) -> ArrayLike:
        """
        Return the affine matrix representation of the transformation.

        Parameters
        ----------
        input_axes
            The axes of the input coordinate system, e.g. ``("x", "y")`` or ``("c", "z", "y", "x")``.
        output_axes
            The axes of the output coordinate system.

        Returns
        -------
        ArrayLike
            A homogeneous affine matrix of shape ``(len(output_axes) + 1, len(input_axes) + 1)``.
            The last row is always ``[0, 0, ..., 1]`` (homogeneity).
        """
        pass

    def to_affine(self, input_axes: tuple[ValidAxis_t, ...], output_axes: tuple[ValidAxis_t, ...]) -> Affine:
        affine_matrix = self.to_affine_matrix(input_axes, output_axes)
        return Affine(affine_matrix, input_axes, output_axes)

    # order of the composition: self is applied first, then the transformation passed as argument
    def compose_with(self, transformations: BaseTransformation | list[BaseTransformation]) -> BaseTransformation:
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
    def _xarray_coords_filter_axes(data: DataArray, axes: tuple[ValidAxis_t, ...] | None = None) -> DataArray:
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
    """
    Transformation that maps input axes to output axes.

    Parameters
    ----------
    map_axis
        Dictionary with keys being the input axes and values the output axes.
    """

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
    def __init__(self, translation: list[Number] | ArrayLike, axes: tuple[ValidAxis_t, ...]) -> None:
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
    def __init__(self, scale: list[Number] | ArrayLike, axes: tuple[ValidAxis_t, ...]) -> None:
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
        matrix: list[Number] | ArrayLike,
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
        if not np.allclose(self.matrix[-1, :-1], np.zeros(len(input_axes))):
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


def _validate_square_affine_for_decomposition(
    transformation: BaseTransformation, input_axes: tuple[ValidAxis_t, ...]
) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    """
    Validate that a transformation can be decomposed, and extract the parts of its affine matrix.

    Parameters
    ----------
    transformation
        The transformation to decompose. It is assumed to be of a type that can be represented as a single affine
        transformation. It should leave the set of input axes unmodified (adding, dropping or renaming an axis is
        not allowed), but the axes are allowed to come out in a different order: the matrix is always queried back
        in ``input_axes`` order before being decomposed. There is no restriction on which axes are present: spatial
        axes (``x``, ``y``, ``z``) and the ``c`` channel axis are all decomposed uniformly, as the matrix is
        treated as a generic square affine.
    input_axes
        The axes of the data the transformation is to be applied to.

    Returns
    -------
    A tuple ``(matrix, translation_part, linear_part)`` where ``matrix`` is the full homogeneous affine matrix (with
    both rows and columns ordered as ``input_axes``), ``translation_part`` is its last column (excluding the
    homogeneous row), and ``linear_part`` is the square matrix obtained by removing the last row and column of
    ``matrix``.

    Raises
    ------
    ValueError
        If the transformation changes the set of input axes (as opposed to merely reordering them).
    RuntimeWarning
        If the linear part of the affine has a large condition number, in which case the decomposition may be
        numerically inaccurate.
    """
    output_axes = _get_current_output_axes(transformation=transformation, input_axes=input_axes)
    if set(input_axes) != set(output_axes):
        raise ValueError("The transformation should leave the set of input axes unmodified.")
    # the axes may come out in a different order than input_axes; querying in input_axes order makes the matrix
    # square with a consistent row/column labeling, which is what the decomposition below relies on
    affine = transformation.to_affine(input_axes=input_axes, output_axes=input_axes)
    matrix = affine.matrix
    translation_part = matrix[:-1, -1]
    linear_part = matrix[:-1, :-1]

    cond = np.linalg.cond(linear_part)
    if cond > 1e10:
        warn(
            f"The linear part of the affine has a large condition number ({cond:.2e}). "
            "The decomposition may be numerically inaccurate.",
            RuntimeWarning,
            stacklevel=2,
        )
    return matrix, translation_part, linear_part


def _decompose_transformation_simple(
    transformation: BaseTransformation, input_axes: tuple[ValidAxis_t, ...]
) -> tuple[Affine, Translation]:
    """
    Decompose a given transformation into its linear part and translation part.

    Parameters
    ----------
    transformation
        The transformation to decompose. See :func:`_validate_square_affine_for_decomposition`.
    input_axes
        The axes of the data the transformation is to be applied to.

    Returns
    -------
    A tuple ``(linear, translation)``, applied in this order (``linear`` first), whose composition equals
    ``transformation``.

        1. Linear part (affine): linear part of the affine transformation, represented as a
           :class:`~spatialdata.transformations.Affine` transformation.
        2. Translation. Represented as a :class:`~spatialdata.transformations.Translation` transformation.

    Note that some of these transformations may be identity transformations.
    """
    matrix, translation_part, linear_part = _validate_square_affine_for_decomposition(transformation, input_axes)

    linear = _compose_affine_from_linear_and_translation(
        linear=linear_part,
        translation=np.zeros(linear_part.shape[0]),
        input_axes=input_axes,
        output_axes=input_axes,
    )
    translation = Translation(translation_part, axes=input_axes)

    check_m = Sequence([linear, translation]).to_affine_matrix(input_axes=input_axes, output_axes=input_axes)
    assert np.allclose(check_m, matrix)
    return linear, translation


def _decompose_transformation_full(
    transformation: BaseTransformation, input_axes: tuple[ValidAxis_t, ...]
) -> tuple[Affine, Affine, Scale, Scale, Translation]:
    """
    Decompose a given transformation into rotation, shear, reflection, scale and translation.

    Parameters
    ----------
    transformation
        The transformation to decompose. See :func:`_validate_square_affine_for_decomposition`.
    input_axes
        The axes of the data the transformation is to be applied to.

    Returns
    -------
    A tuple ``(rotation, shear, reflection, scale, translation)``, applied in this order (``rotation`` first),
    whose composition equals ``transformation``.

        1. Rotation. Represented as an :class:`~spatialdata.transformations.Affine` transformation which in its
           matrix form presents itself as an homogeneous affine matrix with no translation part and determinant 1.
        2. Shear. Represented as an :class:`~spatialdata.transformations.Affine` transformation which in its matrix
           form presents itself as an homogeneous affine matrix with no translation part. The matrix is upper
           triangular with diagonal elements all equal to 1.
        3. Reflection. Represented as :class:`~spatialdata.transformations.Scale` transformation with elements in
           {1, -1}.
        4. Scale. Represented as a :class:`~spatialdata.transformations.Scale` transformation with positive
           elements.
        5. Translation. Represented as a :class:`~spatialdata.transformations.Translation` transformation.

    Note that some of these transformations may be identity transformations.

    Raises
    ------
    RuntimeError
        If the decomposition fails an internal consistency check (please report this as a bug).
    """
    matrix, translation_part, linear_part = _validate_square_affine_for_decomposition(transformation, input_axes)

    # RQ decomposition: linear_part = r @ q  (r upper-triangular, q orthogonal)
    r, q = scipy.linalg.rq(linear_part)

    # Ensure the diagonal of r is strictly positive.
    sign_diag = np.sign(np.diag(r))
    sign_diag[sign_diag == 0] = 1.0  # treat zero pivots as positive
    d = np.diag(sign_diag)
    r_pos = r @ d  # upper-triangular, positive diagonal
    q_adj = d @ q  # still orthogonal

    # Split r_pos into scale and shear.
    scale_values = np.diag(r_pos)  # all positive
    scale_matrix = np.diag(scale_values)
    shear_matrix = np.linalg.inv(scale_matrix) @ r_pos  # upper-tri, 1s on diag

    # Split q_adj into rotation (det = +1) and an axis-aligned reflection.
    # Reflection flips only the first axis when det(q_adj) = -1.
    det_sign = float(np.round(np.linalg.det(q_adj)))  # ±1
    reflection_values = np.ones(linear_part.shape[0])
    reflection_values[0] = det_sign
    reflection_matrix = np.diag(reflection_values)
    # q_adj = rotation_matrix @ reflection_matrix  ->  rotation_matrix = q_adj @ reflection_matrix
    rotation_matrix = q_adj @ reflection_matrix  # det = det_sign * det_sign = 1

    # Conjugate rotation and shear by the reflection so the sequence becomes
    # [rotation', shear', reflection, scale, translation]. This lets callers
    # bundle the reflection with either the shear or the scale.
    # rotation' = reflection @ rotation @ reflection  (still orthogonal, det = 1)
    # shear'    = reflection @ shear    @ reflection  (still upper-tri, 1s on diag)
    rotation_matrix_adj = reflection_matrix @ rotation_matrix @ reflection_matrix
    shear_matrix_adj = reflection_matrix @ shear_matrix @ reflection_matrix

    if not np.allclose(
        scale_matrix @ reflection_matrix @ shear_matrix_adj @ rotation_matrix_adj,
        linear_part,
    ):
        raise RuntimeError("Affine decomposition failed internal consistency check. Please report this bug.")

    rotation = _compose_affine_from_linear_and_translation(
        linear=rotation_matrix_adj,
        translation=np.zeros(rotation_matrix_adj.shape[0]),
        input_axes=input_axes,
        output_axes=input_axes,
    )
    shear = _compose_affine_from_linear_and_translation(
        linear=shear_matrix_adj,
        translation=np.zeros(shear_matrix_adj.shape[0]),
        input_axes=input_axes,
        output_axes=input_axes,
    )
    reflection = Scale(reflection_values, axes=input_axes)
    scale = Scale(scale_values, axes=input_axes)
    translation = Translation(translation_part, axes=input_axes)

    check_m = Sequence([rotation, shear, reflection, scale, translation]).to_affine_matrix(
        input_axes=input_axes, output_axes=input_axes
    )
    assert np.allclose(check_m, matrix)
    return rotation, shear, reflection, scale, translation


TRANSFORMATIONS_MAP[NgffIdentity] = Identity
TRANSFORMATIONS_MAP[NgffMapAxis] = MapAxis
TRANSFORMATIONS_MAP[NgffTranslation] = Translation
TRANSFORMATIONS_MAP[NgffScale] = Scale
TRANSFORMATIONS_MAP[NgffAffine] = Affine
TRANSFORMATIONS_MAP[NgffSequence] = Sequence
