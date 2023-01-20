from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from functools import singledispatchmethod
from numbers import Number
from typing import Any, Optional, Union

import dask.array
import dask_image.ndinterp
import numpy as np
import pyarrow as pa
import xarray as xr
from anndata import AnnData
from geopandas import GeoDataFrame
from multiscale_spatial_image import MultiscaleSpatialImage
from spatial_image import SpatialImage
from xarray import DataArray

from spatialdata import SpatialData
from spatialdata._core.models import ScaleFactors_t, get_schema
from spatialdata._core.core_utils import ValidAxis_t, get_dims, validate_axis_name
from spatialdata._logging import logger

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

DEBUG_WITH_PLOTS = True


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
    def _xarray_coords_get_coords(data: DataArray) -> list[ValidAxis_t]:
        axes = data.coords["dim"].data.tolist()
        assert isinstance(axes, list)
        return axes

    @staticmethod
    def _xarray_coords_get_column(data: DataArray, axis: ValidAxis_t) -> DataArray:
        return data[:, data["dim"] == axis]

    @staticmethod
    def _xarray_coords_validate_axes(data: DataArray) -> None:
        axes = BaseTransformation._xarray_coords_get_coords(data)
        if axes not in [["x", "y"], ["x", "y", "z"]]:
            raise ValueError(f"Invalid axes: {axes}")

    @staticmethod
    def _xarray_coords_filter_axes(data: DataArray, axes: Optional[list[ValidAxis_t]] = None) -> DataArray:
        if axes is None:
            axes = ["x", "y", "z"]
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
        if axes == ("c", "y", "x"):
            n_spatial_dims = 2
        elif axes == ("c", "z", "y", "x"):
            n_spatial_dims = 3
        else:
            raise ValueError(f"Invalid axes: {axes}")
        return n_spatial_dims

    def _transform_raster(self, data: DataArray, axes: tuple[str, ...]) -> DataArray:
        dims = {ch: axes.index(ch) for ch in axes}
        v_list = []
        n_spatial_dims = self._get_n_spatial_dims(axes)
        binary = np.array(list(itertools.product([0, 1], repeat=n_spatial_dims)))
        spatial_shape = data.shape[len(data.shape) - n_spatial_dims :]
        binary *= np.array(spatial_shape)
        v = np.hstack([np.zeros(len(binary)).reshape((-1, 1)), binary, np.ones(len(binary)).reshape((-1, 1))])
        matrix = self.to_affine_matrix(input_axes=axes, output_axes=axes)
        inverse_matrix = self.inverse().to_affine_matrix(input_axes=axes, output_axes=axes)
        new_v = (matrix @ v.T).T
        if "c" in axes:
            c_shape = (data.shape[0],)
        else:
            c_shape = tuple()
        new_spatial_shape = tuple(
            int(np.max(new_v[:, i]) - np.min(new_v[:, i])) for i in range(len(c_shape), n_spatial_dims + len(c_shape))
        )
        output_shape = c_shape + new_spatial_shape
        ##
        translation_vector = np.min(new_v[:, :-1], axis=0)
        inverse_matrix_adjusted = Sequence(
            [
                Translation(translation_vector, axes=axes),
                self.inverse(),
            ]
        ).to_affine_matrix(input_axes=axes, output_axes=axes)

        # fix chunk shape, it should be possible for the user to specify them, and by default we could reuse the chunk shape of the input
        # output_chunks = data.chunks
        ##
        transformed_dask = dask_image.ndinterp.affine_transform(
            data,
            matrix=inverse_matrix_adjusted,
            output_shape=output_shape
            # , output_chunks=output_chunks
        )
        ##

        if DEBUG_WITH_PLOTS:
            if n_spatial_dims == 2:
                ##
                import matplotlib.pyplot as plt

                plt.figure()
                im = data
                new_v_inverse = (inverse_matrix @ v.T).T
                min_x_inverse = np.min(new_v_inverse[:, 2])
                min_y_inverse = np.min(new_v_inverse[:, 1])

                plt.imshow(dask.array.moveaxis(transformed_dask, 0, 2), origin="lower")
                plt.imshow(dask.array.moveaxis(im, 0, 2), origin="lower")
                plt.scatter(v[:, 1:-1][:, 1] - 0.5, v[:, 1:-1][:, 0] - 0.5, c="r")
                plt.scatter(new_v[:, 1:-1][:, 1] - 0.5, new_v[:, 1:-1][:, 0] - 0.5, c="g")
                plt.scatter(new_v_inverse[:, 1:-1][:, 1] - 0.5, new_v_inverse[:, 1:-1][:, 0] - 0.5, c="k")
                plt.show()
                ##
            else:
                assert n_spatial_dims == 3
                raise NotImplementedError()
        return transformed_dask

    @singledispatchmethod
    def transform(self, data: Any) -> Any:
        raise NotImplementedError()

    @transform.register(SpatialData)
    def _(
        self,
        data: SpatialData,
    ) -> SpatialData:
        new_elements: dict[str, dict[str, Any]] = {}
        for element_type in ["images", "labels", "points", "polygons", "shapes"]:
            d = getattr(data, element_type)
            if len(d) > 0:
                new_elements[element_type] = {}
            for k, v in d.items():
                new_elements[element_type][k] = self.transform(v)

        new_sdata = SpatialData(**new_elements)
        return new_sdata

    @transform.register(SpatialImage)
    def _(
        self,
        data: SpatialImage,
    ) -> SpatialImage:
        axes = get_dims(data)
        transformed_dask = self._transform_raster(data.data, axes=axes)
        transformed_data = SpatialImage(transformed_dask, dims=axes)
        schema = get_schema(data)
        schema.parse(transformed_data)
        print('TODO: compose the transformation!!!! we need to put the previous one concatenated with the translation showen above. The translation operates before the other transformation')
        return transformed_data

    @transform.register(MultiscaleSpatialImage)
    def _(
        self,
        data: MultiscaleSpatialImage,
    ) -> MultiscaleSpatialImage:
        axes = get_dims(data)
        transformed_dask = self._transform_raster(data.data, axes=axes)
        transformed_data = MultiscaleSpatialImage(transformed_dask, dims=axes)
        schema = get_schema(data)
        schema.parse(transformed_data)
        print('TODO: compose the transformation!!!! we need to put the previous one concatenated with the translation showen above. The translation operates before the other transformation')
        return transformed_data

    @transform.register(pa.Table)
    def _(
        self,
        data: pa.Table,
    ) -> pa.Table:
        axes = get_dims(data)
        arrays = []
        for ax in axes:
            arrays.append(data[ax].to_numpy())
        xdata = DataArray(np.array(arrays).T, coords={"points": range(len(data)), "dim": list(axes)})
        xtransformed = self._transform_coordinates(xdata)
        transformed = data.drop(axes)
        for ax in axes:
            indices = xtransformed["dim"] == ax
            new_ax = pa.array(xtransformed[:, indices].data.flatten())
            transformed = transformed.append_column(ax, new_ax)

        # to avoid cyclic import
        from spatialdata._core.models import PointsModel

        PointsModel.validate(transformed)
        return transformed

    @transform.register(GeoDataFrame)
    def _(
        self,
        data: GeoDataFrame,
    ) -> GeoDataFrame:
        ##
        ndim = len(get_dims(data))
        # TODO: nitpick, mypy expects a listof literals and here we have a list of strings. I ignored but we may want to fix this
        matrix = self.to_affine_matrix(["x", "y", "z"][:ndim], ["x", "y", "z"][:ndim])  # type: ignore[arg-type]
        shapely_notation = matrix[:-1, :-1].ravel().tolist() + matrix[:-1, -1].tolist()
        transformed_geometry = data.geometry.affine_transform(shapely_notation)
        transformed_data = data.copy(deep=True)
        transformed_data.geometry = transformed_geometry

        # to avoid cyclic import
        from spatialdata._core.models import PolygonsModel

        PolygonsModel.validate(transformed_data)
        return transformed_data

    @transform.register(AnnData)
    def _(
        self,
        data: AnnData,
    ) -> AnnData:
        ndim = len(get_dims(data))
        xdata = DataArray(data.obsm["spatial"], coords={"points": range(len(data)), "dim": ["x", "y", "z"][:ndim]})
        transformed_spatial = self._transform_coordinates(xdata)
        transformed_adata = data.copy()
        transformed_adata.obsm["spatial"] = transformed_spatial.data

        # to avoid cyclic import
        from spatialdata._core.models import ShapesModel

        ShapesModel.validate(transformed_adata)
        return transformed_adata


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

    def _repr_transformation_description(self, indent: int = 0) -> str:
        return ""

    def _transform_coordinates(self, data: DataArray) -> DataArray:
        return data


# Warning on MapAxis vs NgffMapAxis: MapAxis can add new axes that are not present in input. NgffMapAxis can't do
# this. It can only 1) permute the axis order, 2) eventually assiging the same axis to multiple output axes and 3)
# drop axes. When convering from MapAxis to NgffMapAxis this can be done by returing a Sequence of NgffAffine and
# NgffMapAxis, where the NgffAffine corrects the axes
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


class Translation(BaseTransformation):
    def __init__(self, translation: Union[list[Number], ArrayLike], axes: list[ValidAxis_t]) -> None:
        self.translation = self._parse_list_into_array(translation)
        self._validate_axes(axes)
        self.axes = list(axes)
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
        return m

    def _repr_transformation_description(self, indent: int = 0) -> str:
        return f"({', '.join(self.axes)})\n{self._indent(indent)}{self.translation}"

    def _transform_coordinates(self, data: DataArray) -> DataArray:
        self._xarray_coords_validate_axes(data)
        translation = DataArray(self.translation, coords={"dim": self.axes})
        transformed = data + translation
        to_return = self._xarray_coords_reorder_axes(transformed)
        return to_return


class Scale(BaseTransformation):
    def __init__(self, scale: Union[list[Number], ArrayLike], axes: list[ValidAxis_t]) -> None:
        self.scale = self._parse_list_into_array(scale)
        self._validate_axes(axes)
        self.axes = list(axes)
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
        return m

    def _repr_transformation_description(self, indent: int = 0) -> str:
        return f"({', '.join(self.axes)})\n{self._indent(indent)}{self.scale}"

    def _transform_coordinates(self, data: DataArray) -> DataArray:
        self._xarray_coords_validate_axes(data)
        scale = DataArray(self.scale, coords={"dim": self.axes})
        transformed = data * scale
        to_return = self._xarray_coords_reorder_axes(transformed)
        return to_return


class Affine(BaseTransformation):
    def __init__(
        self, matrix: Union[list[Number], ArrayLike], input_axes: list[ValidAxis_t], output_axes: list[ValidAxis_t]
    ) -> None:
        self._validate_axes(input_axes)
        self._validate_axes(output_axes)
        self.input_axes = list(input_axes)
        self.output_axes = list(output_axes)
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

    def to_affine_matrix(self, input_axes: list[ValidAxis_t], output_axes: list[ValidAxis_t]) -> ArrayLike:
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


class Sequence(BaseTransformation):
    def __init__(self, transformations: list[BaseTransformation]) -> None:
        self.transformations = transformations

    def inverse(self) -> BaseTransformation:
        return Sequence([t.inverse() for t in self.transformations[::-1]])

    # this wrapper is used since we want to return just the affine matrix from to_affine_matrix(), but we need to
    # return two values for the recursive logic to work
    def _to_affine_matrix_wrapper(
        self, input_axes: list[ValidAxis_t], output_axes: list[ValidAxis_t], _nested_sequence: bool = False
    ) -> tuple[ArrayLike, list[ValidAxis_t]]:
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
        self, input_axes: list[ValidAxis_t], output_axes: list[ValidAxis_t], _nested_sequence: bool = False
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


def _get_current_output_axes(transformation: BaseTransformation, input_axes: list[ValidAxis_t]) -> list[ValidAxis_t]:
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
        return to_return
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
        return to_return
    elif isinstance(transformation, Sequence):
        return input_axes
    else:
        raise ValueError("Unknown transformation type.")
