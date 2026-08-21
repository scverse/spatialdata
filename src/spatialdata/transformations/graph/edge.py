# pyright: strict

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Final

import numpy as np
import ome_zarr_models.v06.coordinate_transforms as ozm06trans
import pydantic as pyd
import xarray as xr

from spatialdata._types import ArrayLike
from spatialdata.transformations.graph.vert import Axis, CoordSystem


class GarbledInput(Exception):
    def __init__(self, message: str, input: pyd.JsonValue) -> None:
        import json

        super().__init__(message + "\n" + json.dumps(input, indent=4))
        self.input = input


class BaseTransfEdge(ABC):
    """Base class for all the transformations defined by the NGFF specification."""

    input: Final[CoordSystem]
    output: Final[CoordSystem]
    name: str | None

    def __init__(
        self,
        name: str | None,
        *,
        input: CoordSystem,
        output: CoordSystem,
    ) -> None:
        self.input = input
        self.output = output
        self.name = name
        super().__init__()

    def __repr__(self) -> str:
        domain = ", ".join(self.input.axes_names)
        codomain = ", ".join(self.output.axes_names)
        return f"{type(self).__name__} ({domain} -> {codomain})"

    @abstractmethod
    def inverse(self) -> BaseTransfEdge:
        """Return the inverse of the transformation."""

    @abstractmethod
    def transform_points(self, points: ArrayLike) -> ArrayLike:
        """
        Transform points (coordinates).

        Notes
        -------
        This function will check if the dimensionality of the input and output coordinate systems of the
        transformation are compatible with the given points.
        """

    @abstractmethod
    def to_affine(self) -> AffineEdge:
        """Convert the transformation to an affine transformation, whenever the conversion can be made."""

    def _validate_transform_points_shapes(self, points: xr.DataArray | xr.DataTree | ArrayLike) -> None:
        """
        Validate if the shape of the points (coordinats to be transformed) are consistent with the input size of the
        transformation.
        """
        input_size = len(self.input.axes)
        if len(points.shape) != 2 or points.shape[1] != input_size:
            raise ValueError(
                f"points must be a tensor of shape (n, d), where n is the number of points and d is the "
                f"the number of spatial dimensions. Points shape: {points.shape}, input size: {input_size}"
            )

    # order of the composition: self is applied first, then the transformation passed as argument
    def compose_with(self, transformation: BaseTransfEdge) -> BaseTransfEdge:
        """
        Compose the transfomation object with another transformation

        Parameters
        ----------
        transformation
            The transformation to compose with.

        Returns
        -------
        The compoesed transformation.

        Notes
        -------
        Self is applied first, then the transformation passed as argument.
        """
        return SequenceEdge([self, transformation], name=None)  # FIXME: no name?

    @abstractmethod
    def to_model(self) -> ozm06trans.AnyTransform:
        pass


class AffineEdge(BaseTransfEdge):
    """The Affine transformation from the NGFF specification."""

    linear: Final[ArrayLike]
    translation: Final[ArrayLike]
    affine: Final[ArrayLike]

    def __init__(
        self,
        name: str | None,
        *,
        linear: ArrayLike,
        translation: ArrayLike | None = None,
        input: CoordSystem,
        output: CoordSystem,
    ) -> None:
        """
        Parameters
        ----------
        name
            A human readable name for this transformation
        linear
            The linear part of this transformation, i.e., the one that keeps
            the origin in the same place. Shape must be (output.num_axes, input.num_axes)
        translation y
            The translation part of this transformation, of shape (output.num_axes,)
        input
            Input coordinate system of the transformation.
        output
            Output coordinate system of the transformation.
        """
        num_inputs = input.num_axes
        num_outputs = output.num_axes
        translation = np.zeros(num_outputs) if translation is None else translation

        expected_linear_shape = (num_outputs, num_inputs)
        if linear.shape != expected_linear_shape:
            raise ValueError(f"linear's shape is {linear.shape}. Expected f{(num_outputs, num_inputs)}")
        expected_translation_shape = (num_outputs,)
        if translation.shape != expected_translation_shape:
            raise ValueError(f"translation's shape is {translation.shape}. Expected {expected_translation_shape}")

        self.linear = linear
        self.translation = translation

        self.affine = np.zeros((num_outputs + 1, num_inputs + 1))
        self.affine[:-1, :-1] = self.linear
        self.affine[:-1, -1] = self.translation
        self.affine[-1, -1] = 1

        super().__init__(input=input, output=output, name=name)

    def __repr__(self) -> str:
        s = super().__repr__() + "\n"
        s += "\n".join(str(row) for row in self.affine)
        return s

    @classmethod
    def from_affine_matrix(
        cls,
        *,
        name: str | None,
        affine_matrix: ArrayLike,
        input: CoordSystem,
        output: CoordSystem,
    ) -> AffineEdge:
        """Creates an AffineEdge from a raw affine matrix

        Parameters
        ----------
        name
            A human readable name for this transformation
        affine_matrix
            row-major, (output.num_axes + 1, input.num_axes + 1) matrix with:
                - linear part at the top left
                - translation as the rightmost column
                - last row is [0, 0, ..., 0, 1]
        input
            Input coordinate system of the transformation.
        output
            Output coordinate system of the transformation.
        """
        return AffineEdge(
            linear=affine_matrix[:-1, :-1],
            translation=affine_matrix[-1, :-1],
            input=input,
            output=output,
            name=name,
        )

    def inverse(self) -> BaseTransfEdge:
        inv = np.linalg.inv(self.affine)
        return AffineEdge(
            linear=inv[:-1, :-1],
            translation=inv[-1, :-1],
            input=self.output,
            output=self.input,
            name=self.name and f"{self.name}__affine",
        )

    def transform_points(self, points: ArrayLike) -> ArrayLike:
        self._validate_transform_points_shapes(points)
        p = np.vstack([points.T, np.ones(points.shape[0])])
        q = self.affine @ p
        res = q[: self.output.num_axes, :].T
        assert isinstance(res, np.ndarray)
        return res

    def to_affine(self) -> AffineEdge:
        return self

    def to_model(self) -> ozm06trans.Affine:
        return ozm06trans.Affine(
            name=self.name,
            affine=tuple(tuple(row) for row in self.affine[:-1, :]),
            input=self.input.to_model_cs_ident(),
            output=self.output.to_model_cs_ident(),
        )


class IdentityEdge(BaseTransfEdge):
    """The Identity transformation from the NGFF specification."""

    def __init__(
        self,
        name: str | None,
        *,
        input: CoordSystem,
        output: CoordSystem,
    ) -> None:
        """
        Parameters
        ----------
        name
            A human readable name for this transformation
        input
            Input coordinate system of the transformation.
        output
            Output coordinate system of the transformation.
        """
        if input.num_axes != output.num_axes:
            raise ValueError("Input and output must have the same number of dimensions")
        super().__init__(input=input, output=output, name=name)

    def inverse(self) -> BaseTransfEdge:
        return IdentityEdge(input=self.output, output=self.input, name=self.name and f"{self.name}__inverse")

    def transform_points(self, points: ArrayLike) -> ArrayLike:
        self._validate_transform_points_shapes(points)
        return points

    def to_affine(self) -> AffineEdge:
        return AffineEdge(
            linear=np.eye(self.input.num_axes),
            input=self.input,
            output=self.output,
            name=self.name and f"{self.name}__affine",
        )

    def to_model(self) -> ozm06trans.Identity:
        return ozm06trans.Identity(
            name=self.name,
            input=self.input.to_model_cs_ident(),
            output=self.output.to_model_cs_ident(),
        )


class MapAxisEdge(BaseTransfEdge):
    """The MapAxis transformation from the NGFF specification."""

    def __init__(
        self,
        name: str | None,
        *,
        output_to_input: dict[str, str],
        input: CoordSystem,
        output: CoordSystem,
    ) -> None:
        """
        Init the NgffMapAxis object.
        Parameters
        ----------
        name
            A human readable name for this transformation
        output_to_input
            A dictionary mapping the output axes (keys) to the input axes (values).
        input
            Input coordinate system of the transformation.
        output
            Output coordinate system of the transformation.
        """
        for out_ax, inp_ax in output_to_input.items():
            if not input.has_axis(inp_ax):
                raise ValueError(f"input has no axis named {inp_ax}")
            if not output.has_axis(out_ax):
                raise ValueError(f"output has no axis named {out_ax}")
        if not (len(output_to_input) == output.num_axes == input.num_axes):
            raise ValueError("input_to_output, input and output must have the same number of axes entries")
        if len(set(output_to_input.values())) != len(output_to_input):
            raise ValueError("input_to_output must map unique inputs to unique outputs")

        self.output_to_input = output_to_input
        super().__init__(input=input, output=output, name=name)

    def __repr__(self) -> str:
        s = super().__repr__() + "\n"
        s += "\n".join(f"    {out} <- {inp}\n" for out, inp in self.output_to_input.items())
        return s

    def inverse(self) -> BaseTransfEdge:
        return MapAxisEdge(
            output_to_input={v: k for k, v in self.output_to_input.items()},
            input=self.output,
            output=self.input,
            name=self.name and f"{self.name}__inverse",
        )

    def transform_points(self, points: ArrayLike) -> ArrayLike:
        input_axes = self.input.axes_names
        output_axes = self.output.axes_names
        self._validate_transform_points_shapes(points)
        new_indices = [input_axes.index(self.output_to_input[ax]) for ax in output_axes]
        mapped = points[:, new_indices]
        assert isinstance(mapped, np.ndarray)
        return mapped

    def to_affine(self) -> AffineEdge:
        input_axes = self.input.axes_names
        output_axes = self.output.axes_names
        linear: ArrayLike = np.zeros((len(output_axes), len(input_axes)), dtype=float)
        for i, des_axis in enumerate(output_axes):
            for j, src_axis in enumerate(input_axes):
                if src_axis == self.output_to_input[des_axis]:
                    linear[i, j] = 1
        affine = AffineEdge(
            linear=linear, input=self.input, output=self.output, name=self.name and f"{self.name}__affine"
        )
        return affine

    def to_model(self) -> ozm06trans.MapAxis:
        mapAxis: list[int] = []
        for out_ax in self.output.axes_names:
            in_ax = self.output_to_input[out_ax]
            in_idx = self.input.axes_names.index(in_ax)
            mapAxis.append(in_idx)
        return ozm06trans.MapAxis(
            name=self.name,
            mapAxis=tuple(mapAxis),
            input=self.input.to_model_cs_ident(),
            output=self.output.to_model_cs_ident(),
        )


class TranslationEdge(BaseTransfEdge):
    """The Translation transformation from the NGFF specification."""

    def __init__(
        self,
        name: str | None,
        *,
        translation: ArrayLike,
        input: CoordSystem,
        output: CoordSystem,
    ) -> None:
        """
        Init the NgffTranslation object.
        Parameters
        ----------
        name
            A human readable name for this transformation
        translation
            A vector of shape (input.num_axes,) specifying the translation along each axis.
        input
            Input coordinate system of the transformation.
        output
            Output coordinate system of the transformation.
        """
        if input.num_axes != output.num_axes:
            raise ValueError("Number of input and output axes must be the same")
        self.translation = translation
        super().__init__(input=input, output=output, name=name)

    def __repr__(self) -> str:
        return super().__repr__() + str(self.translation)

    def inverse(self) -> BaseTransfEdge:
        return TranslationEdge(
            translation=-self.translation,
            input=self.output,
            output=self.input,
            name=self.name and f"{self.name}__inverse",
        )

    def transform_points(self, points: ArrayLike) -> ArrayLike:
        self._validate_transform_points_shapes(points)
        return points + self.translation

    def to_affine(self) -> AffineEdge:
        return AffineEdge(
            linear=np.identity(self.input.num_axes),
            translation=self.translation,
            input=self.input,
            output=self.output,
            name=self.name and f"{self.name}__affine",
        )

    def to_model(self) -> ozm06trans.Translation:
        return ozm06trans.Translation(
            name=self.name,
            input=self.input.to_model_cs_ident(),
            output=self.output.to_model_cs_ident(),
            translation=tuple(self.translation),
        )


class ScaleEdge(BaseTransfEdge):
    """The Scale transformation from the NGFF specification."""

    def __init__(
        self,
        name: str | None,
        *,
        scale: ArrayLike,
        input: CoordSystem,
        output: CoordSystem,
    ) -> None:
        """
        Init the NgffScale object.
        Parameters
        ----------
        scale
            A list of numbers or a vector specifying the scale along each axis.
        input
            Input coordinate system of the transformation.
        output
            Output coordinate system of the transformation.
        """
        if scale.shape != (input.num_axes,):
            raise ValueError(f"scale should be of shape f{(input.num_axes,)}")
        if input.num_axes != output.num_axes:
            raise ValueError("input and output must have same number of dimensions")
        self.scale = scale
        super().__init__(input=input, output=output, name=name)

    def __repr__(self) -> str:
        return super().__repr__() + str(self.scale)

    def inverse(self) -> ScaleEdge:
        if any(s == 0 for s in self.scale):
            raise ValueError(f"Scaling {self} is not invertible")
        new_scale = 1 / self.scale
        return ScaleEdge(
            scale=new_scale, input=self.output, output=self.input, name=self.name and f"{self.name}__inverse"
        )

    def transform_points(self, points: ArrayLike) -> ArrayLike:
        self._validate_transform_points_shapes(points)
        return points * self.scale

    def to_affine(self) -> AffineEdge:
        return AffineEdge(
            linear=np.diag(self.scale), input=self.input, output=self.output, name=self.name and f"{self.name}__affine"
        )

    def to_model(self) -> ozm06trans.Scale:
        return ozm06trans.Scale(
            name=self.name,
            input=self.input.to_model_cs_ident(),
            output=self.output.to_model_cs_ident(),
            scale=tuple(self.scale),
        )


class RotationEdge(BaseTransfEdge):
    """The Rotation transformation from the NGFF specification."""

    rotation: Final[ArrayLike]

    def __init__(
        self,
        name: str | None,
        *,
        linear_matrix: ArrayLike,
        input: CoordSystem,
        output: CoordSystem,
    ) -> None:
        """
        Init the NgffRotation object.
        Parameters
        ----------
        linear_matrix
            an array of shape (output.num_axes, input.num_axes) representing the rotation
        input
            Input coordinate system of the transformation.
        output
            Output coordinate system of the transformation.
        """
        expected_shape = (output.num_axes, input.num_axes)
        if linear_matrix.shape != expected_shape:
            raise ValueError(f"linear matrix should have shape {expected_shape}")
        if input.num_axes != output.num_axes:
            raise ValueError("input and output should have the same numbe rof axes")
        if not np.isclose(np.linalg.det(linear_matrix), 1.0):
            raise ValueError("det(linear_matrix) should be ~= 1")
        linear_matrix.flags.writeable = False
        self.rotation = linear_matrix
        super().__init__(input=input, output=output, name=name)

    def __repr__(self) -> str:
        s = super().__repr__() + "\n"
        s += "\n".join(str(row) for row in self.rotation)
        return s

    def inverse(self) -> BaseTransfEdge:
        return RotationEdge(
            linear_matrix=self.rotation.T,
            input=self.output,
            output=self.input,
            name=self.name and f"{self.name}__inverse",
        )

    def transform_points(self, points: ArrayLike) -> ArrayLike:
        self._validate_transform_points_shapes(points)
        res = (self.rotation @ points.T).T
        assert isinstance(res, np.ndarray)
        return res

    def to_affine(self) -> AffineEdge:
        return AffineEdge(
            linear=self.rotation, input=self.input, output=self.output, name=self.name and f"{self.name}__affine"
        )

    def to_model(self) -> ozm06trans.Rotation:
        return ozm06trans.Rotation(
            name=self.name,
            input=self.input.to_model_cs_ident(),
            output=self.output.to_model_cs_ident(),
            rotation=tuple(tuple(row) for row in self.rotation),
        )


class SequenceEdge(BaseTransfEdge):
    """The Sequence transformation from the NGFF specification."""

    def __init__(
        self,
        transformations: Sequence[BaseTransfEdge],
        name: str | None,
    ) -> None:
        """
        Init the NgffSequence object.

        Parameters
        ----------
        transformations
            The transformations which compose the sequence.
        """
        if len(transformations) == 0:
            raise ValueError("Empty transformation list")
        previous_transf = transformations[0]
        for current_transf in transformations[1:]:
            if previous_transf.output != current_transf.input:
                raise ValueError(f"Mismatched input/output from {previous_transf} to {current_transf}")
            previous_transf = current_transf
        self.transformations = transformations
        super().__init__(
            input=transformations[0].input,
            output=transformations[-1].output,
            name=name,
        )

    def __repr__(self) -> str:
        from textwrap import indent

        out = super().__repr__() + " [\n"
        for t in self.transformations:
            out += indent(repr(t), prefix="    ") + "\n"
        out += "]"
        return out

    def inverse(self) -> SequenceEdge:
        return SequenceEdge(
            [t.inverse() for t in reversed(self.transformations)], name=self.name and f"{self.name}__inverse"
        )

    def to_affine(self) -> AffineEdge:
        composed = self.transformations[0].to_affine().affine
        for t in self.transformations[1:]:
            a = t.to_affine()
            composed = a.affine @ composed
        return AffineEdge.from_affine_matrix(
            affine_matrix=composed, input=self.input, output=self.output, name=self.name and f"{self.name}__affine"
        )

    def transform_points(self, points: ArrayLike) -> ArrayLike:
        return self.to_affine().transform_points(points)  # FIXME

    def to_model(self) -> ozm06trans.Sequence:
        return ozm06trans.Sequence(
            name=self.name,
            input=self.input.to_model_cs_ident(),
            output=self.output.to_model_cs_ident(),
            transformations=tuple(t.to_model() for t in self.transformations),
        )


class ByDimensionEdge(BaseTransfEdge):
    """The ByDimension transformation from the NGFF specification."""

    transformations: Final[Sequence[BaseTransfEdge]]

    def __init__(
        self,
        name: str | None,
        *,
        transformations: Sequence[BaseTransfEdge],
        input: CoordSystem,
        output: CoordSystem,
    ) -> None:
        """
        Init the ByDimension object.

        Parameters
        ----------
        transformations
            A list of transformations, whose set of output coordinate systems partition the output coordinate system of
            the ByDimension transformation.
        input
            The input coordinate system of the transformation.
        output
            The output coordinate system of the transformation.
        """
        # we check that:
        # 1. each input from each transformation in self.transformation must appear in the set of input axes
        # 2. each output from each transformation in self.transformation must appear at most once in the set of output
        # axes
        input_axes = input.axes_names
        output_axes = output.axes_names
        defined_output_axes: set[str] = set()
        for t in transformations:
            for ax in t.input.axes_names:
                if ax not in input_axes:
                    raise ValueError(f"By dimension axis {ax} not in {input_axes}")
            for ax in t.output.axes_names:
                if ax not in output_axes:
                    raise ValueError(f"Axis {ax} not in output axes {output_axes}")
                if ax in defined_output_axes:
                    raise ValueError(f"Output axis {ax} is defined more than once")
                defined_output_axes.add(ax)
        if len(output_axes) != len(defined_output_axes):
            raise ValueError("Not all outputs are mapped")

        self.transformations = tuple(transformations)
        super().__init__(input=input, output=output, name=name)

    def __repr__(self) -> str:
        from textwrap import indent

        out = super().__repr__() + " [\n"
        for t in self.transformations:
            out += indent(repr(t), prefix="    ") + "\n"
        out += "]"
        return out

    def inverse(self) -> BaseTransfEdge:
        inverse_transformations = [t.inverse() for t in self.transformations]
        return ByDimensionEdge(
            transformations=inverse_transformations,
            input=self.output,
            output=self.input,
            name=self.name and f"{self.name}__inverse",
        )

    def transform_points(self, points: ArrayLike) -> ArrayLike:
        input_axes = self.input.axes_names
        output_axes = self.output.axes_names
        self._validate_transform_points_shapes(points)
        output_columns: dict[str, ArrayLike] = {}
        for t in self.transformations:
            input_columns = [points[:, input_axes.index(ax)] for ax in t.input.axes_names]
            input_columns_stacked: ArrayLike = np.stack(input_columns, axis=1)
            output_columns_t = t.transform_points(input_columns_stacked)
            for ax, col in zip(t.output.axes_names, output_columns_t.T, strict=True):
                output_columns[ax] = col  # type: ignore[assignment]
        output: ArrayLike = np.stack([output_columns[ax] for ax in output_axes], axis=1)
        return output

    def to_affine(self) -> AffineEdge:
        input_axes = self.input.axes_names
        output_axes = self.output.axes_names
        m = np.zeros((len(output_axes) + 1, len(input_axes) + 1))
        m[-1, -1] = 1
        for t in self.transformations:
            t_affine = t.to_affine()
            target_output_indices = [output_axes.index(ax) for ax in t.output.axes_names if ax in output_axes]
            source_output_indices = [t.output.axes_names.index(ax) for ax in t.output.axes_names]
            target_input_indices = [input_axes.index(ax) for ax in t.input.axes_names] + [-1]
            m[np.ix_(target_output_indices, target_input_indices)] = t_affine.affine[source_output_indices, :]
        return AffineEdge.from_affine_matrix(
            affine_matrix=m, input=self.input, output=self.output, name=self.name and f"{self.name}__affine"
        )

    def to_model(self) -> ozm06trans.ByDimension:
        by_dim_transfs: list[ozm06trans.ByDimensionTransform] = []
        for t in self.transformations:
            input_axes = tuple(self.input.axes_names.index(ax_name) for ax_name in t.input.axes_names)
            output_axes = tuple(self.output.axes_names.index(ax_name) for ax_name in t.output.axes_names)
            by_dim_transfs.append(
                ozm06trans.ByDimensionTransform(
                    input_axes=input_axes,
                    output_axes=output_axes,
                    transformation=t.to_model(),
                )
            )
        return ozm06trans.ByDimension(
            name=self.name,
            input=self.input.to_model_cs_ident(),
            output=self.output.to_model_cs_ident(),
            transformations=tuple(by_dim_transfs),
        )


class CsGen:
    """A coordinate system generator

    Use it to create coordinate systems on the fly while avoiding
    repeating names
    """

    def __init__(self, base_name: str):
        self._base_name = base_name
        self._cs_count: int = 0
        super().__init__()

    def generate(self, *, num_axes: int) -> CoordSystem:
        out = CoordSystem(
            name=f"{self._base_name}{self._cs_count}",
            axes=[
                Axis(
                    name=f"axis_{ax_idx}",
                    type="space",  # FIXME
                )
                for ax_idx in range(num_axes)
            ],
            virtual=True,
        )
        self._cs_count += 1
        return out

    def generate_like(self, other: CoordSystem) -> CoordSystem:
        out = CoordSystem(
            name=f"{self._base_name}{self._cs_count}",
            axes=[
                Axis(
                    name=axis.name,
                    type=axis.type,
                    unit=axis.unit,
                    long_name=axis.long_name,
                )
                for axis in other.axes
            ],
            virtual=True,
        )
        self._cs_count += 1
        return out


def parse_identity(
    model: ozm06trans.Identity,
    *,
    input: CoordSystem,
    out: CoordSystem | CsGen,
) -> IdentityEdge:
    output = out.generate_like(input) if isinstance(out, CsGen) else out
    return IdentityEdge(name=model.name, input=input, output=output)


def parse_translation(
    model: ozm06trans.Translation,
    *,
    input: CoordSystem,
    out: CoordSystem | CsGen,
) -> TranslationEdge:
    output = out.generate_like(input) if isinstance(out, CsGen) else out
    return TranslationEdge(
        translation=np.asarray(model.translation, dtype=float),
        input=input,
        output=output,
        name=input.name,
    )


def parse_scale(
    model: ozm06trans.Scale,
    *,
    input: CoordSystem,
    out: CoordSystem | CsGen,
) -> ScaleEdge:
    output = out.generate_like(input) if isinstance(out, CsGen) else out
    return ScaleEdge(
        scale=np.asarray(model.scale, dtype=float),
        input=input,
        output=output,
        name=model.name,
    )


def parse_map_axis(
    model: ozm06trans.MapAxis,
    *,
    input: CoordSystem,
    out: CoordSystem | CsGen,
) -> MapAxisEdge:
    output = out.generate(num_axes=len(model.mapAxis)) if isinstance(out, CsGen) else out
    return MapAxisEdge(
        input=input,
        output=output,
        name=model.name,
        output_to_input={  # FIXME: double check this. Feels like we depend a lot on order
            output.axes[output_axis].name: input.axes[input_axis].name
            for output_axis, input_axis in enumerate(model.mapAxis)
        },
    )


def parse_affine(
    model: ozm06trans.Affine,
    *,
    input: CoordSystem,
    output: CoordSystem | CsGen,
) -> AffineEdge:
    num_output_axes = len(model.affine_matrix)  # spec doesn't save last row
    output = output.generate(num_axes=num_output_axes) if isinstance(output, CsGen) else output
    affine_array = np.asarray(model.affine_matrix, dtype=float)
    return AffineEdge(
        name=model.name, linear=affine_array[:, :-1], translation=affine_array[:, -1], input=input, output=output
    )


def parse_rotation(
    model: ozm06trans.Rotation,
    *,
    input: CoordSystem,
    out: CoordSystem | CsGen,
) -> RotationEdge:
    num_output_axes = len(model.rotation_matrix)
    output = out.generate(num_axes=num_output_axes) if isinstance(out, CsGen) else out
    return RotationEdge(
        name=model.name,
        linear_matrix=np.asarray(model.rotation_matrix, dtype=float),
        input=input,
        output=output,
    )


def parse_sequence(
    model: ozm06trans.Sequence,
    *,
    input: CoordSystem,
    output: CoordSystem | CsGen,
) -> SequenceEdge:
    parsed_inners: list[BaseTransfEdge] = []

    base_name = "intermediate" + ("" if not model.name else f"_for_{model.name}")
    cs_gen: CsGen = output if isinstance(output, CsGen) else CsGen(base_name=base_name)
    parsed = parse_ngff_transf(
        input=input,
        model=model.transformations[0],
        output=cs_gen if len(model.transformations) > 1 else output,
    )
    parsed_inners.append(parsed)

    for t in model.transformations[1:-1]:
        parsed = parse_ngff_transf(input=parsed.output, output=cs_gen, model=t)
        parsed_inners.append(parsed)

    if len(model.transformations) > 1:
        parsed = parse_ngff_transf(input=parsed.output, output=output, model=model.transformations[-1])
        parsed_inners.append(parsed)

    return SequenceEdge(name=model.name, transformations=parsed_inners)


def parse_by_dimension(
    model: ozm06trans.ByDimension,
    *,
    input: CoordSystem,
    output: CoordSystem | CsGen,
) -> ByDimensionEdge:
    if not isinstance(output, CoordSystem):
        max_out_idx = max(ax_idx for t in model.transformations for ax_idx in t.output_axes)
        output = output.generate(num_axes=max_out_idx + 1)

    piecewise_transforms: list[BaseTransfEdge] = []
    for t in model.transformations:
        inp_axes = [input.axes[i] for i in t.input_axes]
        partial_input = CoordSystem(
            axes=inp_axes,
            name=f"{input.name}_{','.join(ax.name for ax in inp_axes)}",
            virtual=True,
        )

        out_axes = [output.axes[i] for i in t.output_axes]
        partial_out = CoordSystem(
            axes=out_axes,
            name=f"{output.name}_{','.join(ax.name for ax in inp_axes)}",
            virtual=True,
        )

        parsed_t = parse_ngff_transf(model=t.transformation, input=partial_input, output=partial_out)
        piecewise_transforms.append(parsed_t)

    return ByDimensionEdge(
        input=input,
        output=output,
        name=model.name,
        transformations=piecewise_transforms,
    )


def parse_ngff_transf(
    input: CoordSystem,
    model: ozm06trans.AnyTransform,
    output: CoordSystem | CsGen,
) -> BaseTransfEdge:
    if isinstance(model, ozm06trans.Identity):
        return parse_identity(model, input=input, out=output)
    elif isinstance(model, ozm06trans.Translation):
        return parse_translation(model, input=input, out=output)
    elif isinstance(model, ozm06trans.Scale):
        return parse_scale(model, input=input, out=output)
    elif isinstance(model, ozm06trans.MapAxis):
        return parse_map_axis(model, input=input, out=output)
    elif isinstance(model, ozm06trans.Affine):
        return parse_affine(model, input=input, output=output)
    elif isinstance(model, ozm06trans.Rotation):
        return parse_rotation(model, input=input, out=output)
    elif isinstance(model, ozm06trans.Sequence):
        return parse_sequence(model, input=input, output=output)
    elif isinstance(model, ozm06trans.ByDimension):
        return parse_by_dimension(model, input=input, output=output)
    else:
        raise NotImplementedError(f"Unsupported transformation: {model.type}")
