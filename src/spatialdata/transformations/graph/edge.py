# pyright: strict

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Final

import numpy as np
import ome_zarr_models.v06.coordinate_transforms as ozm06trans

from spatialdata._core.transformation_manager.exceptions import (
    AxisRedefinitionError,
    EmptyTransformSequenceError,
    IncompatibleCoordSystemsError,
    MissingAxisError,
    NotUnimodularError,
    UnexpectedShapeError,
    UnmappedAxisError,
)
from spatialdata._types import ArrayLike
from spatialdata.transformations.graph.vert import Axis, CoordSystem


class BaseTransfEdge(ABC):
    """Base class for all the transformations defined by the NGFF specification."""

    input: Final[CoordSystem]
    output: Final[CoordSystem]
    name: str | None

    def __init__(
        self,
        *,
        name: str | None = None,
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
    def inverse(self, name: str | None = None) -> BaseTransfEdge | None:
        """Return the inverse of the transformation if it exists"""

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
    def to_affine(self, name: str | None = None) -> AffineEdge:
        """Convert the transformation to an affine transformation, whenever the conversion can be made."""

    def _validate_transform_points_shapes(self, points: ArrayLike) -> None:
        """
        Validate if the shape of the points (coordinates to be transformed) are consistent with the input size of the
        transformation.

        Raises
        ------
        UnexpectedShapeError
            if `points`'s shape is incompatible with this transformation's input shape
        """
        input_size = len(self.input.axes)
        if len(points.shape) != 2 or points.shape[1] != input_size:
            raise UnexpectedShapeError(
                array_name="points",
                expected_shape=f"(<number of points>, {self.input.num_axes})",
                array_shape=points.shape,
            )

    # order of the composition: self is applied first, then the transformation passed as argument
    def compose_with(self, transformation: BaseTransfEdge, name: str | None) -> BaseTransfEdge:
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
        return SequenceEdge(transformations=[self, transformation], name=name)


class AffineEdge(BaseTransfEdge):
    """The Affine transformation from the NGFF specification."""

    linear: Final[ArrayLike]
    translation: Final[ArrayLike]
    affine: Final[ArrayLike]

    def __init__(
        self,
        *,
        name: str | None = None,
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
            raise UnexpectedShapeError(
                array_name="linear", array_shape=linear.shape, expected_shape=expected_linear_shape
            )
        expected_translation_shape = (num_outputs,)
        if translation.shape != expected_translation_shape:
            raise UnexpectedShapeError(
                array_name="translation", array_shape=translation.shape, expected_shape=expected_translation_shape
            )

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
        """Creates an AffineEdge from a raw affine matrix in homogenous coordinates

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
            translation=affine_matrix[:-1, -1],
            input=input,
            output=output,
            name=name,
        )

    @classmethod
    def mapping(cls, input: CoordSystem, output: CoordSystem, name: str | None = None) -> AffineEdge:
        linear: ArrayLike = np.zeros((output.num_axes, input.num_axes), dtype=float)
        for i, des_axis in enumerate(output.axes):
            for j, src_axis in enumerate(input.axes):
                if src_axis.name == des_axis.name:  # FIXME: compare the entire axis?
                    linear[i, j] = 1
        return AffineEdge(
            linear=linear,
            input=input,
            output=output,
            name=name,
        )

    def inverse(self, name: str | None = None) -> BaseTransfEdge | None:
        try:
            # FIXME: I think there are more efficient/precise ways to invert a matrix
            inv = np.linalg.inv(self.affine)
        except np.linalg.LinAlgError:
            return None
        return AffineEdge(
            linear=inv[:-1, :-1],
            translation=inv[:-1, -1],
            input=self.output,
            output=self.input,
            name=name,
        )

    def transform_points(self, points: ArrayLike) -> ArrayLike:
        self._validate_transform_points_shapes(points)
        p = np.vstack([points.T, np.ones(points.shape[0])])
        q = self.affine @ p
        res = q[: self.output.num_axes, :].T
        assert isinstance(res, np.ndarray)
        return res

    def to_affine(self, name: str | None = None) -> AffineEdge:
        return AffineEdge(
            input=self.input, output=self.output, linear=self.linear, translation=self.translation, name=name
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
            raise IncompatibleCoordSystemsError(
                input=input, output=output, message="Axes must have the same number of dimensions"
            )
        super().__init__(input=input, output=output, name=name)

    def inverse(self, name: str | None = None) -> BaseTransfEdge:
        return IdentityEdge(input=self.output, output=self.input, name=name)

    def transform_points(self, points: ArrayLike) -> ArrayLike:
        self._validate_transform_points_shapes(points)
        return points

    def to_affine(self, name: str | None = None) -> AffineEdge:
        return AffineEdge(
            linear=np.eye(self.input.num_axes),
            input=self.input,
            output=self.output,
            name=name,
        )


class MapAxisEdge(BaseTransfEdge):
    """The MapAxis transformation from the NGFF specification."""

    def __init__(
        self,
        *,
        name: str | None = None,
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
            Output coordinate system of the transformation, whose axes
            must be a shuffling of `input`
        """

        if set(input.axes) != set(output.axes):
            raise IncompatibleCoordSystemsError(
                input=input, output=output, message="Input and output must have the same axes"
            )
        super().__init__(input=input, output=output, name=name)

    def __repr__(self) -> str:
        s = super().__repr__() + "\n"
        s += "\n".join(
            f"    {out.name} <- {inp.name}\n" for out, inp in zip(self.output.axes, self.input.axes, strict=True)
        )
        return s

    def inverse(self, name: str | None = None) -> BaseTransfEdge:
        return MapAxisEdge(
            input=self.output,
            output=self.input,
            name=name,
        )

    def transform_points(self, points: ArrayLike) -> ArrayLike:
        self._validate_transform_points_shapes(points)
        new_indices = [self.input.axes.index(out_ax) for out_ax in self.output.axes]
        mapped = points[:, new_indices]
        assert isinstance(mapped, np.ndarray)
        return mapped

    def to_affine(self, name: str | None = None) -> AffineEdge:
        return AffineEdge.mapping(input=self.input, output=self.output, name=name)


class ProjectAxisEdge(BaseTransfEdge):
    dropped_inputs: Final[set[Axis]]
    created_outputs: Final[set[Axis]]

    def __init__(
        self,
        *,
        name: str | None = None,
        input: CoordSystem,
        output: CoordSystem,
        dropped_inputs: set[Axis],
        created_outputs: set[Axis],
    ) -> None:
        """
        Parameters
        ----------
        dropped_inputs
            axes in `input` that will be dropped by this transformation
        created_inputs
            axes in `output` that will be set to 0

        Raises
        ------
        MissingAxisError
            axis in `dropped_inputs` not in `input`
            axis in `created_outputs` not in `output`
        IncompatibleCoordSystemsError
            when input can't be mapped to output given dropped_inputs and created_outputs
        """

        for axis in dropped_inputs:
            if axis not in input.axes:
                raise MissingAxisError(axis=axis, cs=input)
        for axis in created_outputs:
            if axis not in output.axes:
                raise MissingAxisError(axis=axis, cs=output)
        if input.num_axes - len(dropped_inputs) + len(created_outputs) != output.num_axes:
            message = f"Can't map from {input} to {output}"
            if dropped_inputs:
                message += f" dropping {dropped_inputs}"
            if created_outputs:
                message += f" creating {created_outputs}"
            raise IncompatibleCoordSystemsError(
                input=input,
                output=output,
                message=message,
            )

        self.dropped_inputs = set(dropped_inputs)
        self.created_outputs = set(created_outputs)
        super().__init__(name=name, input=input, output=output)

    def to_affine(self, name: str | None = None) -> AffineEdge:
        linear = np.zeros((self.output.num_axes, self.input.num_axes), dtype=float)
        input_indices = iter(range(self.input.num_axes))
        for out_idx, out_ax in enumerate(self.output.axes):
            if out_ax in self.created_outputs:
                continue
            in_idx = next(input_indices)
            if in_idx in self.dropped_inputs:
                continue
            linear[out_idx, in_idx] = 1

        return AffineEdge(name=name, input=self.input, output=self.output, linear=linear)

    def transform_points(self, points: ArrayLike) -> ArrayLike:
        return self.to_affine().transform_points(points)

    def inverse(self, name: str | None = None) -> BaseTransfEdge | None:
        # FIXME: there may be other cases where this is invertible
        if self.input.num_axes != self.output.num_axes:
            return None
        if len(self.dropped_inputs) > 0:
            return None
        if len(self.created_outputs) > 0:
            return None
        return ProjectAxisEdge(
            input=self.output,
            output=self.input,
            dropped_inputs=set(),
            created_outputs=set(),
            name=name,
        )


def parse_project_axis(
    model: ozm06trans.ProjectAxis,
    *,
    input: CoordSystem,
    out: CoordSystem | CsGen,
) -> ProjectAxisEdge:
    if isinstance(out, CoordSystem):
        output = out
    else:
        num_dropped_inputs = len(model.droppedInputs or ())
        num_created_outputs = len(model.createdOutputs or ())
        num_output_axes = input.num_axes - num_dropped_inputs + num_created_outputs
        output = out.generate(num_axes=num_output_axes)

    return ProjectAxisEdge(
        created_outputs={output.axes[i] for i in model.createdOutputs or ()},
        dropped_inputs={input.axes[i] for i in model.droppedInputs or ()},
        input=input,
        output=output,
        name=model.name,
    )


class TranslationEdge(BaseTransfEdge):
    """The Translation transformation from the NGFF specification."""

    def __init__(
        self,
        *,
        name: str | None = None,
        translation: ArrayLike,
        input: CoordSystem,
        output: CoordSystem,
    ) -> None:
        """
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

        Raises
        ------
        IncompatibleCoordSystemsError
            If the input and output have different number of dimensions
        """
        if input.num_axes != output.num_axes:
            raise IncompatibleCoordSystemsError(
                input=input, output=output, message="Number of input and output axes must be the same"
            )
        self.translation = translation
        super().__init__(input=input, output=output, name=name)

    def __repr__(self) -> str:
        return super().__repr__() + str(self.translation)

    def inverse(self, name: str | None = None) -> BaseTransfEdge:
        return TranslationEdge(
            translation=-self.translation,
            input=self.output,
            output=self.input,
            name=name,
        )

    def transform_points(self, points: ArrayLike) -> ArrayLike:
        self._validate_transform_points_shapes(points)
        return points + self.translation

    def to_affine(self, name: str | None = None) -> AffineEdge:
        return AffineEdge(
            linear=np.identity(self.input.num_axes),
            translation=self.translation,
            input=self.input,
            output=self.output,
            name=name,
        )


class ScaleEdge(BaseTransfEdge):
    """The Scale transformation from the NGFF specification."""

    def __init__(
        self,
        *,
        name: str | None = None,
        scale: ArrayLike,
        input: CoordSystem,
        output: CoordSystem,
    ) -> None:
        """
        Parameters
        ----------
        scale
            A vector specifying the scale along each axis of `input`.
        input
            Input coordinate system of the transformation.
        output
            Output coordinate system of the transformation.

        Raises
        ------
        UnexpectedShapeError
            If scale doesn't have the same number of elements as input has axes
        IncompatibleCoordSystemsError
            If input and output have different number of axes
        """
        expected_scale_shape = (input.num_axes,)
        if scale.shape != expected_scale_shape:
            raise UnexpectedShapeError(array_name="scale", array_shape=scale.shape, expected_shape=expected_scale_shape)
        if input.num_axes != output.num_axes:
            raise IncompatibleCoordSystemsError(
                input=input, output=output, message="input and output must have same number of dimensions"
            )
        self.scale = scale
        super().__init__(input=input, output=output, name=name)

    def __repr__(self) -> str:
        return super().__repr__() + str(self.scale)

    def inverse(self, name: str | None = None) -> ScaleEdge | None:
        if any(s == 0 for s in self.scale):
            return None
        new_scale = 1 / self.scale
        return ScaleEdge(
            scale=new_scale,
            input=self.output,
            output=self.input,
            name=name,
        )

    def transform_points(self, points: ArrayLike) -> ArrayLike:
        self._validate_transform_points_shapes(points)
        return points * self.scale

    def to_affine(self, name: str | None = None) -> AffineEdge:
        return AffineEdge(
            linear=np.diag(self.scale),
            input=self.input,
            output=self.output,
            name=name,
        )


class RotationEdge(BaseTransfEdge):
    """The Rotation transformation from the NGFF specification."""

    rotation: Final[ArrayLike]

    def __init__(
        self,
        *,
        name: str | None = None,
        linear_matrix: ArrayLike,
        input: CoordSystem,
        output: CoordSystem,
    ) -> None:
        """
        Parameters
        ----------
        linear_matrix
            an array of shape (output.num_axes, input.num_axes) representing the rotation
        input
            Input coordinate system of the transformation.
        output
            Output coordinate system of the transformation.
        Raises
        ------
        UnexpectedShapeError
            if linear_matrix's shape isn't (output.num_axes, input.num_axes)
        IncompatibleCoordSystemsError
            if input and output don't have the same number of axes
        NotUnimodularError
            if linear_matrix doesn't have determinant ~= 1
        """
        if input.num_axes != output.num_axes:
            raise IncompatibleCoordSystemsError(
                input=input, output=output, message="input and output should have the same numbe rof axes"
            )
        expected_shape = (output.num_axes, input.num_axes)
        if linear_matrix.shape != expected_shape:
            raise UnexpectedShapeError(
                array_name="linear_matrix", array_shape=linear_matrix.shape, expected_shape=expected_shape
            )
        if not np.isclose(np.linalg.det(linear_matrix), 1.0):
            raise NotUnimodularError(matrix=linear_matrix)
        linear_matrix.flags.writeable = False
        self.rotation = linear_matrix
        super().__init__(input=input, output=output, name=name)

    def __repr__(self) -> str:
        s = super().__repr__() + "\n"
        s += "\n".join(str(row) for row in self.rotation)
        return s

    def inverse(self, name: str | None = None) -> BaseTransfEdge:
        return RotationEdge(
            linear_matrix=self.rotation.T,
            input=self.output,
            output=self.input,
            name=name,
        )

    def transform_points(self, points: ArrayLike) -> ArrayLike:
        self._validate_transform_points_shapes(points)
        res = (self.rotation @ points.T).T
        assert isinstance(res, np.ndarray)
        return res

    def to_affine(self, name: str | None = None) -> AffineEdge:
        return AffineEdge(
            linear=self.rotation,
            input=self.input,
            output=self.output,
            name=name,
        )


class SequenceEdge(BaseTransfEdge):
    """The Sequence transformation from the NGFF specification."""

    def __init__(
        self,
        *,
        name: str | None = None,
        transformations: Sequence[BaseTransfEdge],
    ) -> None:
        """
        Init the NgffSequence object.

        Parameters
        ----------
        transformations
            The transformations which compose the sequence.
        Raises
        ------
        EmptyTransformSequence
            if `transformations` is empty
        IncompatibleCoordSystemsError
            if any item in `transformations` is incompatible with its neighbors
        """
        if len(transformations) == 0:
            raise EmptyTransformSequenceError()
        previous_transf = transformations[0]
        for transf_idx, current_transf in enumerate(transformations[1:], start=1):
            if previous_transf.output != current_transf.input:
                raise IncompatibleCoordSystemsError(
                    input=current_transf.input,
                    output=previous_transf.output,
                    message=(
                        f"Output of transformation #{transf_idx - 1} is different "
                        f"from input of transformation #{transf_idx}"
                    ),
                )
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

    def inverse(self, name: str | None = None) -> SequenceEdge | None:
        inverted: list[BaseTransfEdge] = []
        for t in reversed(self.transformations):
            inv = t.inverse()
            if inv is None:
                return None
            inverted.append(inv)
        return SequenceEdge(transformations=inverted, name=name)

    def to_affine(self, name: str | None = None) -> AffineEdge:
        composed = self.transformations[0].to_affine().affine
        for t in self.transformations[1:]:
            a = t.to_affine()
            composed = a.affine @ composed
        return AffineEdge.from_affine_matrix(
            affine_matrix=composed,
            input=self.input,
            output=self.output,
            name=name,
        )

    def transform_points(self, points: ArrayLike) -> ArrayLike:
        return self.to_affine().transform_points(points)  # FIXME


class ByDimensionEdge(BaseTransfEdge):
    """The ByDimension transformation from the NGFF specification."""

    transformations: Final[Sequence[BaseTransfEdge]]

    def __init__(
        self,
        *,
        name: str | None = None,
        transformations: Sequence[BaseTransfEdge],
        input: CoordSystem,
        output: CoordSystem,
    ) -> None:
        """
        Parameters
        ----------
        transformations
            A list of transformations, whose set of output coordinate systems partition the output coordinate system of
            the ByDimension transformation.
        input
            The input coordinate system of the transformation.
        output
            The output coordinate system of the transformation.
        Raises
        ------
        MissingAxisError
            if any input axis from `transformations` is not present in `input` or
            if any output axis from `transformations` is not present in `output`.
        AxisRedefinitionError
            if an output axis is specified by more than one item of `transformations`
        UnmappedAxisError
            if axis of `output` is not covered by any item of `transformations`
        """
        # we check that:
        # 1. each input from each transformation in self.transformation must appear in the set of input axes
        # 2. each output from each transformation in self.transformation must appear at most once in the set of output
        # axes
        defined_output_axes: set[str] = set()
        for t in transformations:
            for ax in t.input.axes:
                if ax not in input.axes:
                    raise MissingAxisError(axis=ax, cs=input)
            for ax in t.output.axes:
                if ax not in output.axes:
                    raise MissingAxisError(axis=ax, cs=output)
                if ax.name in defined_output_axes:
                    raise AxisRedefinitionError(axis=ax)
                defined_output_axes.add(ax.name)
        for ax in output.axes:
            if ax.name not in defined_output_axes:
                raise UnmappedAxisError(axis=ax, cs=output)

        self.transformations = tuple(transformations)
        super().__init__(input=input, output=output, name=name)

    def __repr__(self) -> str:
        from textwrap import indent

        out = super().__repr__() + " [\n"
        for t in self.transformations:
            out += indent(repr(t), prefix="    ") + "\n"
        out += "]"
        return out

    def inverse(self, name: str | None = None) -> BaseTransfEdge | None:
        inverse_transformations: list[BaseTransfEdge] = []
        for t in self.transformations:
            inv = t.inverse()
            if inv is None:
                return None
            inverse_transformations.append(inv)
        return ByDimensionEdge(
            transformations=inverse_transformations,
            input=self.output,
            output=self.input,
            name=name,
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

    def to_affine(self, name: str | None = None) -> AffineEdge:
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
            affine_matrix=m,
            input=self.input,
            output=self.output,
            name=name,
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
    if isinstance(out, CoordSystem):
        output = out
    else:
        dummy_cs = out.generate(num_axes=len(model.mapAxis))
        output = CoordSystem(
            name=dummy_cs.name,
            axes=[input.axes[i] for i in model.mapAxis],
            virtual=True,
        )
    return MapAxisEdge(
        input=input,
        output=output,
        name=model.name,
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
