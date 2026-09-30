# ----------------------------------------------------------------------
#  Shared abstract base for the two dynamic edges
# ----------------------------------------------------------------------
from __future__ import annotations

from abc import abstractmethod

import numpy as np

from spatialdata._types import ArrayLike
from spatialdata.transformations._graph.edges.exceptions import NonZeroDroppedAxisError
from spatialdata.transformations._graph.edges.ngff_aligned_edges import (
    AffineEdge,
    BaseTransformationEdge,
    ProjectAxisEdge,
)
from spatialdata.transformations._graph.vert import Axis, CoordSystem


class _DynamicAxisEdgeBase(BaseTransformationEdge):
    """
    Base for the “dynamic” project-axis edges.

    * The constant that is filled into newly created axes is **always** ``0``,
    * Sub‑classes only have to tell the base class which axes they create
      (`_created_axes`) and which they drop (`_dropped_axes`).
    * All heavy lifting – building a temporary ``ProjectAxisEdge``,
      computing the affine matrix and transforming points – lives here.
    """

    def __init__(
        self,
        *,
        input: CoordSystem,
        output: CoordSystem,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name, input=input, output=output)

    # ------------------------------------------------------------------
    #  Sub‑class contracts – must be provided by the concrete edge
    # ------------------------------------------------------------------
    @property
    @abstractmethod
    def _created_axes(self) -> set[Axis]:
        """Axes that are added to the output (empty for the Drop edge)."""

    @property
    @abstractmethod
    def _dropped_axes(self) -> set[Axis]:
        """Axes that are removed from the input (empty for the Create edge)."""

    # ------------------------------------------------------------------
    #  Helper: build the underlying ProjectAxisEdge on demand
    # ------------------------------------------------------------------
    def _proj_edge(self) -> ProjectAxisEdge:
        """Instantiate a ProjectAxisEdge with the sets supplied by the subclass."""
        return ProjectAxisEdge(
            input=self.input,
            output=self.output,
            dropped_inputs=self._dropped_axes,
            created_outputs=self._created_axes,
            name=self.name,
        )

    # ------------------------------------------------------------------
    #  Public API required by BaseTransformationEdge
    # ------------------------------------------------------------------
    def to_affine(self, name: str | None = None) -> AffineEdge:
        """
        Delegates to a temporary ProjectAxisEdge; because the constant is always
        zero the affine matrix returned by ProjectAxisEdge is already correct.
        """
        return self._proj_edge().to_affine(name=name)

    # ------------------------------------------------------------------
    #  Validation that the dropped columns are all zero before delegating
    # ------------------------------------------------------------------
    def pre_transforming_check(self, points: ArrayLike) -> None:
        """
        Verify that every column belonging to a dropped axis consists only of
        zeros.  If any non‑zero value is found, raise ``NonZeroDroppedAxisError``.
        """
        if not self._dropped_axes:
            return

        # Map input axis → column index.
        col_index = {ax: idx for idx, ax in enumerate(self.input.axes)}
        for ax in self._dropped_axes:
            col = points[:, col_index[ax]]
            if not np.allclose(col, 0):
                raise NonZeroDroppedAxisError(
                    axis_name=ax.name,
                    expected=0,
                    observed=col,
                )

    def transform_points(self, points: ArrayLike) -> ArrayLike:
        """Run the zero‑check, then delegate to the ProjectAxisEdge."""
        self.pre_transforming_check(points)
        return self._proj_edge().transform_points(points)

    # ------------------------------------------------------------------
    #  Human‑readable representation – shows which axes are created / dropped
    # ------------------------------------------------------------------
    def __repr__(self) -> str:  # pragma: no cover – representation only
        base = super().__repr__()
        if self._created_axes:
            created = ", ".join(ax.name for ax in self._created_axes)
            base += f"\n  + creates axes: {created}"
        if self._dropped_axes:
            dropped = ", ".join(ax.name for ax in self._dropped_axes)
            base += f"\n  - drops axes: {dropped}"
        return base


# ----------------------------------------------------------------------
#  DynamicCreateAxisEdge – adds missing axes (filled with zero)
# ----------------------------------------------------------------------
class DynamicCreateAxisEdge(_DynamicAxisEdgeBase):
    """Adds axes that are present in ``output`` but absent from ``input``."""

    @property
    def _created_axes(self) -> set[Axis]:
        # Axes in the output that are not in the input must be created.
        return {ax for ax in self.output.axes if ax not in self.input.axes}

    @property
    def _dropped_axes(self) -> set[Axis]:
        # This edge never drops anything.
        return set()

    def inverse(self, name: str | None = None) -> BaseTransformationEdge:
        return DynamicDropAxisEdge(
            input=self.output,
            output=self.input,
            name=name,
        )


# ----------------------------------------------------------------------
#  DynamicDropAxisEdge – removes axes that must be zero
# ----------------------------------------------------------------------
class DynamicDropAxisEdge(_DynamicAxisEdgeBase):
    """Removes axes present in ``input`` but missing from ``output``."""

    @property
    def _created_axes(self) -> set[Axis]:
        # This edge never creates new axes.
        return set()

    @property
    def _dropped_axes(self) -> set[Axis]:
        # Axes in the input that are not present in the output are to be dropped.
        return {ax for ax in self.input.axes if ax not in self.output.axes}

    def inverse(self, name: str | None = None) -> BaseTransformationEdge:
        return DynamicCreateAxisEdge(
            input=self.output,
            output=self.input,
            name=name,
        )
