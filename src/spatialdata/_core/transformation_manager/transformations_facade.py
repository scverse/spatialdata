from __future__ import annotations

from collections.abc import Sequence

from spatialdata._core.transformation_manager._transformation_graph import TransformationGraph
from spatialdata.transformations import BaseTransformation
from spatialdata.transformations.graph.edge import BaseTransfEdge


class TransformationsFacade:
    def __init__(self) -> None:
        self.graph = TransformationGraph()

    def _get_element_transformations(
        self, element_name: str, target_coordinate_system_names: Sequence[str]
    ) -> list[BaseTransfEdge]:

        return self.graph.get_outgoing_edges(element_name, target_coordinate_system_names)

    def set_element_transformations(self, element_name: str, transformations: Sequence[BaseTransformation]) -> None:

        pass
