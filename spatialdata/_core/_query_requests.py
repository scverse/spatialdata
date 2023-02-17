from typing import Any, Callable

from spatialdata._core.core_utils import SpatialElement


class RequestSequence:
    pass


#
def _dict_query_dispatcher(
    elements: dict[str, SpatialElement], query_function: Callable[[SpatialElement], SpatialElement], **kwargs: Any
) -> dict[str, SpatialElement]:
    queried_elements = {}
    for key, element in elements.items():
        result = query_function(element, **kwargs)
        if result is not None:
            # query returns None if it is empty
            queried_elements[key] = result
    return queried_elements
