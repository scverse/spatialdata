from typing import Callable


from spatialdata._core.core_utils import SpatialElement


class RequestSequence:
    pass


#
def _dict_query_dispatcher(
    elements: dict[str, SpatialElement], query_function: Callable, **kwargs
) -> dict[str, SpatialElement]:
    queried_elements = {}
    for key, element in elements.items():
        queried_elements[key] = query_function(element, **kwargs)
    return queried_elements
