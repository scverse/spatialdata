from __future__ import annotations

# from https://stackoverflow.com/a/24860799/3343783
import re
from typing import TYPE_CHECKING, Union

import numpy as np

from spatialdata._types import ArrayLike

# I was using "from numbers import Number" but this led to mypy errors, so I switched to the following:
Number = Union[int, float]

if TYPE_CHECKING:
    pass


def _parse_list_into_array(array: Union[list[Number], ArrayLike]) -> ArrayLike:
    if isinstance(array, list):
        array = np.array(array)
    if array.dtype != float:
        array = array.astype(float)
    return array


def _atoi(text: str) -> Union[int, str]:
    return int(text) if text.isdigit() else text


# from https://stackoverflow.com/a/5967539/3343783
def _natural_keys(text: str) -> list[Union[int, str]]:
    """
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    """
    return [_atoi(c) for c in re.split(r"(\d+)", text)]


def _affine_matrix_multiplication(matrix: ArrayLike, data: ArrayLike) -> ArrayLike:
    assert len(data.shape) == 2
    assert matrix.shape[1] - 1 == data.shape[1]
    vector_part = matrix[:-1, :-1]
    offset_part = matrix[:-1, -1]
    result = data @ vector_part.T + offset_part
    assert result.shape[0] == data.shape[0]
    return result  # type: ignore[no-any-return]
