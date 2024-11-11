# from https://stackoverflow.com/questions/10307696/how-to-put-a-variable-into-python-docstring
from collections.abc import Callable
from typing import Any, TypeVar

T = TypeVar("T")


def docstring_parameter(*args: Any, **kwargs: Any) -> Callable[[T], T]:
    def dec(obj: T) -> T:
        if obj.__doc__:
            obj.__doc__ = obj.__doc__.format(*args, **kwargs)
        return obj

    return dec
