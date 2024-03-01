from __future__ import annotations

from textwrap import dedent
from typing import Any, Callable


def inject_docs(**kwargs: Any) -> Callable[..., Any]:  # noqa: D103
    # taken from scanpy
    def decorator(obj: Any) -> Any:
        obj.__doc__ = dedent(obj.__doc__).format(**kwargs)
        return obj

    def decorator2(obj: Any) -> Any:
        obj.__doc__ = dedent(kwargs["__doc__"])
        return obj

    if isinstance(kwargs.get("__doc__", None), str) and len(kwargs) == 1:
        return decorator2

    return decorator
