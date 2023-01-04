from abc import ABC, ABCMeta
from enum import Enum, EnumMeta
from functools import wraps
from typing import Any, Callable


class PrettyEnum(Enum):
    """Enum with a pretty :meth:`__str__` and :meth:`__repr__`."""

    @property
    def v(self) -> Any:
        """Alias for :attr`value`."""
        return self.value

    def __repr__(self) -> str:
        return f"{self.value!r}"

    def __str__(self) -> str:
        return f"{self.value!s}"


def _pretty_raise_enum(cls: type["ErrorFormatterABC"], func: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> "ErrorFormatterABC":
        try:
            return func(*args, **kwargs)  # type: ignore[no-any-return]
        except ValueError as e:
            _cls, value, *_ = args
            e.args = (cls._format(value),)
            raise e

    if not issubclass(cls, ErrorFormatterABC):
        raise TypeError(f"Class `{cls}` must be subtype of `ErrorFormatterABC`.")
    elif not len(cls.__members__):  # type: ignore[attr-defined]
        # empty enum, for class hierarchy
        return func

    return wrapper


class ABCEnumMeta(EnumMeta, ABCMeta):
    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        if getattr(cls, "__error_format__", None) is None:
            raise TypeError(f"Can't instantiate class `{cls.__name__}` " f"without `__error_format__` class attribute.")
        return super().__call__(*args, **kwargs)

    def __new__(cls, clsname: str, superclasses: tuple[type], attributedict: dict[str, Any]) -> "ABCEnumMeta":
        res = super().__new__(cls, clsname, superclasses, attributedict)  # type: ignore[arg-type]
        res.__new__ = _pretty_raise_enum(res, res.__new__)  # type: ignore[assignment,arg-type]
        return res


class ErrorFormatterABC(ABC):
    """Mixin class that formats invalid value when constructing an enum."""

    __error_format__ = "Invalid option `{0}` for `{1}`. Valid options are: `{2}`."

    @classmethod
    def _format(cls, value: Enum) -> str:
        return cls.__error_format__.format(
            value, cls.__name__, [m.value for m in cls.__members__.values()]  # type: ignore[attr-defined]
        )


# TODO: simplify this class
# https://github.com/napari/napari/blob/9ea0159ad2b690556fe56ce480886d8f0b79ffae/napari/layers/labels/_labels_constants.py#L9-L38
# https://github.com/napari/napari/blob/9ea0159ad2b690556fe56ce480886d8f0b79ffae/napari/utils/misc.py#L300-L319
class ModeEnum(str, ErrorFormatterABC, PrettyEnum, metaclass=ABCEnumMeta):
    """Enum which prints available values when invalid value has been passed."""
