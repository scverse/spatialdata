from collections.abc import Collection
from types import TracebackType
from typing import NamedTuple, cast

import pandas as pd
from anndata import AnnData


class ErrorDetails(NamedTuple):
    location: tuple[str, ...]
    """Tuple of strings identifying the element for which the error occurred."""

    message: str
    """A human readable error message."""


class ValidationError(ValueError):
    def __init__(self, title: str, errors: list[ErrorDetails]):
        self._errors: list[ErrorDetails] = list(errors)
        super().__init__(title)

    @property
    def title(self) -> str:
        return str(self.args[0]) if self.args else ""

    @property
    def errors(self) -> list[ErrorDetails]:
        return list(self._errors)

    def __str__(self) -> str:
        return f"{self.title}\n" + "\n".join(
            f"  {'/'.join(str(key) for key in details.location)}: {details.message}" for details in self.errors
        )


def check_target_region_column_symmetry(table: AnnData, region_key: str, target: str | pd.Series) -> None:
    """
    Check region and region_key column symmetry.

    This checks whether the specified targets are also present in the region key column in obs and raises an error
    if this is not the case.

    Parameters
    ----------
    table
        Table annotating specific SpatialElements
    region_key
        The column in obs containing for each row which SpatialElement is annotated by that row.
    target
         Name of target(s) SpatialElement(s)

    Raises
    ------
    ValueError
        If there is a mismatch between specified target regions and regions in the region key column of table.obs.

    Example
    -------
    Assuming we have a table with region column in obs given by `region_key` called 'region' for which we want to check
    whether it contains the specified annotation targets in the `target` variable as `pd.Series['region1', 'region2']`:

    ```python
    check_target_region_column_symmetry(table, region_key=region_key, target=target)
    ```

    This returns None if both specified targets are present in the region_key obs column. In this case the annotation
    targets can be safely set. If not then a ValueError is raised stating the elements that are not shared between
    the region_key column in obs and the specified targets.
    """
    found_regions = set(table.obs[region_key].unique().tolist())
    target_element_set = [target] if isinstance(target, str) else target
    symmetric_difference = found_regions.symmetric_difference(target_element_set)
    if symmetric_difference:
        raise ValueError(
            f"Mismatch(es) found between regions in region column in obs and target element: "
            f"{', '.join(diff for diff in symmetric_difference)}"
        )


def check_valid_name(name: str) -> None:
    """
    Check that a name is valid for SpatialData elements.

    This checks whether the proposed name fulfills the naming restrictions and raises an error
    otherwise.

    Parameters
    ----------
    name
        The name for a SpatialData element

    Raises
    ------
    TypeError
        If given argument is not of type string.
    ValueError
        If the proposed name violates a naming restriction.
    """
    if not isinstance(name, str):
        raise TypeError(f"Name must be a string, not {type(name).__name__}.")
    if len(name) == 0:
        raise ValueError("Name cannot be an empty string.")
    if name == ".":
        raise ValueError("Name cannot be '.'.")
    if name == "..":
        raise ValueError("Name cannot be '..'.")
    if name.startswith("__"):
        raise ValueError("Name cannot start with '__'.")
    if not all(c.isalnum() or c in "_-." for c in name):
        raise ValueError("Name must contain only alphanumeric characters, underscores, dots and hyphens.")


def check_all_keys_case_insensitively_unique(keys: Collection[str], location: tuple[str, ...] = ()) -> None:
    """
    Check that all keys are unique when ignoring case.

    This checks whether the keys contain no duplicates on an case-insensitive system. If keys
    differ in character case, an error is raised.

    Parameters
    ----------
    keys
        A collection of string keys
    location
        Tuple of strings identifying the parent element

    Raises
    ------
    ValueError
        If two keys differ only in character case.

    Example
    -------

    ```pycon
    >>> check_all_keys_case_insensitively_unique(["abc", "def"])
    >>> check_all_keys_case_insensitively_unique(["abc", "def", "Abc"])
    Traceback (most recent call last):
        ...
    ValueError: Key `Abc` is not unique, or another case-variant of it exists.
    ```
    """
    seen: set[str | None] = set()
    with raise_validation_errors(
        title="Element contains conflicting keys.\n"
        "For renaming, please see the discussion here https://github.com/scverse/spatialdata/discussions/707 .",
        exc_type=ValueError,
    ) as collect_error:
        for key in keys:
            normalized_key = key.lower()
            with collect_error(location=location + (key,)):
                check_key_is_case_insensitively_unique(key, seen)
            seen.add(normalized_key)


def check_key_is_case_insensitively_unique(key: str, other_keys: set[str | None]) -> None:
    """
    Check that a specific key is not contained in a set of keys, ignoring case.

    This checks whether a given key is not contained among a set of reference keys. If the key or
    a case-variant of it is contained, an error is raised.

    Parameters
    ----------
    key
        A string key
    other_keys
        A collection of string keys

    Raises
    ------
    ValueError
        If reference keys contain a variant of the key that only differs in character case.

    Example
    -------

    ```pycon
    >>> check_key_is_case_insensitively_unique("def", ["abc"])
    >>> check_key_is_case_insensitively_unique("abc", ["def", "Abc"])
    Traceback (most recent call last):
        ...
    ValueError: Key `Abc` is not unique, or another case-variant of it exists.
    ```
    """
    normalized_key = key.lower()
    if normalized_key in other_keys:
        raise ValueError(
            f"Key `{key}` is not unique as it exists with a different element type, or another "
            f"case-variant of it exists."
        )


def check_valid_dataframe_column_name(name: str) -> None:
    """
    Check that a name is valid for SpatialData table dataframe.

    This checks whether the proposed name fulfills the naming restrictions and raises an error
    otherwise. In addition to the element naming restriction, a column cannot be named "_index".

    Parameters
    ----------
    name
        The name for a table column

    Raises
    ------
    TypeError
        If given argument is not of type string.
    ValueError
        If the proposed name violates a naming restriction.
    """
    check_valid_name(name)
    if name == "_index":
        raise ValueError("Name cannot be '_index'")


def validate_table_attr_keys(data: AnnData, location: tuple[str, ...] = ()) -> None:
    """
    Check that all keys of all AnnData attributes have valid names.

    This checks for AnnData obs, var, obsm, obsp, varm, varp, uns, layers whether their keys fulfill the
    naming restrictions and raises an error otherwise.

    Parameters
    ----------
    data
        The AnnData table
    location
        Tuple of strings identifying the parent element

    Raises
    ------
    ValueError
        If the AnnData contains one or several invalid keys.
    """
    with raise_validation_errors(
        title="Table contains invalid names.\n"
        "For renaming, please see the discussion here https://github.com/scverse/spatialdata/discussions/707 .",
        exc_type=ValueError,
    ) as collect_error:
        for attr in ("obs", "obsm", "obsp", "var", "varm", "varp", "uns", "layers"):
            attr_path = location + (attr,)
            with collect_error(location=attr_path):
                check_all_keys_case_insensitively_unique(getattr(data, attr).keys(), location=attr_path)
            for key in getattr(data, attr):
                key_path = attr_path + (key,)
                with collect_error(location=key_path):
                    if attr in ("obs", "var"):
                        check_valid_dataframe_column_name(key)
                    else:
                        check_valid_name(key)


class _ErrorDetailsCollector:
    """
    Context manager to collect possible exceptions into a list.

    This is syntactic sugar for shortening the try/except construction when the error handling is
    the same. Only for internal use by `raise_validation_errors`.

    Parameters
    ----------
    exc_type
        The class of the exception to catch. Other exceptions are raised.
    """

    def __init__(
        self,
        exc_type: type[BaseException] | tuple[type[BaseException], ...],
    ) -> None:
        self.errors: list[ErrorDetails] = []
        self._location: tuple[str, ...] = ()
        self._exc_type = exc_type
        self._exc_type_override: type[BaseException] | tuple[type[BaseException], ...] | None = None

    def __call__(
        self,
        location: str | tuple[str, ...] = (),
        expected_exception: type[BaseException] | tuple[type[BaseException], ...] | None = None,
    ) -> "_ErrorDetailsCollector":
        """
        Set or override error details in advance before an exception is raised.

        Parameters
        ----------
        location
            Tuple of strings identifying the parent element
        expected_exception
            The class of the exception to catch. Other exceptions are raised.
        """
        if isinstance(location, str):
            location = (location,)
        self._location = location
        if expected_exception is not None:
            self._exc_type_override = expected_exception
        return self

    def __enter__(self) -> None:
        pass

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        # No exception happened
        if exc_type is None:
            return True
        # An exception that we cannot handle, let the interpreter raise it.
        handled_exception_type = self._exc_type_override or self._exc_type
        if not issubclass(exc_type, handled_exception_type):
            return False
        # One of the expected exceptions happened, or another ValidationError that we can merge.
        assert exc_val is not None
        if issubclass(exc_type, ValidationError):
            exc_val = cast(ValidationError, exc_val)
            self.errors += exc_val.errors
        else:
            details = ErrorDetails(location=self._location, message=str(exc_val.args[0]))
            self.errors.append(details)
        # Reset temporary attributes
        self._location = ()
        self._exc_type_override = None
        return True


class raise_validation_errors:
    """
    Context manager to raise collected exceptions together as one ValidationError.

    This is syntactic sugar for shortening the try/except construction when the error handling is
    the same.

    Parameters
    ----------
    title
        A validation error summary to display above the individual errors
    exc_type
        The class of the exception to catch. Other exceptions are raised.

    Example
    -------

    ```pycon
    >>> with raise_validation_errors(
    ...     "Some errors happened", exc_type=ValueError
    ... ) as collect_error:
    ...     for key, value in {"first": 1, "second": 2, "third": 3}.items():
    ...         with collect_error(location=key):
    ...             if value % 2 != 0:
    ...                 raise ValueError("Odd value encountered")
    ...
    spatialdata._core.validation.ValidationErro: Some errors happened
    first: Odd value encountered
    third: Odd value encountered
    ```
    """

    def __init__(
        self,
        title: str = "Validation errors happened",
        exc_type: type[BaseException] | tuple[type[BaseException], ...] = ValueError,
    ) -> None:
        self._message = title
        self._collector = _ErrorDetailsCollector(exc_type=exc_type)

    def __enter__(self) -> _ErrorDetailsCollector:
        return self._collector

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool:
        # An unexpected exception happened, let the interpreter handle it.
        if exc_type is not None:
            return False
        # Exceptions were collected that we want to raise as a combined validation error.
        if self._collector.errors:
            raise ValidationError(
                title=self._message + "\nTo fix, run `spatialdata.utils.sanitize_table(adata)`.",
                errors=self._collector.errors,
            )
        return True
