import functools
import re
import warnings
from collections.abc import Callable, Generator
from itertools import islice
from typing import Any, TypeVar

import numpy as np
import pandas as pd
from anndata import AnnData
from dask import array as da
from dask.array import Array as DaskArray
from xarray import DataArray, Dataset, DataTree

from spatialdata._types import ArrayLike
from spatialdata.transformations import Sequence, Translation, get_transformation, set_transformation

# I was using "from numbers import Number" but this led to mypy errors, so I switched to the following:
Number = int | float
RT = TypeVar("RT")


def _parse_list_into_array(array: list[Number] | ArrayLike) -> ArrayLike:
    if isinstance(array, list):
        array = np.array(array)
    if array.dtype != float:
        return array.astype(float)
    return array


def _atoi(text: str) -> int | str:
    return int(text) if text.isdigit() else text


# from https://stackoverflow.com/a/5967539/3343783
def _natural_keys(text: str) -> list[int | str]:
    """Sort keys in natural order.

    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments).
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


def unpad_raster(raster: DataArray | DataTree) -> DataArray | DataTree:
    """
    Remove padding from a raster type that was eventually added by the rotation component of a transformation.

    Parameters
    ----------
    raster
        The raster to unpad. Contiguous zero values are considered padding.

    Returns
    -------
    The unpadded raster.
    """
    from spatialdata.models import get_axes_names
    from spatialdata.transformations._utils import compute_coordinates

    def _compute_paddings(data: DataArray, axis: str) -> tuple[int, int]:
        others = list(data.dims)
        others.remove(axis)
        # mypy (luca's pycharm config) can't see the isclose method of dask array
        s = da.isclose(data.sum(dim=others), 0)
        # TODO: rewrite this to use dask array; can't get it to work with it
        x = s.compute()
        non_zero = np.where(x == 0)[0]
        if len(non_zero) == 0:
            min_coordinate, max_coordinate = data.coords[axis].min().item(), data.coords[axis].max().item()
            if not min_coordinate != 0:
                raise ValueError(
                    f"Expected minimum coordinate for axis {axis} to be 0,"
                    f"but got {min_coordinate}. Please report this bug."
                )
            if max_coordinate != data.shape[data.dims.index(axis)] - 1:
                raise ValueError(
                    f"Expected maximum coordinate for axis {axis} to be"
                    f"{data.shape[data.dims.index(axis)] - 1},"
                    f"but got {max_coordinate}. Please report this bug."
                )
            return 0, data.shape[data.dims.index(axis)]

        left_pad = non_zero[0]
        right_pad = non_zero[-1] + 1
        return left_pad, right_pad

    axes = get_axes_names(raster)
    translation_axes = []
    translation_values: list[float] = []
    unpadded = raster

    # TODO: this "if else" will be unnecessary once we remove the
    #  concept of intrinsic coordinate systems and we make the
    #  transformations and xarray coordinates more interoperable
    if isinstance(unpadded, DataArray):
        for ax in axes:
            if ax != "c":
                left_pad, right_pad = _compute_paddings(data=unpadded, axis=ax)
                unpadded = unpadded.isel({ax: slice(left_pad, right_pad)})
                translation_axes.append(ax)
                translation_values.append(left_pad)
    elif isinstance(unpadded, DataTree):
        for ax in axes:
            if ax != "c":
                # let's just operate on the highest resolution. This is not an efficient implementation but we can
                # always optimize later
                d = dict(unpadded["scale0"])
                assert len(d) == 1
                xdata = d.values().__iter__().__next__()

                left_pad, right_pad = _compute_paddings(data=xdata, axis=ax)
                unpadded = unpadded.sel({ax: slice(left_pad, right_pad)})
                translation_axes.append(ax)
                translation_values.append(left_pad)
        d = {}
        for k, v in unpadded.items():
            assert len(v.values()) == 1
            xdata = v.values().__iter__().__next__()
            if 0 not in xdata.shape:
                d[k] = Dataset({"image": xdata})
        unpadded = DataTree.from_dict(d)
    else:
        raise TypeError(f"Unsupported type: {type(raster)}")

    translation = Translation(translation_values, axes=tuple(translation_axes))
    old_transformations = get_transformation(element=raster, get_all=True)
    assert isinstance(old_transformations, dict)
    for target_cs, old_transform in old_transformations.items():
        assert old_transform is not None
        sequence = Sequence([translation, old_transform])
        set_transformation(element=unpadded, transformation=sequence, to_coordinate_system=target_cs)
    return compute_coordinates(unpadded)


def get_pyramid_levels(image: DataTree, attr: str | None = None, n: int | None = None) -> list[Any] | Any:
    """
    Access the data/attribute of the pyramid levels of a multiscale spatial image.

    Parameters
    ----------
    image
        The multiscale spatial image.
    attr
        If `None`, return the data of the pyramid level as a `DataArray`, if not None, return the specified attribute
        within the `DataArray` data.
    n
        If not None, return only the `n` pyramid level.

    Returns
    -------
    The pyramid levels data (or an attribute of it) as a list or a generator.
    """
    generator = iterate_pyramid_levels(image, attr)
    if n is not None:
        return next(iter(islice(generator, n, None)))
    return list(generator)


def iterate_pyramid_levels(
    data: DataTree,
    attr: str | None,
) -> Generator[Any, None, None]:
    """
    Iterate over the pyramid levels of a multiscale spatial image.

    Parameters
    ----------
    data
        The multiscale spatial image
    attr
        If `None`, return the data of the pyramid level as a `DataArray`, if not None, return the specified attribute
        within the `DataArray` data.

    Returns
    -------
    A generator to iterate over the pyramid levels.
    """
    names = data["scale0"].ds.keys()
    name: str = next(iter(names))
    for scale in data:
        yield data[scale][name] if attr is None else getattr(data[scale][name], attr)


def _inplace_fix_subset_categorical_obs(subset_adata: AnnData, original_adata: AnnData) -> None:
    """
    Fix categorical obs columns of subset_adata to match the categories of original_adata.

    Parameters
    ----------
    subset_adata
        The subset AnnData object
    original_adata
        The original AnnData object

    Notes
    -----
    See discussion here: https://github.com/scverse/anndata/issues/997
    """
    if not hasattr(subset_adata, "obs") or not hasattr(original_adata, "obs"):
        return
    obs = pd.DataFrame(subset_adata.obs)
    for column in obs.columns:
        is_categorical = isinstance(obs[column].dtype, pd.CategoricalDtype)
        if is_categorical:
            c = obs[column].cat.set_categories(original_adata.obs[column].cat.categories)
            obs[column] = c
    subset_adata.obs = obs


# TODO: change to paramspec as soon as we drop support for python 3.9, see https://stackoverflow.com/a/68290080
def _deprecation_alias(**aliases: str) -> Callable[[Callable[..., RT]], Callable[..., RT]]:
    """
    Decorate a function to warn user of use of arguments set for deprecation.

    Parameters
    ----------
    aliases
        Deprecation argument aliases to be mapped to the new arguments. Must include version with as value a string
        indicating the version from which the old argument will be deprecated.

    Returns
    -------
    A decorator that can be used to mark an argument for deprecation and substituting it with the new argument.

    Raises
    ------
    TypeError
        If the provided aliases are not of string type.

    Example
    -------
    Assuming we have an argument 'table' set for deprecation and we want to warn the user and substitute with 'tables':

    ```python
    @_deprecation_alias(table="tables", version="0.1.0")
    def my_function(tables: AnnData | dict[str, AnnData]):
        pass
    ```
    """

    def deprecation_decorator(f: Callable[..., RT]) -> Callable[..., RT]:
        @functools.wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> RT:
            class_name = f.__qualname__
            library = f.__module__.split(".")[0]
            alias_copy = aliases.copy()
            version = alias_copy.pop("version") if alias_copy.get("version") is not None else None
            if version is None:
                raise ValueError("version for deprecation must be specified")
            rename_kwargs(f.__name__, kwargs, alias_copy, class_name, library, version)
            return f(*args, **kwargs)

        return wrapper

    return deprecation_decorator


def rename_kwargs(
    func_name: str, kwargs: dict[str, Any], aliases: dict[str, str], class_name: None | str, library: str, version: str
) -> None:
    """Rename function arguments set for deprecation and gives warning in case of usage of these arguments."""
    for alias, new in aliases.items():
        if alias in kwargs:
            class_name = class_name + "." if class_name else ""
            if new in kwargs:
                raise TypeError(
                    f"{class_name}{func_name} received both {alias} and {new} as arguments!"
                    f" {alias} is being deprecated in {library} {version}, only use {new} instead."
                )
            warnings.warn(
                message=(
                    f"`{alias}` is being deprecated as an argument to `{class_name}{func_name}` in {library} version "
                    f"{version}, switch to `{new}` instead."
                ),
                category=DeprecationWarning,
                stacklevel=3,
            )
            kwargs[new] = kwargs.pop(alias)


def _error_message_add_element() -> None:
    raise RuntimeError(
        "The functions add_image(), add_labels(), add_points() and add_shapes() have been removed in favor of "
        "dict-like access to the elements. Please use the following syntax to add an element:\n"
        "\n"
        '\tsdata.images["image_name"] = image\n'
        '\tsdata.labels["labels_name"] = labels\n'
        "\t...\n"
        "\n"
        "The new syntax does not automatically updates the disk storage, so you need to call sdata.write() when "
        "the in-memory object is ready to be saved.\n"
        "To save only a new specific element to an existing Zarr storage please use the functions write_image(), "
        "write_labels(), write_points(), write_shapes() and write_table(). We are going to make these calls more "
        "ergonomic in a follow up PR."
    )


def _check_match_length_channels_c_dim(
    data: DaskArray | DataArray | DataTree, c_coords: str | list[str], dims: tuple[str, ...]
) -> list[str]:
    """
    Check whether channel names `c_coords` are of equal length to the `c` dimension of the data.

    Parameters
    ----------
    data
        The image array
    c_coords
        The channel names
    dims
        The axes names in the order that is the same as the `ImageModel` from which it is derived.

    Returns
    -------
    c_coords
        The channel names as list
    """
    c_index = dims.index("c")
    c_length = (
        data.shape[c_index] if isinstance(data, DataArray | DaskArray) else data["scale0"]["image"].shape[c_index]
    )
    if isinstance(c_coords, str):
        c_coords = [c_coords]
    if c_coords is not None and len(c_coords) != c_length:
        raise ValueError(
            f"The number of channel names `{len(c_coords)}` does not match the length of dimension 'c'"
            f" with length {c_length}."
        )
    return c_coords
