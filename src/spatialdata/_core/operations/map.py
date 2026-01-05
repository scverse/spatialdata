import math
import operator
from collections.abc import Callable, Iterable, Mapping
from functools import reduce
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import dask.array as da
import numpy as np
from dask.array.overlap import coerce_depth
from xarray import DataArray, DataTree

from spatialdata._types import IntArrayLike
from spatialdata.models._utils import get_axes_names, get_channel_names, get_raster_model_from_data_dims
from spatialdata.transformations import get_transformation

__all__ = ["map_raster", "relabel_sequential"]


def map_raster(
    data: DataArray | DataTree,
    func: Callable[[da.Array], da.Array],
    func_kwargs: Mapping[str, Any] = MappingProxyType({}),
    blockwise: bool = True,
    depth: int | tuple[int, ...] | dict[int, int] | None = None,
    chunks: tuple[tuple[int, ...], ...] | None = None,
    c_coords: Iterable[int] | Iterable[str] | None = None,
    dims: tuple[str, ...] | None = None,
    transformations: dict[str, Any] | None = None,
    relabel: bool = True,
    **kwargs: Any,
) -> DataArray:
    """
    Apply a callable to raster data.

    Applies a `func` callable to raster data. If `blockwise` is set to `True`,
    distributed processing will be achieved with:

        - :func:`dask.array.map_overlap` if `depth` is not `None`
        - :func:`dask.array.map_blocks`, if `depth` is `None`

    otherwise `func` is applied to the full data.

    Parameters
    ----------
    data
        The data to process. It can be a :class:`xarray.DataArray` or :class:`datatree.DataTree`.
        If it's a `DataTree`, the callable is applied to the first scale (`scale0`, the full-resolution data).
    func
        The callable that is applied to the data.
    func_kwargs
        Additional keyword arguments to pass to the callable `func`.
    blockwise
        If `True`, `func` will be distributed with :func:`dask.array.map_overlap` or :func:`dask.array.map_blocks`,
        otherwise `func` is applied to the full data. If `False`, `depth`, `chunks` and `kwargs` are ignored.
    depth
        Specifies the overlap between chunks, i.e. the number of elements that each chunk
        should share with its neighboring chunks. If not `None`, distributed processing will be achieved with
        :func:`dask.array.map_overlap`, otherwise with :func:`dask.array.map_blocks`.
    chunks
        Chunk shape of resulting blocks if the callable does not preserve the data shape.
        For example, if the input block has `shape: (3,100,100)` and the resulting block after the `map_raster`
        call has `shape: (1, 100,100)`, the argument `chunks` should be passed accordingly.
        Passed to :func:`dask.array.map_overlap` or :func:`dask.array.map_blocks`. Ignored if `blockwise` is `False`.
    c_coords
        The channel coordinates for the output data. If not provided, the channel coordinates of the input data are
        used. If the callable `func` is expected to change the number of channel coordinates,
        this argument should be provided, otherwise will default to `range(len(output_coords))`.
    dims
        The dimensions of the output data. If not provided, the dimensions of the input data are used. It must be
        specified if the callable changes the data dimensions, e.g. `('c', 'y', 'x') -> ('y', 'x')`.
    transformations
        The transformations of the output data. If not provided, the transformations of the input data are copied to the
        output data. It should be specified if the callable changes the data transformations.
    relabel
        Whether to relabel the blocks of the output data.
        This option is ignored when the output data is not a labels layer (i.e., when `dims` does not contain `c`).
        It is recommended to enable relabeling if `func` returns labels that are not unique across chunks.
        Relabeling will be done by performing a bit shift. When a cell or entity to be labeled is split between two
        adjacent chunks, the current implementation does not assign the same label across blocks.
        See https://github.com/scverse/spatialdata/pull/664 for discussion.
    kwargs
        Additional keyword arguments to pass to :func:`dask.array.map_overlap` or :func:`dask.array.map_blocks`.
        Ignored if `blockwise` is set to `False`.

    Returns
    -------
    The processed data as a :class:`xarray.DataArray`.
    """
    if isinstance(data, DataArray):
        arr = data.data
    elif isinstance(data, DataTree):
        arr = data["scale0"].values().__iter__().__next__().data
    else:
        raise ValueError("Only 'DataArray' and 'DataTree' are supported.")

    dims = dims if dims is not None else get_axes_names(data)

    if "c" not in dims and c_coords is not None:
        raise ValueError(
            "Channel coordinates `c_coords` can not be provided if output data consists of labels "
            "('c' channel missing)."
        )

    kwargs = kwargs.copy()
    kwargs["chunks"] = chunks

    if not blockwise:
        arr = func(arr, **func_kwargs)
    else:
        if depth is not None:
            kwargs.setdefault("boundary", "reflect")

            if not isinstance(depth, int) and len(depth) != arr.ndim:
                raise ValueError(
                    f"Depth {depth} is provided for {len(depth)} dimensions. "
                    f"Please provide depth for {arr.ndim} dimensions."
                )
            kwargs["depth"] = coerce_depth(arr.ndim, depth)
            map_func = da.map_overlap
        else:
            map_func = da.map_blocks

        arr = map_func(func, arr, **func_kwargs, **kwargs)

    if arr.ndim != len(dims):
        raise ValueError(
            f"The number of dimensions of the output data ({arr.ndim}) "
            f"differs from the number of dimensions in 'dims' ({dims}). "
            "Please provide correct output dimension via the 'dims' parameter."
        )

    if "c" in dims:
        if c_coords is None:
            c_coords = range(arr.shape[0]) if arr.shape[0] != len(get_channel_names(data)) else get_channel_names(data)
    else:
        c_coords = None
    if transformations is None:
        d = get_transformation(data, get_all=True)
        if TYPE_CHECKING:
            assert isinstance(d, dict)
        transformations = d

    if "c" not in dims and relabel:
        arr = _relabel(arr)

    model_kwargs = {
        "chunks": arr.chunksize,
        "c_coords": c_coords,
        "dims": dims,
        "transformations": transformations,
    }
    model = get_raster_model_from_data_dims(dims)
    return model.parse(arr, **model_kwargs)


def _relabel(arr: da.Array) -> da.Array:
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"Relabeling is only supported for arrays of type {np.integer}.")
    num_blocks = arr.numblocks

    shift = (math.prod(num_blocks) - 1).bit_length()

    meta = np.empty((0,) * arr.ndim, dtype=arr.dtype)

    def _relabel_block(
        block: IntArrayLike, block_id: tuple[int, ...], num_blocks: tuple[int, ...], shift: int
    ) -> IntArrayLike:
        def _calculate_block_num(block_id: tuple[int, ...], num_blocks: tuple[int, ...]) -> int:
            if len(num_blocks) != len(block_id):
                raise ValueError("num_blocks and block_id must have the same length")
            block_num = 0
            for i in range(len(num_blocks)):
                multiplier = reduce(operator.mul, num_blocks[i + 1 :], 1)
                block_num += block_id[i] * multiplier
            return block_num

        available_bits = np.iinfo(block.dtype).max.bit_length()
        max_bits_block = int(block.max()).bit_length()

        if max_bits_block + shift > available_bits:
            # Note: because of no harmonization across blocks, adjusting number of chunks lowers the required bits.
            raise ValueError(
                f"Relabel was set to True, but "
                f"the number of bits required to represent the labels in the block ({max_bits_block}) "
                f"+ required shift ({shift}) exceeds the available_bits ({available_bits}). In other words"
                f"the number of labels exceeds the number of integers that can be represented by the dtype"
                "of the individual blocks."
                "To solve this issue, please consider the following solutions:"
                "   1. Rechunking using a larger chunk size, lowering the number of blocks and thereby"
                "      lowering the value of required shift."
                "   2. Cast to a data type with a higher maximum value  "
                "   3. Perform sequential relabeling of the dask array using `relabel_sequential` in `spatialdata`,"
                "      potentially lowering the maximum value of a label (though number of distinct labels values "
                "      stays the same). For example if the unique labels values are `[0, 1, 1000]`, after the "
                "      sequential relabeling the unique labels value will be `[0, 1, 2]`, thus requiring less bits "
                "      to store the labels."
            )

        block_num = _calculate_block_num(block_id=block_id, num_blocks=num_blocks)

        mask = block > 0
        block[mask] = (block[mask] << shift) | block_num

        return block

    return da.map_blocks(
        _relabel_block,
        arr,
        dtype=arr.dtype,
        num_blocks=num_blocks,
        shift=shift,
        meta=meta,
    )


def relabel_sequential(arr: da.Array) -> da.Array:
    """
    Relabels integers in a Dask array sequentially.

    This function assigns sequential labels to the integers in a Dask array starting from 1.
    For example, if the unique values in the input array are [0, 9, 5],
    they will be relabeled to [0, 1, 2] respectively.
    Note that currently if a cell or entity to be labeled is split across adjacent chunks the same label is not
    assigned to the cell across blocks. See discussion https://github.com/scverse/spatialdata/pull/664.

    Parameters
    ----------
    arr
        input array.

    Returns
    -------
    The relabeled array.
    """
    if not np.issubdtype(arr.dtype, np.integer):
        raise ValueError(f"Sequential relabeling is only supported for arrays of type {np.integer}.")

    unique_labels = da.unique(arr).compute()
    if 0 not in unique_labels:
        # otherwise first non zero label would be relabeled to 0
        unique_labels = np.insert(unique_labels, 0, 0)

    max_label = unique_labels[-1]

    new_labeling = da.full(max_label + 1, -1, dtype=arr.dtype)

    # Note that both sides are ordered as da.unique returns an ordered array.
    new_labeling[unique_labels] = da.arange(len(unique_labels), dtype=arr.dtype)

    return da.map_blocks(operator.getitem, new_labeling, arr, dtype=arr.dtype, chunks=arr.chunks)
