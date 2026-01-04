# type: ignore
"""
This utility module contains functions that are used in the benchmarks.

Functions that make running and debugging benchmarks easier:
- class Skip is used to skip benchmarks based on environment variables.
- function run_benchmark_from_module is used to run benchmarks from a module.
- function run_benchmark is used to run the benchmarks.

Performant dataset generation functions so the benchmarks run fast even for large artificial datasets.
The object is to generate a dataset containing many cells. By copying the same cell values instead of
doing gaussian blur on the whole image, we can generate the same dataset in a fraction of the time.
- function labeled_particles is used to generate labeled blobs.
- function _generate_ball is used to generate a ball of given radius and dimension.
- function _generate_density is used to generate gaussian density of given radius and dimension.
- function cluster_blobs is used to generate a SpatialData object with blobs.
- function _structure_at_coordinates is used to update data with structure at given coordinates.
- function _get_slices_at is used to get slices at a given point.
- function _update_data_with_mask is used to update data with struct where struct is nonzero.
"""

import itertools
import os
from collections.abc import Sequence
from functools import lru_cache
from types import ModuleType
from typing import Callable, Literal, overload

import anndata as ad
import numpy as np
import pandas as pd
from skimage import morphology

import spatialdata as sd
from spatialdata import SpatialData
from spatialdata.models import Image2DModel, TableModel
from spatialdata.transformations import Identity


def always_false(*_):
    return False


class Skip:
    def __init__(
        self,
        if_in_pr: Callable[..., bool] = always_false,
        if_on_ci: Callable[..., bool] = always_false,
        always: Callable[..., bool] = always_false,
    ):
        self.func_pr = if_in_pr if "PR" in os.environ else always_false
        self.func_ci = if_on_ci if "CI" in os.environ else always_false
        self.func_always = always

    def __contains__(self, item):
        return self.func_pr(*item) or self.func_ci(*item) or self.func_always(*item)


def _generate_ball(radius: int, ndim: int) -> np.ndarray:
    """Generate a ball of given radius and dimension.

    Parameters
    ----------
    radius : int
        Radius of the ball.
    ndim : int
        Dimension of the ball.

    Returns
    -------
    ball : ndarray of uint8
        Binary array of the hyper ball.
    """
    if ndim == 2:
        return morphology.disk(radius)
    if ndim == 3:
        return morphology.ball(radius)
    shape = (2 * radius + 1,) * ndim
    radius_sq = radius**2
    coords = np.indices(shape) - radius
    return (np.sum(coords**2, axis=0) <= radius_sq).astype(np.uint8)


def _generate_density(radius: int, ndim: int) -> np.ndarray:
    """Generate gaussian density of given radius and dimension."""
    shape = (2 * radius + 1,) * ndim
    coords = np.indices(shape) - radius
    dist = np.sqrt(np.sum(coords**2 / ((radius / 4) ** 2), axis=0))
    res = np.exp(-dist)
    res[res < 0.02] = 0
    return res


def _structure_at_coordinates(
    shape: tuple[int],
    coordinates: np.ndarray,
    structure: np.ndarray,
    *,
    multipliers: Sequence = itertools.repeat(1),
    dtype=None,
    reduce_fn: Callable[[np.ndarray, np.ndarray, np.ndarray | None], np.ndarray],
):
    """Update data with structure at given coordinates.

    Parameters
    ----------
    data : ndarray
        Array to update.
    coordinates : ndarray
        Coordinates of the points. The structures will be added at these
        points (center).
    structure : ndarray
        Array with encoded structure. For example, ball (boolean) or density
        (0,1) float.
    multipliers : ndarray
        These values are multiplied by the values in the structure before
        updating the array. Can be used to generate different labels, or to
        vary the intensity of floating point gaussian densities.
    reduce_fn : function
        Function with which to update the array at a particular position. It
        should take two arrays as input and an optional output array.
    """
    radius = (structure.shape[0] - 1) // 2
    data = np.zeros(shape, dtype=dtype)

    for point, value in zip(coordinates, multipliers):
        slice_im, slice_ball = _get_slices_at(shape, point, radius)
        reduce_fn(data[slice_im], value * structure[slice_ball], out=data[slice_im])
    return data


def _get_slices_at(shape, point, radius):
    slice_im = []
    slice_ball = []
    for i, p in enumerate(point):
        slice_im.append(slice(max(0, p - radius), min(shape[i], p + radius + 1)))
        ball_start = max(0, radius - p)
        ball_stop = slice_im[-1].stop - slice_im[-1].start + ball_start
        slice_ball.append(slice(ball_start, ball_stop))
    return tuple(slice_im), tuple(slice_ball)


def _update_data_with_mask(data, struct, out=None):
    """Update ``data`` with ``struct`` where ``struct`` is nonzero."""
    # these branches are needed because np.where does not support
    # an out= keyword argument
    if out is None:
        return np.where(struct, struct, data)
    else:  # noqa: RET505
        nz = struct != 0
        out[nz] = struct[nz]
        return out


def _smallest_dtype(n: int) -> np.dtype:
    """Find the smallest dtype that can hold n values."""
    for dtype in [np.uint8, np.uint16, np.uint32, np.uint64]:
        if np.iinfo(dtype).max >= n:
            return dtype
            break
    else:
        raise ValueError(f"{n=} is too large for any dtype.")


@overload
def labeled_particles(
    shape: Sequence[int],
    dtype: np.dtype | None = None,
    n: int = 144,
    seed: int | None = None,
    return_density: Literal[False] = False,
) -> np.ndarray: ...


@overload
def labeled_particles(
    shape: Sequence[int],
    dtype: np.dtype | None = None,
    n: int = 144,
    seed: int | None = None,
    return_density: Literal[True] = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]: ...


@lru_cache
def labeled_particles(
    shape: Sequence[int],
    dtype: np.dtype | None = None,
    n: int = 144,
    seed: int | None = None,
    return_density: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate labeled blobs of given shape and dtype.

    Parameters
    ----------
    shape : Sequence[int]
        Shape of the resulting array.
    dtype : np.dtype | None
        Dtype of the resulting array.
    n : int
        Number of blobs to generate.
    seed : int | None
        Seed for the random number generator.
    return_density : bool
        Whether to return the density array and center coordinates.
    """
    if dtype is None:
        dtype = _smallest_dtype(n)
    rng = np.random.default_rng(seed)
    ndim = len(shape)
    points = rng.integers(shape, size=(n, ndim))
    # create values from 1 to max of number of points
    values = np.linspace(1, n, n, dtype=dtype)
    rng.shuffle(values)
    # values = rng.integers(
    #     np.iinfo(dtype).min + 1, np.iinfo(dtype).max, size=n, dtype=dtype
    # )
    sigma = int(max(shape) / (4.0 * n ** (1 / ndim)))
    ball = _generate_ball(sigma, ndim)

    labels = _structure_at_coordinates(
        shape,
        points,
        ball,
        multipliers=values,
        reduce_fn=_update_data_with_mask,
        dtype=dtype,
    )

    if return_density:
        dens = _generate_density(sigma * 2, ndim)
        densities = _structure_at_coordinates(shape, points, dens, reduce_fn=np.maximum, dtype=np.float32)

        return labels, densities, points, values
    else:  # noqa: RET505
        return labels


def run_benchmark_from_module(module: ModuleType, klass_name: str, method_name: str):
    klass = getattr(module, klass_name)
    if getattr(klass, "params", None):
        skip_if = getattr(klass, "skip_params", {})
        if isinstance(klass.params[0], Sequence):
            params = itertools.product(*klass.params)
        else:
            params = ((i,) for i in klass.params)
        for param in params:
            if param in skip_if:
                continue
            obj = klass()
            try:
                obj.setup(*param)
            except NotImplementedError:
                continue
            getattr(obj, method_name)(*param)
            getattr(obj, "teardown", lambda: None)()
    else:
        obj = klass()
        try:
            obj.setup()
        except NotImplementedError:
            return
        getattr(obj, method_name)()
        getattr(obj, "teardown", lambda: None)()


def run_benchmark():
    import argparse
    import inspect

    parser = argparse.ArgumentParser(description="Run benchmark")
    parser.add_argument("benchmark", type=str, help="Name of the benchmark to run", default="")

    args = parser.parse_args()

    benchmark_selection = args.benchmark.split(".")

    # get module of parent frame
    call_module = inspect.getmodule(inspect.currentframe().f_back)
    run_benchmark_from_module(call_module, *benchmark_selection)


# TODO: merge functionality of this cluster_blobs with the one in SpatialData https://github.com/scverse/spatialdata/issues/796
@lru_cache
def cluster_blobs(
    length=512,
    n_cells=None,
    region_key="region_key",
    instance_key="instance_key",
    image_name="blobs_image",
    labels_name="blobs_labels",
    points_name="blobs_points",
    n_transcripts_per_cell=None,
    table_name="table",
    coordinate_system="global",
):
    """Faster `spatialdata.datasets.make_blobs` using napari.datasets code."""
    if n_cells is None:
        n_cells = length
    # cells
    labels, density, points, values = labeled_particles((length, length), return_density=True, n=n_cells)

    im_el = Image2DModel.parse(
        data=density[None, ...],
        dims="cyx",
        transformations={coordinate_system: Identity()},
    )
    label_el = sd.models.Labels2DModel.parse(labels, dims="yx", transformations={coordinate_system: Identity()})
    points_cells_el = sd.models.PointsModel.parse(points, transformations={coordinate_system: Identity()})

    # generate dummy table
    adata = ad.AnnData(X=np.ones((length, 10)))
    adata.obs[region_key] = pd.Categorical([labels_name] * len(adata))
    # adata.obs_names = values.astype(np.uint64)
    adata.obs[instance_key] = adata.obs_names.values
    adata.obs.index = adata.obs.index.astype(str)
    adata.obs.index.name = instance_key
    # del adata.uns[TableModel.ATTRS_KEY]
    table = TableModel.parse(
        adata,
        region=labels_name,
        region_key=region_key,
        instance_key=instance_key,
    )

    sdata = SpatialData(
        images={
            image_name: im_el,
        },
        labels={
            labels_name: label_el,
        },
        points={points_name: points_cells_el},
        tables={table_name: table},
    )

    if n_transcripts_per_cell:
        # transcript points
        # generate 100 transcripts per cell
        rng = np.random.default_rng(None)
        points_transcripts = rng.integers(length, size=(n_cells * n_transcripts_per_cell, 2))
        points_transcripts_el = sd.models.PointsModel.parse(
            points_transcripts, transformations={coordinate_system: Identity()}
        )
        sdata["transcripts_" + points_name] = points_transcripts_el

    # if shapes_name:
    #     sdata[shapes_name] = sd.to_circles(sdata[labels_name])
    # add_regionprop_features(sdata, labels_layer=labels_name, table_layer=table_name)
    return sdata
