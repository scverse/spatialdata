from functools import singledispatch
from typing import Any, Optional

import anndata as ad
import numpy as np
import xarray as xr

from spatialdata._types import ArrayLike


class Transform:
    def __init__(
        self,
        translation: Optional[ArrayLike] = None,
        scale_factors: Optional[ArrayLike] = None,
        ndim: Optional[int] = None,
    ):
        """
        class for storing transformations, required to align the different spatial data layers.
        """
        if translation is None:
            if scale_factors is not None:
                translation = np.zeros(len(scale_factors), dtype=float)
            elif ndim is not None:
                translation = np.zeros(ndim, dtype=float)
            else:
                raise ValueError("Either translation or scale_factors or ndim must be specified")

        if scale_factors is None:
            if translation is not None:
                scale_factors = np.ones(len(translation), dtype=float)
            elif ndim is not None:
                scale_factors = np.ones(ndim, dtype=float)
            else:
                raise ValueError("Either translation or scale_factors or ndim must be specified")

        if ndim is not None:
            if translation is not None:
                assert len(translation) == ndim
            if scale_factors is not None:
                assert len(scale_factors) == ndim

        self.translation = translation
        self.scale_factors = scale_factors
        self.ndim = ndim


@singledispatch
def set_transform(arg: Any, transform: Optional[Transform]) -> None:
    raise ValueError(f"Unsupported type: {type(arg)}")


@set_transform.register
def _(arg: xr.DataArray, transform: Optional[Transform]) -> None:
    arg.attrs["transform"] = transform


@set_transform.register
def _(arg: ad.AnnData, transform: Optional[Transform]) -> None:
    if transform is not None:
        arg.uns["transform"] = {
            "translation": transform.translation,
            "scale_factors": transform.scale_factors,
        }  # TODO: do we save it like this in uns?


@singledispatch
def get_transform(arg: Any) -> Transform:
    raise ValueError(f"Unsupported type: {type(arg)}")


@get_transform.register
def _(arg: xr.DataArray) -> Transform:
    return Transform(
        translation=arg.attrs["transform"].translation,
        scale_factors=arg.attrs["transform"].scale_factors,
        ndim=arg.ndim,
    )


@get_transform.register
def _(arg: np.ndarray) -> Transform:  # type: ignore[type-arg]
    return Transform(
        ndim=arg.ndim,
    )


@get_transform.register
def _(arg: ad.AnnData) -> Transform:
    # check if the AnnData has transform information, otherwise fetches it from default scanpy storage
    if "transform" in arg.uns:
        return Transform(
            translation=arg.uns["transform"]["translation"], scale_factors=arg.uns["transform"]["scale_factors"]
        )
    elif "spatial" in arg.uns and "spatial" in arg.obsm:
        ndim = arg.obsm["spatial"].shape[1]
        libraries = arg.uns["spatial"]
        assert len(libraries) == 1
        library = libraries.__iter__().__next__()
        scale_factors = library["scalefactors"]["tissue_hires_scalef"]
        return Transform(
            translation=np.zeros(ndim, dtype=float), scale_factors=np.array([scale_factors] * ndim, dtype=float)
        )
    else:
        return Transform(ndim=2)
