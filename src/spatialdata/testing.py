from __future__ import annotations

from anndata import AnnData
from anndata.tests.helpers import assert_equal as assert_anndata_equal
from dask.dataframe import DataFrame as DaskDataFrame
from dask.dataframe.tests.test_dataframe import assert_eq as assert_dask_dataframe_equal
from datatree.testing import assert_equal as assert_datatree_equal
from geopandas import GeoDataFrame
from geopandas.testing import assert_geodataframe_equal
from multiscale_spatial_image import MultiscaleSpatialImage
from spatial_image import SpatialImage
from xarray.testing import assert_equal as assert_xarray_equal

from spatialdata import SpatialData
from spatialdata._core._elements import Elements
from spatialdata.models._utils import SpatialElement


def assert_elements_dict_are_identical(elements0: Elements, elements1: Elements) -> None:
    """
    Compare two dictionaries of elements and assert that they are identical (except for the order of the keys).

    The dictionaries of elements can be obtained from a SpatialData object using the `.shapes`, `.labels`, `.points`,
    `.images` and `.tables` properties.

    Parameters
    ----------
    elements0
        The first dictionary of elements.
    elements1
        The second dictionary of elements.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the two dictionaries of elements are not identical.
    """
    assert set(elements0.keys()) == set(elements1.keys())
    for k in elements0:
        element0 = elements0[k]
        element1 = elements1[k]
        assert_elements_are_identical(element0, element1)


def assert_elements_are_identical(element0: SpatialElement | AnnData, element1: SpatialElement | AnnData) -> None:
    """
    Compare two elements (two SpatialElements or two tables) and assert that they are identical.

    Parameters
    ----------
    element0
        The first element.
    element1
        The second element.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the two elements are not identical.
    """
    assert type(element0) == type(element1)
    if isinstance(element0, AnnData):
        assert_anndata_equal(element0, element1)
    elif isinstance(element0, SpatialImage):
        assert_xarray_equal(element0, element1)
    elif isinstance(element0, MultiscaleSpatialImage):
        assert_datatree_equal(element0, element1)
    elif isinstance(element0, GeoDataFrame):
        assert_geodataframe_equal(element0, element1)
    else:
        assert isinstance(element0, DaskDataFrame)
        assert_dask_dataframe_equal(element0, element1)


def assert_spatial_data_objects_are_identical(sdata0: SpatialData, sdata1: SpatialData) -> None:
    """
    Compare two SpatialData objects and assert that they are identical.

    Parameters
    ----------
    sdata0
        The first SpatialData object.
    sdata1
        The second SpatialData object.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the two SpatialData objects are not identical.
    """
    # this is not a full comparison, but it's fine anyway
    element_names0 = [element_name for _, element_name, _ in sdata0.gen_elements()]
    element_names1 = [element_name for _, element_name, _ in sdata1.gen_elements()]
    assert len(set(element_names0)) == len(element_names0)
    assert len(set(element_names1)) == len(element_names1)
    assert set(sdata0.coordinate_systems) == set(sdata1.coordinate_systems)
    for element_name in element_names0:
        element0 = sdata0[element_name]
        element1 = sdata1[element_name]
        assert_elements_are_identical(element0, element1)
