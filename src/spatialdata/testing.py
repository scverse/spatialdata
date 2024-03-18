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
from spatialdata.transformations.operations import get_transformation


def assert_elements_dict_are_identical(
    elements0: Elements, elements1: Elements, check_transformations: bool = True
) -> None:
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

    Notes
    -----
    With the current implementation, the transformations Translate([1.0, 2.0],
    axes=('x', 'y')) and Translate([2.0, 1.0], axes=('y', 'x')) are considered different.
    A quick way to avoid an error in this case is to use the check_transformations=False parameter.
    """
    assert set(elements0.keys()) == set(elements1.keys())
    for k in elements0:
        element0 = elements0[k]
        element1 = elements1[k]
        assert_elements_are_identical(element0, element1, check_transformations=check_transformations)


def assert_elements_are_identical(
    element0: SpatialElement | AnnData, element1: SpatialElement | AnnData, check_transformations: bool = True
) -> None:
    """
    Compare two elements (two SpatialElements or two tables) and assert that they are identical.

    Parameters
    ----------
    element0
        The first element.
    element1
        The second element.
    check_transformations
        Whether to check if the transformations are identical, for each element.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the two elements are not identical.

    Notes
    -----
    With the current implementation, the transformations Translate([1.0, 2.0],
    axes=('x', 'y')) and Translate([2.0, 1.0], axes=('y', 'x')) are considered different.
    A quick way to avoid an error in this case is to use the check_transformations=False parameter.
    """
    assert type(element0) == type(element1)

    # compare transformations (only for SpatialElements)
    if not isinstance(element0, AnnData):
        transformations0 = get_transformation(element0, get_all=True)
        transformations1 = get_transformation(element1, get_all=True)
        assert isinstance(transformations0, dict)
        assert isinstance(transformations1, dict)
        if check_transformations:
            assert transformations0.keys() == transformations1.keys()
            for key in transformations0:
                assert (
                    transformations0[key] == transformations1[key]
                ), f"transformations0[{key}] != transformations1[{key}]"

    # compare the elements
    if isinstance(element0, AnnData):
        assert_anndata_equal(element0, element1)
    elif isinstance(element0, SpatialImage):
        assert_xarray_equal(element0, element1)
    elif isinstance(element0, MultiscaleSpatialImage):
        assert_datatree_equal(element0, element1)
    elif isinstance(element0, GeoDataFrame):
        assert_geodataframe_equal(element0, element1, check_less_precise=True)
    else:
        assert isinstance(element0, DaskDataFrame)
        assert_dask_dataframe_equal(element0, element1)


def assert_spatial_data_objects_are_identical(
    sdata0: SpatialData, sdata1: SpatialData, check_transformations: bool = True
) -> None:
    """
    Compare two SpatialData objects and assert that they are identical.

    Parameters
    ----------
    sdata0
        The first SpatialData object.
    sdata1
        The second SpatialData object.
    check_transformations
        Whether to check if the transformations are identical, for each element.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the two SpatialData objects are not identical.

    Notes
    -----
    With the current implementation, the transformations Translate([1.0, 2.0],
    axes=('x', 'y')) and Translate([2.0, 1.0], axes=('y', 'x')) are considered different.
    A quick way to avoid an error in this case is to use the check_transformations=False parameter.
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
        assert_elements_are_identical(element0, element1, check_transformations=check_transformations)
