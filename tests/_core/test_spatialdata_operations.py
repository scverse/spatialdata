from anndata import AnnData
from dask.dataframe.core import DataFrame as DaskDataFrame
from dask.delayed import Delayed
from geopandas import GeoDataFrame
from multiscale_spatial_image import MultiscaleSpatialImage
from spatial_image import SpatialImage
import numpy as np

from spatialdata import SpatialData
from spatialdata._core._spatialdata_ops import set_transformation
from spatialdata._core.transformations import Identity, Scale
from spatialdata.utils import get_table_mapping_metadata


def _assert_elements_left_to_right_seem_identical(sdata0: SpatialData, sdata1: SpatialData):
    for element_type, element_name, element in sdata0._gen_elements():
        elements = sdata1.__getattribute__(element_type)
        assert element_name in elements
        element1 = elements[element_name]
        if isinstance(element, AnnData) or isinstance(element, SpatialImage) or isinstance(element, GeoDataFrame):
            assert element.shape == element1.shape
        elif isinstance(element, DaskDataFrame):
            for s0, s1 in zip(element.shape, element1.shape):
                if isinstance(s0, Delayed):
                    s0 = s0.compute()
                if isinstance(s1, Delayed):
                    s1 = s1.compute()
                assert s0 == s1
        elif isinstance(element, MultiscaleSpatialImage):
            assert len(element) == len(element1)
        else:
            raise TypeError(f"Unsupported type {type(element)}")


def _assert_tables_seem_identical(table0: AnnData, table1: AnnData):
    assert table0.shape == table1.shape


def _assert_spatialdata_objects_seem_identical(sdata0: SpatialData, sdata1: SpatialData):
    # this is not a full comparison, but it's fine anyway
    assert len(list(sdata0._gen_elements())) == len(list(sdata1._gen_elements()))
    assert set(sdata0.coordinate_systems) == set(sdata1.coordinate_systems)
    _assert_elements_left_to_right_seem_identical(sdata0, sdata1)
    _assert_elements_left_to_right_seem_identical(sdata1, sdata0)
    _assert_tables_seem_identical(sdata0.table, sdata1.table)


def test_filter_by_coordinate_system(full_sdata):
    sdata = full_sdata.filter_by_coordinate_system(coordinate_system="global")
    _assert_spatialdata_objects_seem_identical(sdata, full_sdata)

    scale = Scale([2.0], axes=("x",))
    set_transformation(full_sdata.images["image2d"], scale, "my_space")
    set_transformation(full_sdata.shapes["shapes_0"], Identity(), "my_space")

    sdata_my_space = full_sdata.filter_by_coordinate_system(coordinate_system="my_space")
    assert len(list(sdata_my_space._gen_elements())) == 2
    _assert_tables_seem_identical(sdata_my_space.table, full_sdata.table)

def test_filter_by_coordinate_system_also_table(full_sdata):
    from spatialdata._core.models import TableModel

    full_sdata.table.obs['annotated_shapes'] = np.random.choice(['shapes/shapes_0', 'shapes/shapes_1'], size=full_sdata.table.shape[0])
    adata = full_sdata.table
    del adata.uns[TableModel.ATTRS_KEY]
    del full_sdata.table
    full_sdata.table = TableModel.parse(adata, region=['shapes/shapes_0', 'shapes/shapes_1'], region_key='annotated_shapes', instance_key='instance_id')

    scale = Scale([2.0], axes=("x",))
    set_transformation(full_sdata.shapes['shapes_0'], scale, 'my_space0')
    set_transformation(full_sdata.shapes['shapes_1'], scale, 'my_space1')

    filtered_sdata0 = full_sdata.filter_by_coordinate_system(coordinate_system="my_space0", filter_table=True)
    filtered_sdata1 = full_sdata.filter_by_coordinate_system(coordinate_system="my_space1", filter_table=True)
    filtered_sdata2 = full_sdata.filter_by_coordinate_system(coordinate_system="my_space0")

    assert len(filtered_sdata0.table) + len(filtered_sdata1.table) == len(full_sdata.table)
    assert len(filtered_sdata2.table) == len(full_sdata.table)
