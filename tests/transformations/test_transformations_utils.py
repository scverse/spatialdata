from spatialdata.transformations._utils import convert_transformations_to_affine
from spatialdata.transformations.operations import get_transformation, set_transformation
from spatialdata.transformations.transformations import Affine, Scale, Sequence, Translation


def test_convert_transformations_to_affine(full_sdata):
    translation = Translation([1, 2, 3], axes=("x", "y", "z"))
    scale = Scale([1, 2, 3], axes=("x", "y", "z"))
    sequence = Sequence([translation, scale])
    for _, _, element in full_sdata.gen_spatial_elements():
        set_transformation(element, transformation=sequence, to_coordinate_system="test")
    convert_transformations_to_affine(full_sdata, "test")
    for _, _, element in full_sdata.gen_spatial_elements():
        t = get_transformation(element, "test")
        assert isinstance(t, Affine)
