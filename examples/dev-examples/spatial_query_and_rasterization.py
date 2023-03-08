import numpy as np
from spatial_image import SpatialImage

from spatialdata import Labels2DModel
from spatialdata._core._spatial_query import bounding_box_query
from spatialdata._core._spatialdata_ops import (
    get_transformation,
    remove_transformation,
    set_transformation,
)
from spatialdata._core.transformations import Affine, Scale, Sequence


def _visualize_crop_affine_labels_2d() -> None:
    """
    This examples show how the bounding box spatial query works for data that has been rotated.

    Notes
    -----
    The bounding box query gives the data, from the intrinsic coordinate system, that is inside the bounding box of
    the inverse-transformed query bounding box.
    In this example I show this data, and I also show how to obtain the data back inside the original bounding box.

    To undertand the example I suggest to run it and then:
    1) select the "rotated" coordinate system from napari
    2) disable all the layers but "0 original"
    3) then enable "1 cropped global", this shows the data in the extrinsic coordinate system we care ("rotated"),
    and the bounding box we want to query
    4) then enable "2 cropped rotated", this show the data that has been queries (this is a bounding box of the
    requested crop, as exaplained above)
    5) then enable "3 cropped rotated processed", this shows the data that we wanted to query in the first place,
    in the target coordinate system ("rotated"). This is probaly the data you care about if for instance you want to
    use tiles for deep learning.
    6) Note that for obtaning the previous answer there is also a better function rasterize().
    This is what "4 rasterized" shows, which is faster and more accurate, so it should be used instead. The function
    rasterize() transforms all the coordinates of the data into the target coordinate system, and it returns only
    SpatialImage objects. So it has different use cases than the bounding box query. BUG: Note that it is not pixel
     perfect. I think this is due to the difference between considering the origin of a pixel its center or its corner.
    7) finally switch to the "global" coordinate_system. This is, for how we constructed the example, showing the
    original image as it would appear its intrinsic coordinate system (since the transformation that maps the
    original image to "global" is an identity. It then shows how the data showed at the point 5), localizes in the
    original image.
    """
    ##
    # in this test let's try some affine transformations, we could do that also for the other tests
    # image = scipy.misc.face()[:100, :100, :].copy()
    image = np.random.randint(low=10, high=100, size=(100, 100))
    multiscale_image = np.repeat(np.repeat(image, 4, axis=0), 4, axis=1)

    # y: [5, 9], x: [0, 4] has value 1
    image[50:, :50] = 2
    # labels_element = Image2DModel.parse(image, dims=('y', 'x', 'c'))
    labels_element = Labels2DModel.parse(image)
    affine = Affine(
        np.array(
            [
                [np.cos(np.pi / 6), np.sin(-np.pi / 6), 0],
                [np.sin(np.pi / 6), np.cos(np.pi / 6), 0],
                [0, 0, 1],
            ]
        ),
        input_axes=("x", "y"),
        output_axes=("x", "y"),
    )
    set_transformation(
        labels_element,
        affine,
        "rotated",
    )

    # bounding box: y: [5, 9], x: [0, 4]
    labels_result_rotated = bounding_box_query(
        labels_element,
        axes=("y", "x"),
        min_coordinate=np.array([25, 25]),
        max_coordinate=np.array([75, 100]),
        target_coordinate_system="rotated",
    )
    labels_result_global = bounding_box_query(
        labels_element,
        axes=("y", "x"),
        min_coordinate=np.array([25, 25]),
        max_coordinate=np.array([75, 100]),
        target_coordinate_system="global",
    )
    from napari_spatialdata import Interactive

    from spatialdata import SpatialData

    old_transformation = get_transformation(labels_result_global, "global")
    remove_transformation(labels_result_global, "global")
    set_transformation(labels_result_global, old_transformation, "rotated")
    d = {
        "1 cropped_global": labels_result_global,
        "0 original": labels_element,
    }
    if labels_result_rotated is not None:
        d["2 cropped_rotated"] = labels_result_rotated

        assert isinstance(labels_result_rotated, SpatialImage)
        transform = labels_result_rotated.attrs["transform"]["rotated"]
        transform_rotated_processed = transform.transform(labels_result_rotated, maintain_positioning=True)
        transform_rotated_processed_recropped = bounding_box_query(
            transform_rotated_processed,
            axes=("y", "x"),
            min_coordinate=np.array([25, 25]),
            max_coordinate=np.array([75, 100]),
            target_coordinate_system="rotated",
        )
        d["3 cropped_rotated_processed_recropped"] = transform_rotated_processed_recropped
        remove_transformation(labels_result_rotated, "global")

    multiscale_image[200:, :200] = 2
    # multiscale_labels = Labels2DModel.parse(multiscale_image)
    multiscale_labels = Labels2DModel.parse(multiscale_image, scale_factors=[2, 2, 2, 2])
    sequence = Sequence([Scale([0.25, 0.25], axes=("x", "y")), affine])
    set_transformation(multiscale_labels, sequence, "rotated")

    from spatialdata._core._rasterize import rasterize

    rasterized = rasterize(
        multiscale_labels,
        axes=("y", "x"),
        min_coordinate=np.array([25, 25]),
        max_coordinate=np.array([75, 100]),
        target_coordinate_system="rotated",
        target_width=300,
    )
    d["4 rasterized"] = rasterized

    sdata = SpatialData(labels=d)

    # to see only what matters when debugging https://github.com/scverse/spatialdata/issues/165
    del sdata.labels["1 cropped_global"]
    del sdata.labels["2 cropped_rotated"]
    del sdata.labels["3 cropped_rotated_processed_recropped"]
    del sdata.labels["0 original"].attrs["transform"]["global"]

    Interactive(sdata)
    ##


if __name__ == "__main__":
    _visualize_crop_affine_labels_2d()
