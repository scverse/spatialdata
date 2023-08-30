import pytest
from spatialdata.datasets import blobs


def test_spatialdata_plot():
    """Test whether the current state breaks spatialdata-plot."""

    import spatialdata_plot

    ax_blobs = (
        blobs()
        .pl.render_images()
        .pl.render_labels()
        .pl.render_shapes()
        .pl.render_points()
        .pl.show()
    )

    assert ax_blobs is not None
