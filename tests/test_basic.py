import pytest

import spatialdata


def test_package_has_version():  # type: ignore[no-untyped-def]
    spatialdata.__version__


@pytest.mark.skip(reason="This decorator should be removed when test passes.")
def test_example():  # type: ignore[no-untyped-def]
    assert 1 == 0  # type: ignore[comparison-overlap]
