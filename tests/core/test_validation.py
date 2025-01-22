import pytest

from spatialdata._core.validation import ValidationError, raise_validation_errors


def test_raise_validation_errors():
    with pytest.raises(expected_exception=ValidationError, match="Some errors happened") as actual_exc_info:
        ...
        with raise_validation_errors("Some errors happened", exc_type=ValueError) as collect_error:
            with collect_error(expected_exception=TypeError):
                raise TypeError("Another error type")
            for key, value in {"first": 1, "second": 2, "third": 3}.items():
                with collect_error(location=key):
                    if value % 2 != 0:
                        raise ValueError("Odd value encountered")
    actual_message = str(actual_exc_info.value)
    assert "Another error" in actual_message
    assert "first" in actual_message
    assert "second" not in actual_message
    assert "third" in actual_message


def test_raise_validation_errors_does_not_catch_other_errors():
    with pytest.raises(expected_exception=RuntimeError, match="Not to be collected"):
        ...
        with raise_validation_errors(exc_type=ValueError) as collect_error:
            ...
            with collect_error:
                raise RuntimeError("Not to be collected as ValidationError")
