import types

try:
    import spatialdata_io as io
except ImportError:

    io = types.ModuleType("spatialdata_io")

    def _raise_install_spatialdata_io() -> None:
        raise AttributeError(
            "To use readers, `spatialdata_io` must be installed, e.g. via `pip install spatialdata-io`."
        )

    io.__getattr__ = lambda *_: _raise_install_spatialdata_io()

__all__ = ["io"]
