def _check_geopandas_using_shapely() -> None:
    """Check if geopandas is using shapely.

    This is required until it becomes the default.
    """
    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            "Shapely 2.0 is installed, but because PyGEOS is also installed",
            category=UserWarning,
        )
        import geopandas

        if geopandas.options.use_pygeos is True:
            geopandas.options.use_pygeos = False

            warnings.warn(
                (
                    "Geopandas was set to use PyGEOS, changing to shapely 2.0 with:"
                    "\n\n\tgeopandas.options.use_pygeos = True\n\n"
                    "If you intended to use PyGEOS, set the option to False."
                ),
                UserWarning,
                stacklevel=2,
            )
