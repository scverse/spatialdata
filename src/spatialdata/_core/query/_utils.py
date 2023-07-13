import geopandas as gpd


def circles_to_polygons(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # We should only be buffering points, not polygons. Unfortunately this is an expensive check.
    from spatialdata.models import ShapesModel

    values_geotypes = list(df.geom_type.unique())
    if values_geotypes == ["Point"]:
        buffered_df = df.set_geometry(df.geometry.buffer(df[ShapesModel.RADIUS_KEY]))
        # TODO replace with a function to copy the metadata (the parser could also do this): https://github.com/scverse/spatialdata/issues/258
        buffered_df.attrs[ShapesModel.TRANSFORM_KEY] = df.attrs[ShapesModel.TRANSFORM_KEY]
        return buffered_df
    if "Point" in values_geotypes:
        raise TypeError("Geometry contained shapes and polygons.")
    return df
