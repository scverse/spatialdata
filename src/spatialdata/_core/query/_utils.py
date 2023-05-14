import geopandas as gpd


def circles_to_polygons(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    # We should only be buffering points, not polygons. Unfortunately this is an expensive check.
    from spatialdata.models import ShapesModel

    values_geotypes = list(df.geom_type.unique())
    if values_geotypes == ["Point"]:
        df = df.set_geometry(df.geometry.buffer(df[ShapesModel.RADIUS_KEY]))
    elif "Point" in values_geotypes:
        raise TypeError("Geometry contained shapes and polygons.")
    return df
