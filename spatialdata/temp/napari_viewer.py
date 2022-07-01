import pathlib
import spatialdata as sd


def view_with_napari(path: str):
    assert pathlib.Path(path).exists()
    # sdata = sd.from_zarr(path)
    # table_key = {sdata.regions.keys().__iter__().__next__()}
    table_key = 'regions_table'

    from ngff_tables_prototype.reader import load_to_napari_viewer
    import napari

    viewer = load_to_napari_viewer(
        file_path=path,
        groups=[
            "circles/circles_table",
            f"tables/{table_key}",
            "points/points_table",
            # "polygons/polygons_table",
        ],
    )
    napari.run()
