import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import zarr
from anndata import AnnData
from anndata._io import read_zarr as read_anndata_zarr
from ome_zarr.io import ZarrLocation
from ome_zarr.reader import Label, Multiscales, Reader

from spatialdata._core._spatialdata import SpatialData
from spatialdata._core.coordinate_system import CoordinateSystem
from spatialdata._core.transform import BaseTransformation, get_transformation_from_dict
from spatialdata._io.format import SpatialDataFormat


def read_zarr(store: Union[str, Path, zarr.Group]) -> SpatialData:

    if isinstance(store, Path):
        store = str(store)

    fmt = SpatialDataFormat()

    f = zarr.open(store, mode="r")
    images = {}
    labels = {}
    points = {}
    table: Optional[AnnData] = None
    polygons = {}
    transformations: Dict[Tuple[str, str], BaseTransformation] = {}
    coordinate_systems: Dict[str, CoordinateSystem] = {}
    # images_transform = {}
    # labels_transform = {}
    # points_transform = {}
    # polygons_transform = {}

    def _get_transformations_and_coordinate_systems_from_group(
        group: zarr.Group,
    ) -> Tuple[Dict[Tuple[str, str], BaseTransformation], Dict[str, CoordinateSystem]]:
        multiscales = group.attrs["multiscales"]
        # TODO: parse info from multiscales['axes']
        assert len(multiscales) == 1, f"TODO: expecting only one multiscale, got {len(multiscales)}"
        datasets = multiscales[0]["datasets"]
        assert len(datasets) == 1, "Expecting only one dataset"
        coordinate_systems = multiscales[0]["coordinateSystems"]
        from spatialdata._core._spatialdata import _validate_coordinate_systems

        cs = _validate_coordinate_systems(coordinate_systems)
        coordinate_transformations = datasets[0]["coordinateTransformations"]
        ct = {}
        for e in coordinate_transformations:
            # e['input'] has values like 'anatomical/polygons/anatomical', 'cells/points/cells',
            # 'points8/labels/points8' or 'points8'. The first three cases are for polygons, points and labels and
            # the string between the two '/' tells the type of the group. The last case is for images.
            s_input = e["input"].split("/")
            assert len(s_input) == 3
            element_type = "/".join(s_input[1:])
            t = get_transformation_from_dict(e)
            ct[(element_type, e["output"])] = t
        return ct, cs

    def _update_ct_and_cs(ct: Dict[Tuple[str, str], BaseTransformation], cs: Dict[str, CoordinateSystem]):
        assert not any([k in transformations for k in ct.keys()])
        transformations.update(ct)
        for k, v in cs.items():
            if k in coordinate_systems:
                assert coordinate_systems[k] == v
            else:
                coordinate_systems[k] = v

    for k in f.keys():
        f_elem = f[k].name
        f_elem_store = f"{store}{f_elem}"
        image_loc = ZarrLocation(f_elem_store)
        image_reader = Reader(image_loc)()
        image_nodes = list(image_reader)
        # read multiscale images that are not labels
        start = time.time()
        if len(image_nodes):
            for node in image_nodes:
                if np.any([isinstance(spec, Multiscales) for spec in node.specs]) and np.all(
                    [not isinstance(spec, Label) for spec in node.specs]
                ):
                    print(f"action0: {time.time() - start}")
                    start = time.time()
                    images[k] = node.load(Multiscales).array(resolution="0", version=fmt.version)
                    ct, cs = _get_transformations_and_coordinate_systems_from_group(zarr.open(node.zarr.path, mode="r"))
                    _update_ct_and_cs(ct, cs)

                    print(f"action1: {time.time() - start}")
        # read all images/labels for the level
        # warnings like "no parent found for <ome_zarr.reader.Label object at 0x1c789f310>: None" are expected,
        # since we don't link the image and the label inside .zattrs['image-label']
        labels_loc = ZarrLocation(f"{f_elem_store}/labels")
        start = time.time()
        if labels_loc.exists():
            labels_reader = Reader(labels_loc)()
            labels_nodes = list(labels_reader)
            if len(labels_nodes):
                for node in labels_nodes:
                    if np.any([isinstance(spec, Label) for spec in node.specs]):
                        print(f"action0: {time.time() - start}")
                        start = time.time()
                        labels[k] = node.load(Multiscales).array(resolution="0", version=fmt.version)
                        ct, cs = _get_transformations_and_coordinate_systems_from_group(
                            zarr.open(node.zarr.path, mode="r")
                        )
                        _update_ct_and_cs(ct, cs)
                        print(f"action1: {time.time() - start}")
        # now read rest
        start = time.time()
        g = zarr.open(f_elem_store, mode="r")
        for j in g.keys():
            g_elem = g[j].name
            g_elem_store = f"{f_elem_store}{g_elem}{f_elem}"
            if g_elem == "/points":
                points[k] = read_anndata_zarr(g_elem_store)
                ct, cs = _get_transformations_and_coordinate_systems_from_group(zarr.open(g_elem_store, mode="r"))
                _update_ct_and_cs(ct, cs)

            if g_elem == "/polygons":
                polygons[k] = read_anndata_zarr(g_elem_store)
                ct, cs = _get_transformations_and_coordinate_systems_from_group(zarr.open(g_elem_store, mode="r"))
                _update_ct_and_cs(ct, cs)

            if g_elem == "/table":
                table = read_anndata_zarr(f"{f_elem_store}{g_elem}")
        print(f"rest: {time.time() - start}")

    # transformations = {}
    # for prefix, t in zip(
    #     ["images", "labels", "points", "polygons"],
    #     [images_transform, labels_transform, points_transform, polygons_transform],
    # ):
    #     for name, transformation in t.items():
    #         transformations[f"{prefix}/{name}"] = transformation
    # merged_transformations = {k: v for d in [images_transform, labels_transform, points_transform, polygons_transform] for k, v in d.items()}
    # merged_coordinate_systems =

    return SpatialData(
        images=images,
        labels=labels,
        points=points,
        polygons=polygons,
        table=table,
        transformations=transformations,
        coordinate_systems=list(coordinate_systems.values()) if len(coordinate_systems) > 0 else None,
    )


def load_table_to_anndata(file_path: str, table_group: str) -> AnnData:
    return read_zarr(os.path.join(file_path, table_group))


if __name__ == "__main__":
    sdata = SpatialData.read("../../spatialdata-sandbox/nanostring_cosmx/data_small.zarr")
    print(sdata)
    from napari_spatialdata import Interactive

    Interactive(sdata)
