import os
import random
import warnings
from typing import Any, Optional

import dask.array as da
import matplotlib.cm
import numpy as np
import pandas as pd
import zarr
from anndata import AnnData
from anndata._io import read_zarr
from napari.types import LayerDataTuple
from napari_ome_zarr._reader import transform
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader


def load_table_to_anndata(file_path: str, table_group: str) -> AnnData:
    return read_zarr(os.path.join(file_path, table_group))


def _anndata_to_napari_labels_features(anndata_obj: AnnData, table_instance_key: str) -> pd.DataFrame:
    df = pd.concat([anndata_obj.to_df(), anndata_obj.obs], axis=1)

    # hacky add dummy data to first row to meet the condition for napari features (first row is the background)
    # https://github.com/napari/napari/blob/2648c307168fdfea7b5c03aaad4590972e5dfe2c/napari/layers/labels/labels.py#L72-L74
    warnings.warn("background value is arbitrary, this is just for testing purposes and will have to be changed")
    first_row = df.iloc[0, :].copy()
    first_row[table_instance_key] = 0
    return pd.concat([pd.DataFrame(first_row).T, df])


def get_napari_image_layer_data(file_path: str) -> LayerDataTuple:
    ome_zarr = parse_url(file_path)
    reader = Reader(ome_zarr)
    reader_func = transform(reader())
    layer_data = reader_func(file_path)
    # here we consider only images, as we deal with labels and other types in dedicated functions
    image_layer_data = []
    for layer in layer_data:
        data, kwargs, layer_type = layer
        if layer_type == "image":
            # this doesn't work, so the RGB image is split into three channels
            # first element of the pyramidal array
            # if data[0].shape[0] in [3, 4]:
            #     kwargs['rgb'] = True
            # assuming y and x are the two last dimesions
            image_layer_data.append(([da.moveaxis(d, -1, -2) for d in data], kwargs, layer_type))
    return image_layer_data


def _colorize_regions_for_napari_layer(
    features_table: Any,
    kwargs: Any,
    color_key_name: str,
    tables_instance_key: Optional[str] = None,
):
    """helper function to build a list of colors to colorize some regions

    Parameters
    ----------
    features_table : anndata.AnnData
        AnnData object with the features table and, if tables_instance_key is not None, also a label ids column
    kwargs : dict[str, Any]
        the kwargs passed to _add_layer_from_data for constructing Napari layers
    color_key_name : str
        the colors will be placed (by this function) inside the kwargs under this key
    tables_instance_key : Optional[str], optional
        the column of the AnnData object containing the label ids, by default None

    Raises
    ------
    ValueError
        return an expection if color_key_name is not a string supported by napari for constructing layers
    """
    # manually assigning colors from a colormap
    to_color_as_background = np.zeros(len(features_table), dtype=bool)
    if tables_instance_key is not None:
        to_color_as_background[features_table[tables_instance_key] == 0] = True
    # let's show a random feature for the moment
    random_feature = random.sample(features_table.columns.values.tolist(), 1)[0]
    values = features_table[random_feature].to_numpy().astype(float)
    a = np.min(values)
    b = np.max(values)
    values = (values - a) / (b - a)
    invalid = np.logical_or(np.isnan(values), np.isinf(values))
    to_color_as_background[invalid] = True
    values[invalid] = 0.0
    cmap = matplotlib.cm.get_cmap("viridis")
    colors = cmap(values)
    # let's show background in red for debug purposes
    background_color = (1.0, 0.0, 0.0, 1.0)
    colors[to_color_as_background] = background_color
    if color_key_name == "color":
        color_value = {i: colors[i, :] for i in range(len(features_table))}
    elif color_key_name == "face_color":
        color_value = colors
    else:
        raise ValueError()
    kwargs.update(
        {
            "features": features_table,
            "name": random_feature,
            color_key_name: color_value,
        }
    )


def get_napari_label_layer_data(
    file_path: str,
    labels_group_key: str,
    features_table: Optional[pd.DataFrame] = None,
    tables_instance_key: Optional[str] = None,
) -> LayerDataTuple:
    ome_zarr = parse_url(os.path.join(file_path, "labels"))
    reader = Reader(ome_zarr)
    reader_func = transform(reader())
    layer_data = reader_func(os.path.join(file_path, "labels"))

    # currently no matters how many label_images are present in /labels, the zarr reader will onl y read the first
    # one. This because multiple image labels are not (currently) supported by the OME-NGFF specification
    # so let's deal with this constraint
    assert len(layer_data) == 1
    data, kwargs, layer_type = layer_data[0]
    assert layer_type.lower() == "labels"

    # here we colorize the labels, either by matching the features and labels by order, either by their value (when tables_instance_key is specified)
    if features_table is not None:
        image_ids = set(da.unique(data[0]).compute().tolist())
        display_feature = False
        if tables_instance_key is not None:
            warnings.warn("currently ignoring tables_instance_key and matching regions and features just by the order")
            table_ids = set(features_table[tables_instance_key].tolist())
            image_ids.add(0)
            ids_not_in_table = image_ids.difference(table_ids)
            ids_not_in_image = table_ids.difference(image_ids)
            if len(ids_not_in_table) > 0:
                warnings.warn(
                    "labels in the image do not have a corresponding value in the table! displaying them as background"
                )
                warnings.warn(
                    "background value is arbitrary, this is just for testing purposes and will have to be changed"
                )
                rows_to_append = []
                for label in ids_not_in_table:
                    new_row = features_table.iloc[0, :].copy()
                    new_row[tables_instance_key] = label
                    rows_to_append.append(new_row)
                features_table = pd.concat([features_table, pd.DataFrame(rows_to_append)])
            elif len(ids_not_in_image) > 0:
                warnings.warn("some rows from the table do not have a corresponding label in the image, ignoring them")
                features_table = features_table[~features_table[tables_instance_key].isin(ids_not_in_image), :]
            assert len(image_ids) == len(features_table[tables_instance_key])
            features_table.sort_values(by=tables_instance_key, inplace=True)
            display_feature = True
        else:
            n = len(image_ids)
            # to count the number of labels that are not background
            if 0 in image_ids:
                n -= 1
            if n + 1 == len(features_table):
                display_feature = True
            else:
                raise ValueError(
                    "the number of labels does not match the number of features. It is recommended to match labels and features via a column in the dataframe that specify the label of each feature"
                )
        if display_feature:
            _colorize_regions_for_napari_layer(
                features_table=features_table,
                kwargs=kwargs,
                color_key_name="color",
                tables_instance_key=tables_instance_key,
            )

    new_layer_data = (data, kwargs, layer_type)
    return new_layer_data


def get_napari_circles_layer_data(
    file_path: str,
    circles_group_key: str,
    features_table: Optional[pd.DataFrame] = None,
    tables_instance_key: Optional[str] = None,
) -> LayerDataTuple:
    ome_zarr = parse_url(file_path)
    circles_group = zarr.group(ome_zarr.store)[circles_group_key]
    assert "@type" in circles_group.attrs and circles_group.attrs["@type"] == "ngff:circles_table"
    anndata_obj = load_table_to_anndata(file_path, circles_group_key)
    xy = anndata_obj.obsm["spatial"]
    assert len(anndata_obj) == len(xy)

    # here we are supporing both the Visium-derived squidpy storage and a simpler one in which we just put the radii values in obsm
    try:
        radii = anndata_obj.obsm["region_radius"]

    except KeyError:
        scale_factors = anndata_obj.uns["spatial"].values().__iter__().__next__()["scalefactors"]
        radius = scale_factors["spot_diameter_fullres"] * scale_factors["tissue_hires_scalef"]
        radii = np.array([radius] * len(anndata_obj))

    kwargs = {"edge_width": 0.0, "size": radii}
    if features_table is not None:
        if tables_instance_key is not None:
            raise NotImplementedError()
        else:
            n_circles = len(anndata_obj)
            n_features = len(features_table)
            assert n_circles == n_features
        _colorize_regions_for_napari_layer(
            features_table=features_table,
            kwargs=kwargs,
            tables_instance_key=tables_instance_key,
            color_key_name="face_color",
        )
    else:
        random_colors = np.concatenate(
            (
                np.random.rand(len(anndata_obj), 3),
                np.ones((len(anndata_obj), 1), dtype=float),
            ),
            axis=1,
        )
        kwargs["face_color"] = random_colors

    new_layer_data = (np.fliplr(xy), kwargs, "points")
    return new_layer_data


def get_napari_table_layer_data(file_path: str, table_group_key: str) -> LayerDataTuple:
    ome_zarr = parse_url(os.path.join(file_path, table_group_key))
    # load table
    anndata_obj = load_table_to_anndata(file_path, table_group_key)

    # then load the regions object associated to the table, necessary to map features to space
    table_group = zarr.group(ome_zarr.store)
    tables_region = table_group.attrs["region"]
    if tables_region is None:
        raise ValueError(
            "every feature table must be associated with a regions object, otherwise there is no way to map the features to space"
        )
    tables_region_key = table_group.attrs["region_key"]
    if tables_region_key is not None:
        raise NotImplementedError('currently only a single "regions" object can be mapped to a feature table')
    table_instance_key = table_group.attrs["instance_key"]
    group_type = tables_region.split("/")[0]
    if group_type == "labels":
        features_table = _anndata_to_napari_labels_features(anndata_obj, table_instance_key=table_instance_key)
        return get_napari_label_layer_data(
            file_path=file_path,
            labels_group_key=tables_region,
            features_table=features_table,
            tables_instance_key=table_instance_key,
        )
    elif group_type == "circles":
        return get_napari_circles_layer_data(
            file_path=file_path,
            circles_group_key=tables_region,
            features_table=anndata_obj.to_df(),
            tables_instance_key=table_instance_key,
        )
    elif group_type == "polygons":
        raise NotImplementedError()
    else:
        raise ValueError('region must specify the zarr path (relative to the root) of a "region" type')


def get_napari_points_layer_data(file_path: str, points_group_key: str) -> LayerDataTuple:
    # load table
    anndata_obj = load_table_to_anndata(file_path, points_group_key)

    ome_zarr = parse_url(file_path)
    table_group = zarr.group(ome_zarr.store)[points_group_key]
    assert "@type" in table_group.attrs and table_group.attrs["@type"] == "ngff:points_table"

    # what is layer_properties for? notice that not all the layers set this
    layer_properties = anndata_obj.obs
    new_layer_data = (
        np.fliplr(anndata_obj.X),
        {"edge_width": 0.0, "size": 1, "properties": layer_properties},
        "points",
    )
    return new_layer_data


def get_napari_polygons_layer_data(file_path: str, polygons_group_key: str) -> LayerDataTuple:
    # load table
    anndata_obj = load_table_to_anndata(file_path, polygons_group_key)

    ome_zarr = parse_url(file_path)
    table_group = zarr.group(ome_zarr.store)[polygons_group_key]
    assert "@type" in table_group.attrs and table_group.attrs["@type"] == "ngff:polygons_table"

    # this is just temporary, it is a security problem. It is converting the string repr of a np.array back to a np.array
    anndata_obj.obs["vertices"] = anndata_obj.obs["vertices"].apply(lambda x: eval("np." + x))
    list_of_vertices = anndata_obj.obs["vertices"].tolist()
    list_of_vertices = [np.fliplr(v) for v in list_of_vertices]

    new_layer_data = (
        list_of_vertices,
        {
            "edge_width": 5.0,
            "shape_type": "polygon",
            "face_color": np.array([0.0, 0, 0.0, 0.0]),
            "edge_color": np.random.rand(len(list_of_vertices), 3),
        },
        "shapes",
    )
    return new_layer_data


def _get_layer(file_path: str, group_key: str) -> LayerDataTuple:
    # here I assume that the root of zarr is /, which implies that eventual "labels" "regions_table",
    # "points" groups, are the root level of the hierarchy if present
    group_type = group_key.split("/")[0]
    if group_type == "points":
        return get_napari_points_layer_data(file_path=file_path, points_group_key=group_key)
    elif group_type == "labels":
        return get_napari_label_layer_data(file_path=file_path, labels_group_key=group_key)
    elif group_type == "tables":
        return get_napari_table_layer_data(file_path=file_path, table_group_key=group_key)
    elif group_type == "circles":
        return get_napari_circles_layer_data(file_path=file_path, circles_group_key=group_key)
    elif group_type == "polygons":
        return get_napari_polygons_layer_data(file_path=file_path, polygons_group_key=group_key)
    else:
        raise ValueError()
