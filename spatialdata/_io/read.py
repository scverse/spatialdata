import os

from anndata import AnnData
from anndata._io import read_zarr


def load_table_to_anndata(file_path: str, table_group: str) -> AnnData:
    return read_zarr(os.path.join(file_path, table_group))
