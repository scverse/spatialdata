from __future__ import annotations

# from https://stackoverflow.com/a/24860799/3343783
import filecmp
import os.path
import tempfile
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from spatialdata import SpatialData


class dircmp(filecmp.dircmp):  # type: ignore[type-arg]
    """
    Compare the content of dir1 and dir2. In contrast with filecmp.dircmp, this
    subclass compares the content of files with the same path.
    """

    def phase3(self) -> None:
        """
        Find out differences between common files.
        Ensure we are using content comparison with shallow=False.
        """
        fcomp = filecmp.cmpfiles(self.left, self.right, self.common_files, shallow=False)
        self.same_files, self.diff_files, self.funny_files = fcomp


def are_directories_identical(dir1: Any, dir2: Any) -> bool:
    """
    Compare two directory trees content.
    Return False if they differ, True is they are the same.
    """
    compared = dircmp(dir1, dir2)
    if compared.left_only or compared.right_only or compared.diff_files or compared.funny_files:
        return False
    for subdir in compared.common_dirs:
        if not are_directories_identical(os.path.join(dir1, subdir), os.path.join(dir2, subdir)):
            return False
    return True


def compare_sdata_on_disk(a: SpatialData, b: SpatialData) -> bool:
    from spatialdata import SpatialData

    if not isinstance(a, SpatialData) or not isinstance(b, SpatialData):
        return False
    # TODO: if the sdata object is backed on disk, don't create a new zarr file
    with tempfile.TemporaryDirectory() as tmpdir:
        a.write(os.path.join(tmpdir, "a.zarr"))
        b.write(os.path.join(tmpdir, "b.zarr"))
        return are_directories_identical(os.path.join(tmpdir, "a.zarr"), os.path.join(tmpdir, "b.zarr"))
