from __future__ import annotations

# from https://stackoverflow.com/a/24860799/3343783
import filecmp
import os.path
import re
import tempfile
from typing import TYPE_CHECKING, Any, Optional

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


def are_directories_identical(
    dir1: Any,
    dir2: Any,
    exclude_regexp: Optional[str] = None,
    _root_dir1: Optional[str] = None,
    _root_dir2: Optional[str] = None,
) -> bool:
    """
    Compare two directory trees content.
    Return False if they differ, True is they are the same.
    """
    if _root_dir1 is None:
        _root_dir1 = dir1
    if _root_dir2 is None:
        _root_dir2 = dir2
    if exclude_regexp is not None:
        if re.match(rf"{_root_dir1}/" + exclude_regexp, dir1) or re.match(rf"{_root_dir2}/" + exclude_regexp, dir2):
            return True
        ##
        # loc = ZarrLocation(_root_dir1)
        # reader = Reader(loc)()
        # nodes = list(reader)
        # assert len(nodes) == 1
        # node = nodes[0]
        # im_read = node.load(Multiscales).array(resolution="1", version=CurrentFormat().version)
        #
        # loc = ZarrLocation(_root_dir2)
        # reader = Reader(loc)()
        # nodes = list(reader)
        # assert len(nodes) == 1
        # node = nodes[0]
        # im_read2 = node.load(Multiscales).array(resolution="1", version=CurrentFormat().version)
        # print(im_read[:5, :5, 0].compute())
        # print(im_read2[:5, :5, 0].compute())
        ##
    compared = dircmp(dir1, dir2)
    if compared.left_only or compared.right_only or compared.diff_files or compared.funny_files:
        return False
        # # temporary workaround for https://github.com/ome/ome-zarr-py/issues/219
        # ##
        # if compared.diff_files == ['.zarray']:
        #     with open(os.path.join(dir1, '.zarray'), 'r') as f1:
        #         with open(os.path.join(dir2, '.zarray'), 'r') as f2:
        #             d = difflib.unified_diff(f1.readlines(), f2.readlines())
        #             diffs = [dd for dd in d]
        #             # the workaroud permits only diffs that look like this
        #             # ['--- \n',
        #             #  '+++ \n',
        #             #  '@@ -4,13 +4,7 @@\n',
        #             #  '         100,\n',
        #             #  '         100\n',
        #             #  '     ],\n',
        #             #  '-    "compressor": {\n',
        #             #  '-        "blocksize": 0,\n',
        #             #  '-        "clevel": 5,\n',
        #             #  '-        "cname": "lz4",\n',
        #             #  '-        "id": "blosc",\n',
        #             #  '-        "shuffle": 1\n',
        #             #  '-    },\n',
        #             #  '+    "compressor": null,\n',
        #             #  '     "dimension_separator": "/",\n',
        #             #  '     "dtype": "<f8",\n',
        #             #  '     "fill_value": 0.0,\n']
        #     ##
        #     regexp = r"\['(:?-|\+)+ \\n', '(:?-|\+)+ \\n', '@@ (:?-|\+)[0-9]+,[0-9]+ (:?-|\+)[0-9]+,[0-9]+ @@\\n'," \
        #              r"[\s\S]*?\"compressor\": {([\s\S]*?)},\\n', '(:?-|\+)\s+\"compressor\"([" \
        #              r"^-\+]+?)'\s+\"dimension_separator\"[\s\S]*"
        #     s = str(diffs)
        #     m = re.search(regexp, s)
        #     groups = m.groups()
        #     if len(groups) > 0:
        #         for g in groups:
        #             if '{' in g or '}' in g:
        #                 return False
        #     else:
        #         return False
        #     ##
        # else:
        #     return False
    for subdir in compared.common_dirs:
        if not are_directories_identical(
            os.path.join(dir1, subdir),
            os.path.join(dir2, subdir),
            exclude_regexp=exclude_regexp,
            _root_dir1=_root_dir1,
            _root_dir2=_root_dir2,
        ):
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
