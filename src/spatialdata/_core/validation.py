from __future__ import annotations

import pandas as pd
from anndata import AnnData


def check_target_region_column_symmetry(table: AnnData, region_key: str, target: str | pd.Series) -> None:
    """
    Check region and region_key column symmetry.

    This checks whether the specified targets are also present in the region key column in obs and raises an error
    if this is not the case.

    Parameters
    ----------
    table
        Table annotating specific SpatialElements
    region_key
        The column in obs containing for each row which SpatialElement is annotated by that row.
    target
         Name of target(s) SpatialElement(s)

    Raises
    ------
    ValueError
        If there is a mismatch between specified target regions and regions in the region key column of table.obs.

    Example
    -------
    Assuming we have a table with region column in obs given by `region_key` called 'region' for which we want to check
    whether it contains the specified annotation targets in the `target` variable as `pd.Series['region1', 'region2']`:

    ```python
    check_target_region_column_symmetry(table, region_key=region_key, target=target)
    ```

    This returns None if both specified targets are present in the region_key obs column. In this case the annotation
    targets can be safely set. If not then a ValueError is raised stating the elements that are not shared between
    the region_key column in obs and the specified targets.
    """
    found_regions = set(table.obs[region_key].unique().tolist())
    target_element_set = [target] if isinstance(target, str) else target
    symmetric_difference = found_regions.symmetric_difference(target_element_set)
    if symmetric_difference:
        raise ValueError(
            f"Mismatch(es) found between regions in region column in obs and target element: "
            f"{', '.join(diff for diff in symmetric_difference)}"
        )


def check_valid_name(name: str) -> None:
    """
    Check that a name is valid for SpatialData elements.

    This checks whether the proposed name fulfills the naming restrictions and raises an error
    otherwise.

    Parameters
    ----------
    name
        The name for a SpatialData element

    Raises
    ------
    TypeError
        If given argument is not of type string.
    ValueError
        If the proposed name viloates a naming restriction.
    """
    if not isinstance(name, str):
        raise TypeError(f"Name must be a string, not {type(name).__name__}.")
    if len(name) == 0:
        raise ValueError("Name cannot be an empty string.")
    if not all(c.isalnum() or c in "_-" for c in name):
        raise ValueError("Name must contain only alphanumeric characters, underscores, and hyphens.")
