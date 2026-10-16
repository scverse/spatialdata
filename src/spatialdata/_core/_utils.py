from __future__ import annotations

from collections.abc import Iterable, MutableMapping
from typing import Any, cast

from mudata import MuData

from spatialdata._core.spatialdata import SpatialData
from spatialdata._types import Table_T
from spatialdata._utils import _copy_table


def _find_common_table_keys(sdatas: Iterable[SpatialData]) -> set[str]:
    """
    Find table keys present in more than one SpatialData object.

    Parameters
    ----------
    sdatas
        An `Iterable` of SpatialData objects.

    Returns
    -------
    A set of common keys that are present in the tables of more than one SpatialData object.
    """
    common_keys: set[str] = set()

    for sdata in sdatas:
        if len(common_keys) == 0:
            common_keys = set(sdata.tables.keys())
        else:
            common_keys.intersection_update(sdata.tables.keys())

    return common_keys


def sanitize_name(name: str, is_dataframe_column: bool = False) -> str:
    """
    Sanitize a name to comply with SpatialData naming rules.

    This function converts invalid names into valid ones by:
    1. Converting to string if not already
    2. Removing invalid characters
    3. Handling special cases like "__" prefix
    4. Ensuring the name is not empty
    5. Handling special cases for dataframe columns

    See a discussion on the naming rules, and how to avoid naming collisions, here:
    https://github.com/scverse/spatialdata/discussions/707

    Parameters
    ----------
    name
        The name to sanitize
    is_dataframe_column
        Whether this name is for a dataframe column (additional restrictions apply)

    Returns
    -------
    A sanitized version of the name that complies with SpatialData naming rules. If a
    santized name cannoted be generated, it returns "unnamed".

    Examples
    --------
    >>> sanitize_name("my@invalid#name")
    'my_invalid_name'
    >>> sanitize_name("__private")
    'private'
    >>> sanitize_name("_index", is_dataframe_column=True)
    'index'
    """
    # Convert to string if not already
    name = str(name)

    # Handle empty string case
    if not name:
        return "unnamed"

    # Handle special cases
    if name in {".", ".."}:
        return "unnamed"

    sanitized = "".join(char if char.isalnum() or char in "_-." else "_" for char in name)

    # remove double underscores if found as a prefix
    while sanitized.startswith("__"):
        sanitized = sanitized[1:]

    if is_dataframe_column and sanitized == "_index":
        return "index"

    # Ensure we don't end up with an empty string after sanitization
    return sanitized or "unnamed"


def _rename_keys_in_place(mapping: MutableMapping[str, Any], new_keys: dict[str, str]) -> None:
    """Rename the keys of a mapping in place, keeping their order."""
    if all(new_keys.get(key, key) == key for key in mapping):
        return
    items = [(new_keys.get(key, key), mapping.pop(key)) for key in list(mapping)]
    for key, value in items:
        mapping[key] = value


def sanitize_table(data: Table_T, inplace: bool = True) -> Table_T | None:
    """
    Sanitize all keys in an AnnData or MuData table to comply with SpatialData naming rules.

    This function sanitizes all keys in obs, var, obsm, obsp, varm, varp, uns, and layers
    while maintaining case-insensitive uniqueness. It can either modify the table in-place
    or return a new sanitized copy. For a MuData table, the names of the modalities, the global attributes and every
    modality are sanitized.

    See a discussion on the naming rules here:
    https://github.com/scverse/spatialdata/discussions/707

    Parameters
    ----------
    data
        The AnnData or MuData table to sanitize
    inplace
        Whether to modify the table in-place or return a new copy

    Returns
    -------
    If inplace is False, returns a new table with sanitized keys.
    If inplace is True, returns None as the original object is modified.

    Examples
    --------
    >>> import anndata as ad
    >>> adata = ad.AnnData(obs=pd.DataFrame({"@invalid#": [1, 2]}))
    >>> # Create a new sanitized copy
    >>> sanitized = sanitize_table(adata)
    >>> print(sanitized.obs.columns)
    Index(['invalid_'], dtype='object')
    >>> # Or modify in-place
    >>> sanitize_table(adata, inplace=True)
    >>> print(adata.obs.columns)
    Index(['invalid_'], dtype='object')
    """
    from collections import defaultdict

    # Create a copy if not modifying in-place
    sanitized = data if inplace else _copy_table(data)

    # Track used names to maintain case-insensitive uniqueness
    used_names_lower: dict[str, set[str]] = defaultdict(set)

    def get_unique_name(name: str, attr: str, is_dataframe_column: bool = False) -> str:
        base_name = sanitize_name(name, is_dataframe_column)
        normalized_base = base_name.lower()

        # If this exact name is already used, add a number
        if normalized_base in used_names_lower[attr]:
            counter = 1
            while f"{base_name}_{counter}".lower() in used_names_lower[attr]:
                counter += 1
            base_name = f"{base_name}_{counter}"

        used_names_lower[attr].add(base_name.lower())
        return base_name

    is_mudata = isinstance(sanitized, MuData)
    if isinstance(sanitized, MuData):
        # the modalities are also the keys of their boolean masks in obsm/varm and of their maps in obsmap/varmap,
        # which are renamed with them; the masks are then excluded from the renaming of obsm/varm below
        new_mod_names = {old: get_unique_name(old, "mod") for old in sanitized.mod}
        for mapping in (sanitized.mod, sanitized.obsm, sanitized.varm, sanitized.obsmap, sanitized.varmap):
            _rename_keys_in_place(cast(MutableMapping[str, Any], mapping), new_mod_names)
        for attr in ("obsm", "varm"):
            used_names_lower[attr].update(name.lower() for name in sanitized.mod)

    # Handle obs and var (dataframe columns)
    for attr in ("obs", "var"):
        df = getattr(sanitized, attr)
        new_columns = {old: get_unique_name(old, attr, is_dataframe_column=True) for old in df.columns}
        df.rename(columns=new_columns, inplace=True)

    # Handle other attributes; MuData has no global layers, they live in the modalities
    modality_names = set(sanitized.mod) if isinstance(sanitized, MuData) else set()
    for attr in ("obsm", "obsp", "varm", "varp", "uns") + (() if is_mudata else ("layers",)):
        d = getattr(sanitized, attr)
        # None is a valid key in layers (anndata >= 0.13: represents X); skip sanitizing it
        new_keys = {
            old: get_unique_name(old, attr)
            for old in d
            if old is not None and not (attr in ("obsm", "varm") and old in modality_names)
        }
        if is_mudata:
            # the setters of MuData rebuild the mappings (and validate varp against the observations), so the keys
            # are renamed in place
            _rename_keys_in_place(d, new_keys)
        else:
            new_dict = {(new_keys[old] if old is not None else old): value for old, value in d.items()}
            setattr(sanitized, attr, new_dict)

    if isinstance(sanitized, MuData):
        for modality in sanitized.mod.values():
            sanitize_table(modality, inplace=True)

    return None if inplace else sanitized
