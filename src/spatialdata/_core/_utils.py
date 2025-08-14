from collections.abc import Iterable

from anndata import AnnData

from spatialdata._core.spatialdata import SpatialData


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


def sanitize_table(data: AnnData, inplace: bool = True) -> AnnData | None:
    """
    Sanitize all keys in an AnnData table to comply with SpatialData naming rules.

    This function sanitizes all keys in obs, var, obsm, obsp, varm, varp, uns, and layers
    while maintaining case-insensitive uniqueness. It can either modify the table in-place
    or return a new sanitized copy.

    See a discussion on the naming rules here:
    https://github.com/scverse/spatialdata/discussions/707

    Parameters
    ----------
    data
        The AnnData table to sanitize
    inplace
        Whether to modify the table in-place or return a new copy

    Returns
    -------
    If inplace is False, returns a new AnnData object with sanitized keys.
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
    import copy
    from collections import defaultdict

    # Create a deep copy if not modifying in-place
    sanitized = data if inplace else copy.deepcopy(data)

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

    # Handle obs and var (dataframe columns)
    for attr in ("obs", "var"):
        df = getattr(sanitized, attr)
        new_columns = {old: get_unique_name(old, attr, is_dataframe_column=True) for old in df.columns}
        df.rename(columns=new_columns, inplace=True)

    # Handle other attributes
    for attr in ("obsm", "obsp", "varm", "varp", "uns", "layers"):
        d = getattr(sanitized, attr)
        new_keys = {old: get_unique_name(old, attr) for old in d}
        # Create new dictionary with sanitized keys
        new_dict = {new_keys[old]: value for old, value in d.items()}
        setattr(sanitized, attr, new_dict)

    return None if inplace else sanitized
