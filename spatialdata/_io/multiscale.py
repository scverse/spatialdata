from spatialdata._core.coordinate_system import CoordinateSystem
from spatialdata._core.transform import BaseTransformation


def _generate_multiscale_dict(
    transformations: Dict[str, BaseTransformation],
    coordinate_systems: Dict[str, CoordinateSystem],
    fmt: Format,
    base_element_path: str,
) -> Dict[str, Any]:
    """Generate a multiscale dictionary for a given transformation and coordinate system."""
    coordinate_systems = []
    datasets = []
    description = {
        "version": "0.5-dev",
        "name": base_element_path,
        "coordinateSystems": coordinate_systems,
        "datasets": datasets,
    }
    multiscale_dict = {"multiscales": [description]}
    return multiscale_dict
