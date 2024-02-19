"""
The CLI Interaction module.

This module provides command line interface (CLI) interactions for the SpatialData library, allowing users to perform
various operations through a terminal. Currently, it implements the "peek" function, which allows users to inspect
the contents of a SpatialData .zarr dataset. Additional CLI functionalities will be implemented in the future.
"""

from typing import Literal

import click


@click.command(help="Peek inside the SpatialData .zarr dataset")
@click.argument("path", default=False, type=str)
@click.argument("selection", type=click.Choice(["images", "labels", "points", "shapes", "table"]), nargs=-1)
def peek(path: str, selection: tuple[Literal["images", "labels", "points", "shapes", "table"]]) -> None:
    """
    Peek inside the SpatialData .zarr dataset.

    This function takes a path to a local or remote .zarr dataset, reads and prints
    its contents using the SpatialData library. If any ValueError is raised, it is caught and printed to the
    terminal along with a help message.

    Parameters
    ----------
    path
        The path to the .zarr dataset to be inspected.
    selection
        Optional, a list of keys (among images, labels, points, shapes, table) to load only a subset of the dataset.
        Example: `python -m spatialdata peek data.zarr images labels`
    """
    import spatialdata as sd

    try:
        sdata = sd.SpatialData.read(path, selection=selection)
        print(sdata)  # noqa: T201
    except ValueError as e:
        # checking if a valid path was provided is difficult given the various ways in which
        # a possibly remote path and storage access options can be specified
        # so we just catch the ValueError and print a help message
        print(e)  # noqa: T201
        print(  # noqa: T201
            f"Error: .zarr storage not found at {path}. Please specify a valid OME-NGFF spatial data (.zarr) file. "
            "Examples "
            '"python -m spatialdata peek data.zarr"'
            '"python -m spatialdata peek https://remote/.../data.zarr labels table"'
        )


@click.group()
def cli() -> None:
    """Provide the main Click command group.

    This function serves as the main entry point for the command-line interface. It creates a Click command
    group and adds the various cli commands to it.
    """


cli.add_command(peek)


def main() -> None:
    """Initialize and run the command-line interface.

    This function initializes the Click command group and runs the command-line interface, processing user
    input and executing the appropriate commands.
    """
    cli()


if __name__ == "__main__":
    main()
