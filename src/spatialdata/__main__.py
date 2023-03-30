"""
The CLI Interaction module.

This module provides command line interface (CLI) interactions for the SpatialData library, allowing users to perform
various operations through a terminal. Currently, it implements the "peek" function, which allows users to inspect
the contents of a SpatialData .zarr file. Additional CLI functionalities will be implemented in the future.
"""
import os

import click


@click.command(help="Peek inside the SpatialData .zarr file")
@click.argument("path", default=False, type=str)
def peek(path: str) -> None:
    """
    Peek inside the SpatialData .zarr file.

    This function takes a path to a .zarr file, checks if it is a valid directory, and then reads and prints
    its contents using the SpatialData library.

    Parameters
    ----------
    path
        The path to the .zarr file to be inspected.
    """
    if not os.path.isdir(path):
        print(  # noqa: T201
            f"Error: .zarr storage not found at {path}. Please specify a valid OME-NGFF spatial data (.zarr) file. "
            "Example "
            '"python -m '
            'spatialdata peek data.zarr"'
        )
    else:
        import spatialdata as sd

        sdata = sd.SpatialData.read(path)
        print(sdata)  # noqa: T201


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
