import os

import click


@click.command(help="Peek inside the SpatialData .zarr file")
@click.argument("path", default=False, type=str)
def peek(path):
    if not os.path.isdir(path):
        print(
            f"Error: .zarr storage not found at {path}. Please specify a valid OME-NGFF spatial data (.zarr) file. "
            "Example "
            '"python -m '
            'spatialdata peek data.zarr"'
        )
    else:
        import spatialdata as sd

        sdata = sd.SpatialData.read(path, filter_table=True)
        print(sdata)


@click.group()
def cli():
    pass


cli.add_command(peek)


def main():
    cli()


if __name__ == "__main__":
    main()
