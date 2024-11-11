"""Command-line interface."""

import click


@click.command()
@click.version_option()
def main() -> None:
    """Chemoecology Tools."""


if __name__ == "__main__":
    main(prog_name="chemoecology-tools")  # pragma: no cover
