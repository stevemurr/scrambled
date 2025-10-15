"""Command-line interface for Scrambled Hacks."""

import click


@click.group()
@click.version_option(version="0.1.0")
def main():
    """Scrambled Hacks: ML-based audio scrambling system."""
    pass


@main.command()
def hello():
    """Test command to verify installation."""
    click.echo("✅ Scrambled Hacks CLI is working!")
    click.echo("Run 'scrambled-hacks --help' for available commands.")


if __name__ == "__main__":
    main()
