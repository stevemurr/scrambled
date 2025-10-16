import click
import sys
import json
from pathlib import Path
from loguru import logger

from .pipeline import ScrambledHacksPipeline
from .database import SegmentDatabase, recommend_index_type
from .reconstruction import SimpleConcatenation, DirectConcatenation

# Configure loguru
logger.remove()  # Remove default handler
logger.add(
    sys.stderr,
    format="<level>{level: <8}</level> | <level>{message}</level>",
    level="INFO"
)


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


@main.command("build-db")
@click.option(
    "--input-dir", "-i",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="Directory containing audio files to process"
)
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output directory for the database"
)
@click.option(
    "--index-type",
    type=click.Choice(["flat", "hnsw", "auto"], case_sensitive=False),
    default="auto",
    help="FAISS index type (auto=recommend based on size)"
)
@click.option(
    "--pattern",
    default="*.wav",
    help="File pattern to match (default: *.wav)"
)
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda", "mps"], case_sensitive=False),
    default="cpu",
    help="Device to use for CLAP embeddings"
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Enable verbose logging"
)
def build_db(input_dir, output, index_type, pattern, device, verbose):
    """Build a database from audio files.

    Scans INPUT_DIR for audio files matching PATTERN, segments them by beats,
    extracts CLAP embeddings, and builds a searchable FAISS database.

    Example:
        scrambled-hacks build-db -i ./audio -o ./my_database
    """
    # Configure logging level
    if verbose:
        logger.remove()
        logger.add(
            sys.stderr,
            format="<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> | <level>{message}</level>",
            level="DEBUG"
        )

    # Find audio files
    click.echo(f"Scanning {input_dir} for {pattern} files...")
    audio_files = sorted(input_dir.glob(pattern))

    if not audio_files:
        click.echo(f"❌ No audio files found matching pattern: {pattern}", err=True)
        sys.exit(1)

    click.echo(f"Found {len(audio_files)} audio files")

    # Auto-select index type if requested
    if index_type == "auto":
        # Estimate segments (rough: 1000 segments per file)
        estimated_segments = len(audio_files) * 1000
        selected_index_type, reason = recommend_index_type(estimated_segments)
        click.echo(f"\n{reason}")
        click.echo(f"Selected index type: {selected_index_type.upper()}")
        index_type = selected_index_type
    else:
        click.echo(f"Using index type: {index_type.upper()}")

    # Create database
    if index_type == "flat":
        database = SegmentDatabase(embedding_dim=512, index_type="flat")
    elif index_type == "hnsw":
        database = SegmentDatabase(
            embedding_dim=512,
            index_type="hnsw",
            M=32,
            efConstruction=40
        )
    else:
        click.echo(f"❌ Unsupported index type: {index_type}", err=True)
        sys.exit(1)

    # Create pipeline
    click.echo(f"\nInitializing pipeline with device={device}...")
    pipeline = ScrambledHacksPipeline(database=database, device=device)

    # Build database
    click.echo(f"\nProcessing {len(audio_files)} audio files...")
    click.echo("This may take several minutes (CLAP embedding extraction is slow)")
    click.echo("-" * 70)

    try:
        pipeline.build_database(audio_files, show_progress=True)
    except Exception as e:
        click.echo(f"\n❌ Error building database: {e}", err=True)
        sys.exit(1)

    # Show statistics
    click.echo("\n" + "=" * 70)
    click.echo("Database Statistics")
    click.echo("=" * 70)
    click.echo(f"Total segments: {pipeline.database.size:,}")

    # Save database
    output.mkdir(parents=True, exist_ok=True)
    click.echo(f"\nSaving database to {output}/")
    pipeline.save_database(output)

    # Calculate disk size
    total_size = sum(f.stat().st_size for f in output.rglob("*") if f.is_file())
    size_mb = total_size / (1024 * 1024)
    click.echo(f"Database size on disk: {size_mb:.1f} MB")

    # Save statistics
    stats_file = output / "statistics.json"
    stats = {
        "num_segments": pipeline.database.size,
        "num_source_files": len(audio_files),
        "index_type": index_type,
        "embedding_dim": 512,
        "database_size_mb": size_mb,
    }

    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    click.echo(f"Statistics saved to {stats_file}")

    # Success
    click.echo("\n" + "=" * 70)
    click.echo("✅ Database Build Complete!")
    click.echo("=" * 70)
    click.echo(f"Location: {output}/")
    click.echo(f"Segments: {pipeline.database.size:,}")
    click.echo(f"Size: {size_mb:.1f} MB")


@main.command("info")
@click.argument(
    "database",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path)
)
def info(database):
    """Show information about a database.

    Displays statistics and metadata about a built database.

    Example:
        scrambled-hacks info ./my_database
    """
    # Load config
    config_path = database / "config.json"
    if not config_path.exists():
        click.echo(f"❌ Not a valid database: {database}", err=True)
        click.echo(f"Missing config.json", err=True)
        sys.exit(1)

    with open(config_path, 'r') as f:
        config = json.load(f)

    # Load statistics if available
    stats_path = database / "statistics.json"
    stats = None
    if stats_path.exists():
        with open(stats_path, 'r') as f:
            stats = json.load(f)

    # Display info
    click.echo("=" * 70)
    click.echo(f"Database: {database.name}")
    click.echo("=" * 70)
    click.echo(f"Segments:       {config['num_segments']:,}")
    click.echo(f"Embedding dim:  {config['embedding_dim']}")
    click.echo(f"Index type:     {config['index_type'].upper()}")
    click.echo(f"Version:        {config.get('version', 'unknown')}")

    if stats:
        click.echo(f"\nSource files:   {stats.get('num_source_files', 'unknown')}")
        click.echo(f"Database size:  {stats.get('database_size_mb', 0):.1f} MB")

    # Calculate disk size
    total_size = sum(f.stat().st_size for f in database.rglob("*") if f.is_file())
    size_mb = total_size / (1024 * 1024)
    click.echo(f"Disk usage:     {size_mb:.1f} MB")

    # List files
    click.echo(f"\nFiles:")
    for filepath in sorted(database.iterdir()):
        if filepath.is_file():
            size = filepath.stat().st_size / 1024  # KB
            click.echo(f"  - {filepath.name} ({size:.1f} KB)")


@main.command("generate")
@click.option(
    "--input", "-i",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    help="Input audio file to scramble"
)
@click.option(
    "--database", "-d",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="Database directory to use"
)
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output audio file path"
)
@click.option(
    "--k",
    type=int,
    default=1,
    help="Number of nearest neighbors to retrieve (default: 1)"
)
@click.option(
    "--reconstructor", "-r",
    type=click.Choice(["simple", "direct"], case_sensitive=False),
    default="simple",
    help="Reconstruction strategy: 'simple' (crossfade), 'direct' (no crossfade)"
)
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda", "mps"], case_sensitive=False),
    default="cpu",
    help="Device to use for CLAP embeddings"
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    help="Enable verbose logging"
)
def generate(input, database, output, k, reconstructor, device, verbose):
    """Generate scrambled audio from input file.

    Takes an input audio file, finds matching segments in the database,
    and reconstructs a new version using the matched segments.

    Example:
        scrambled-hacks generate -i input.wav -d ./my_database -o output.wav
    """
    # Configure logging level
    if verbose:
        logger.remove()
        logger.add(
            sys.stderr,
            format="<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> | <level>{message}</level>",
            level="DEBUG"
        )

    click.echo(f"Loading database from {database}...")

    # Load database
    try:
        db = SegmentDatabase.load(database)
    except Exception as e:
        click.echo(f"❌ Error loading database: {e}", err=True)
        sys.exit(1)

    click.echo(f"Database loaded: {db.size:,} segments")

    # Create reconstructor based on user selection
    if reconstructor == "simple":
        reconstructor_module = SimpleConcatenation(crossfade_ms=10.0, normalize=True)
        click.echo(f"Using reconstruction: SimpleConcatenation (10ms crossfade)")
    elif reconstructor == "direct":
        reconstructor_module = DirectConcatenation()
        click.echo(f"Using reconstruction: DirectConcatenation (no crossfade)")
    else:
        click.echo(f"❌ Unknown reconstructor: {reconstructor}", err=True)
        sys.exit(1)

    # Create pipeline
    click.echo(f"Initializing pipeline with device={device}...")
    pipeline = ScrambledHacksPipeline(
        database=db,
        reconstructor=reconstructor_module,
        device=device
    )

    # Generate output
    click.echo(f"\nProcessing input: {input.name}")
    click.echo(f"Retrieving k={k} nearest neighbors per segment")
    click.echo("-" * 70)

    try:
        output_audio = pipeline.generate(input, k=k, show_progress=True)
    except Exception as e:
        click.echo(f"\n❌ Error generating output: {e}", err=True)
        sys.exit(1)

    # Get sample rate from input file
    import soundfile as sf
    _, sr = sf.read(input, frames=1)

    # Save output
    click.echo(f"\nSaving output to {output}...")
    pipeline.save_audio(output_audio, output, sr=sr)

    # Success
    click.echo("\n✅ Generation complete!")
    click.echo(f"Output: {output}")
    click.echo(f"Duration: {len(output_audio)/sr:.2f}s")


if __name__ == "__main__":
    main()
