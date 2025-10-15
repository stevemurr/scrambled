#!/usr/bin/env python3
"""Build Production POC Database from test audio files.

This script processes all audio files from the test_audio/ directory and builds
a production-ready database using the Scrambled Hacks pipeline.
"""

import sys
import json
from pathlib import Path
from collections import Counter
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from scrambled_hacks.pipeline import ScrambledHacksPipeline
from scrambled_hacks.database import SegmentDatabase, recommend_index_type
from loguru import logger


def main():
    """Build POC database from test audio files."""
    # Setup
    test_audio_dir = Path("/Users/murr/Code/scrambled/test_audio")
    output_dir = Path("/Users/murr/Code/scrambled/databases/poc_db")

    logger.info("=" * 70)
    logger.info("Building Production POC Database")
    logger.info("=" * 70)

    # Find all audio files
    audio_files = sorted(test_audio_dir.glob("*.wav"))
    logger.info(f"Found {len(audio_files)} audio files in {test_audio_dir}")

    if not audio_files:
        logger.error(f"No audio files found in {test_audio_dir}")
        return 1

    # Estimate number of segments (rough estimate: 1 segment per 0.5s)
    total_duration = 0
    logger.info("\nAudio files:")
    for i, filepath in enumerate(audio_files, 1):
        size_mb = filepath.stat().st_size / (1024 * 1024)
        logger.info(f"  {i:2d}. {filepath.name} ({size_mb:.1f} MB)")

    # Recommend index type based on estimated segments
    # Assuming ~500-2000 segments per song, use middle estimate
    estimated_segments = len(audio_files) * 1000
    index_type, reason = recommend_index_type(estimated_segments)
    logger.info(f"\n{reason}")
    logger.info(f"Using index type: {index_type.upper()}")

    # Create database with recommended index type
    database = SegmentDatabase(
        embedding_dim=512,
        index_type=index_type,
        M=32,
        efConstruction=40
    )

    # Create pipeline
    logger.info("\nInitializing pipeline...")
    pipeline = ScrambledHacksPipeline(
        database=database,
        device="cpu"  # Use CPU for compatibility
    )

    # Build database
    logger.info(f"\nProcessing {len(audio_files)} audio files...")
    logger.info("This may take several minutes (CLAP embedding extraction is slow)")
    logger.info("-" * 70)

    try:
        pipeline.build_database(audio_files, show_progress=True)
    except Exception as e:
        logger.error(f"Error building database: {e}")
        return 1

    # Generate statistics
    logger.info("\n" + "=" * 70)
    logger.info("Database Statistics")
    logger.info("=" * 70)

    num_segments = len(pipeline.database)
    logger.info(f"Total segments: {num_segments:,}")

    # Analyze metadata
    source_files = Counter()
    durations = []
    tempos = []
    energies = []

    for metadata in pipeline.database.metadata:
        source_files[metadata.get('source_file', 'unknown')] += 1

        # Note: Duration not in metadata, but we can calculate from segment index
        # Tempo and energy would need to be extracted from segments
        if 'tempo_bpm' in metadata:
            tempos.append(metadata['tempo_bpm'])

    logger.info(f"Source files: {len(source_files)}")
    logger.info(f"Average segments per file: {num_segments / len(source_files):.1f}")

    # Show top source files
    logger.info("\nTop 5 files by segment count:")
    for filename, count in source_files.most_common(5):
        logger.info(f"  {Path(filename).name}: {count:,} segments")

    if tempos:
        logger.info(f"\nTempo statistics:")
        logger.info(f"  Mean: {np.mean(tempos):.1f} BPM")
        logger.info(f"  Std:  {np.std(tempos):.1f} BPM")
        logger.info(f"  Min:  {np.min(tempos):.1f} BPM")
        logger.info(f"  Max:  {np.max(tempos):.1f} BPM")

    # Save database
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"\nSaving database to {output_dir}/")
    pipeline.save_database(output_dir)

    # Calculate disk size
    total_size = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
    size_mb = total_size / (1024 * 1024)
    logger.info(f"Database size on disk: {size_mb:.1f} MB")

    # Save statistics report
    stats_file = output_dir / "statistics.json"
    stats = {
        "num_segments": num_segments,
        "num_source_files": len(source_files),
        "avg_segments_per_file": num_segments / len(source_files),
        "index_type": index_type,
        "embedding_dim": 512,
        "database_size_mb": size_mb,
        "source_files": {str(Path(k).name): v for k, v in source_files.items()},
    }

    if tempos:
        stats["tempo_stats"] = {
            "mean": float(np.mean(tempos)),
            "std": float(np.std(tempos)),
            "min": float(np.min(tempos)),
            "max": float(np.max(tempos)),
        }

    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Statistics saved to {stats_file}")

    # Success!
    logger.info("\n" + "=" * 70)
    logger.info("Database Build Complete!")
    logger.info("=" * 70)
    logger.info(f"Location: {output_dir}/")
    logger.info(f"Segments: {num_segments:,}")
    logger.info(f"Size: {size_mb:.1f} MB")
    logger.info("\nDatabase is ready for use with the CLI tool (Phase 1.10)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
