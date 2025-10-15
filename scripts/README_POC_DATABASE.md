# POC Database Build

## Overview

This directory contains the script to build the Production POC database for the Scrambled Hacks project.

## Script: `build_poc_database.py`

Builds a production-ready FAISS database from 26 audio files in the `test_audio/` directory.

### Features

- **Automatic index selection**: Uses `recommend_index_type()` to select optimal FAISS index
- **Progress tracking**: Real-time progress bars showing file processing status
- **Statistics generation**: Comprehensive database statistics saved to JSON
- **Enhanced metadata**: Tempo (BPM), segment counts, source file tracking
- **Production ready**: Uses IndexHNSWFlat for optimal search performance

### Usage

```bash
# Run the database build script
cd /Users/murr/Code/scrambled
./venv/bin/python scripts/build_poc_database.py
```

### Expected Output

```
databases/poc_db/
├── index.faiss           # FAISS HNSW index
├── metadata.json         # Segment metadata (source files, tempo, positions)
├── config.json           # Database configuration
└── statistics.json       # Database statistics report
```

### Expected Results

Based on 26 audio files (~2.3GB, 48kHz stereo):

- **Total segments**: ~13,000 - 52,000 (varies by song tempo/complexity)
- **Index type**: IndexHNSWFlat (optimal for this size)
- **Database size**: ~50-200 MB on disk
- **Build time**: ~10-15 minutes (CLAP embedding extraction is slow)

### Statistics Report

The script generates `statistics.json` containing:

```json
{
  "num_segments": 26000,
  "num_source_files": 26,
  "avg_segments_per_file": 1000.0,
  "index_type": "hnsw",
  "embedding_dim": 512,
  "database_size_mb": 150.5,
  "tempo_stats": {
    "mean": 120.5,
    "std": 25.3,
    "min": 85.2,
    "max": 180.0
  },
  "source_files": {
    "Acolyte-LongKissGoodnight.wav": 269,
    ...
  }
}
```

### Performance

Processing time per file (approximate):
- Segmentation: 2-3 seconds
- CLAP embedding extraction: 10-15 seconds (depends on segment count)
- Total per file: 15-20 seconds

With 26 files: **~10 minutes total**

### Notes

- Files must be at 48kHz (CLAP's expected sample rate)
- Test audio files are already at 48kHz - no resampling needed
- CLAP model download happens automatically on first run (~500MB)
- Uses CPU by default for compatibility (can be changed to MPS/CUDA)

### Troubleshooting

**Out of memory**: Reduce batch size in CLAP embedder (default handles ~1000 segments)

**Slow processing**: Normal - CLAP embedding extraction is compute-intensive

**Missing CLAP model**: Will auto-download on first run (requires internet)

## Next Steps

Once the database is built:

1. **Phase 1.10**: Create CLI tool to use this database
2. **Testing**: Run sample queries to verify database quality
3. **Generate audio**: Test end-to-end audio generation with real music

## Files

- `build_poc_database.py` - Main database build script
- `poc_build_log.txt` - Build log (created during execution)
- `benchmark_index_types.py` - FAISS index benchmarking tool
- `benchmark_results.txt` - Benchmark results
