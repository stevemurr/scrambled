# Scrambled Hacks

Recreation of scrambled hacks using modern ML-based audio embeddings.

## Overview

Scrambled Hacks is an audio manipulation system that takes an input audio file and reconstructs it using segments from a database of audio files. Unlike the original MFCC+DTW approach, this version uses:

- **CLAP embeddings** for semantic audio understanding
- **FAISS vector search** for efficient similarity matching
- **Pluggable reconstruction modules** for flexible output generation

## Features

- 🎵 Works with music, vocals, speech, and any audio input
- ⚡ Fast similarity search using FAISS
- 🧠 Semantic matching using pre-trained CLAP models
- 🔌 Modular architecture with pluggable components
- 📦 Easy installation and CLI interface

## Installation

### Prerequisites

- Python 3.10 or 3.11
- macOS, Linux, or Windows

### Quick Start

```bash
# Clone the repository
git clone <repo-url>
cd scrambled

# Create environment and install dependencies
make env

# Activate the virtual environment
source venv/bin/activate

# Verify installation
scrambled-hacks hello

# Test CLAP model loading (downloads ~1GB model on first run)
make verify-clap
```

### Manual Installation

If you prefer manual setup:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install -e .
```

## CLI Usage

The Scrambled Hacks CLI provides three main commands:

### 1. Building a Database

Build a searchable database from a collection of audio files:

```bash
scrambled-hacks build-db -i ./test_audio -o ./databases/my_database
```

**Options:**

- `-i, --input-dir`: Directory containing audio files (required)
- `-o, --output`: Output directory for the database (required)
- `--index-type`: Index type - `flat`, `hnsw`, or `auto` (default: auto)
- `--pattern`: File pattern to match (default: `*.wav`)
- `--device`: Device to use - `cpu`, `cuda`, or `mps` (default: cpu)
- `-v, --verbose`: Enable detailed logging

**What happens:**

1. Scans input directory for audio files
2. Segments each file by beat onsets
3. Extracts CLAP embeddings (512-dimensional) for each segment
4. Builds a FAISS index for fast similarity search
5. Saves database with metadata and statistics

**Output structure:**

```
databases/my_database/
├── index.faiss          # FAISS index file
├── metadata.json        # Segment metadata
├── config.json          # Database configuration
└── statistics.json      # Build statistics
```

### 2. Viewing Database Info

Display information about an existing database:

```bash
scrambled-hacks info ./databases/my_database
```

Shows segment count, index type, embedding dimensions, source files, and disk usage.

### 3. Generating Scrambled Audio

Generate scrambled audio by matching input segments to database segments:

```bash
scrambled-hacks generate \
  -i ./input.wav \
  -d ./databases/my_database \
  -o ./output.wav
```

**Options:**

- `-i, --input`: Input audio file to scramble (required)
- `-d, --database`: Database directory to use (required)
- `-o, --output`: Output audio file path (required)
- `--k`: Number of nearest neighbors per segment (default: 1)
- `-r, --reconstructor`: Reconstruction strategy - `simple` or `direct` (default: simple)
  - `simple`: Concatenation with 10ms crossfading for smooth transitions
  - `direct`: Direct concatenation without crossfading
- `--device`: Device to use - `cpu`, `cuda`, or `mps` (default: cpu)
- `-v, --verbose`: Enable detailed logging

**What happens:**

1. Loads the database
2. Segments input audio by beats
3. For each input segment, finds k most similar database segments
4. Reconstructs output audio from matched segments
5. Saves result to output file

### Quick Start Workflow

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Build database from audio collection
scrambled-hacks build-db -i ./test_audio -o ./databases/production_db

# 3. View database info
scrambled-hacks info ./databases/production_db

# 4. Generate scrambled audio
scrambled-hacks generate \
  -i ./test_audio/input.wav \
  -d ./databases/production_db \
  -o ./scrambled_output.wav

# 5. Try different reconstruction strategies
scrambled-hacks generate \
  -i ./test_audio/input.wav \
  -d ./databases/production_db \
  -o ./output_direct.wav \
  -r direct
```

### Performance Notes

- **CLAP embedding extraction is slow** (~15-20 seconds per file on CPU)
- Building a database from 26 files takes ~10 minutes
- Database auto-selects HNSW index for better search performance on larger collections
- Once built, databases load in <5 seconds
- Generation is much faster than building

### Troubleshooting

**No audio files found:**
- Check `--pattern` matches your files (default: `*.wav`)
- Try patterns like `*.mp3`, `*.flac`, etc.

**Memory issues:**
- Use HNSW index for large databases
- The flat index loads all vectors into memory

**Device errors:**
- Use `--device cpu` for maximum compatibility
- `mps` works on Apple Silicon but may have issues
- `cuda` requires NVIDIA GPU with CUDA installed

## Project Status

### Current Phase: Phase 1 - Core Pipeline (Complete! ✅)

- [x] Environment setup with src/ layout
- [x] Project structure with pyproject.toml
- [x] Basic CLI framework
- [x] CLAP embedding extraction
- [x] Beat-based segmentation
- [x] FAISS database implementation
- [x] Simple reconstruction module
- [x] End-to-end pipeline

See [docs/DEVELOPMENT_PLAN.md](docs/DEVELOPMENT_PLAN.md) for full roadmap.

## Development

### Available Commands

```bash
make help          # Show all available commands
make env           # Create virtual environment and install dependencies
make test          # Run tests with pytest
make format        # Format code with black and isort
make lint          # Run linters (flake8, mypy)
make clean         # Remove virtual environment and cache files
make verify-clap   # Test CLAP model loading
```

### Running Tests

```bash
make test
```

### Project Structure

```
scrambled/
├── src/
│   └── scrambled_hacks/
│       ├── embeddings/        # CLAP embedding extraction
│       ├── segmentation/      # Beat-based audio segmentation
│       ├── database/          # FAISS vector database
│       ├── reconstruction/    # Audio reconstruction modules
│       ├── cli.py            # Command-line interface
│       └── pipeline.py       # End-to-end pipeline (coming soon)
├── tests/                    # Test suite
├── docs/                     # Documentation
├── examples/                 # Example workflows
├── pyproject.toml           # Project configuration
├── Makefile                 # Development commands
└── README.md
```

## Documentation

- [PLAN.md](docs/PLAN.md) - Comprehensive architecture and research
- [DEVELOPMENT_PLAN.md](docs/DEVELOPMENT_PLAN.md) - Phased implementation plan

## Original Scrambled Hacks

[Original Example of Scrambled Hacks](https://www.youtube.com/watch?v=eRlhKaxcKpA)

## License

MIT

## Acknowledgments

- CLAP: Contrastive Language-Audio Pretraining
- FAISS: Facebook AI Similarity Search
- librosa: Audio analysis library
