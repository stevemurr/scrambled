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

## Project Status

### Current Phase: Phase 1 - Core Pipeline (In Progress)

- [x] Environment setup with src/ layout
- [x] Project structure with pyproject.toml
- [x] Basic CLI framework
- [ ] CLAP embedding extraction
- [ ] Beat-based segmentation
- [ ] FAISS database implementation
- [ ] Simple reconstruction module
- [ ] End-to-end pipeline

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
