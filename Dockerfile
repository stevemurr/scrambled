# Scrambled Hacks + AudioCraft Container
# Simple pip-based installation following AudioCraft's official instructions

FROM --platform=linux/amd64 python:3.9-slim-bullseye

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies (ffmpeg is required by AudioCraft)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    ffmpeg \
    libsndfile1 \
    pkg-config \
    libavcodec-dev \
    libavformat-dev \
    libavdevice-dev \
    libavutil-dev \
    libavfilter-dev \
    libswscale-dev \
    libswresample-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Copy only dependency files first (for better caching)
COPY pyproject.toml ./

# Install PyTorch CPU-only first (as recommended by AudioCraft docs)
# Using CPU-only to avoid downloading 2GB+ of CUDA libraries
RUN pip install --no-cache-dir torch==2.1.0+cpu torchaudio==2.1.0+cpu --index-url https://download.pytorch.org/whl/cpu

# Install setuptools and wheel
RUN pip install --no-cache-dir setuptools wheel

# Copy the rest of the application
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY examples/ ./examples/

# Install the application with all dependencies (including AudioCraft)
# With x86 emulation, we get pre-built wheels so this should be fast
RUN pip install --no-cache-dir -e .

# Create necessary directories
RUN mkdir -p /workspace/databases /workspace/test_audio /workspace/output

# Verify AudioCraft installation
RUN python -c "import audiocraft; print(f'AudioCraft {audiocraft.__version__} installed successfully')"

# Set the default command
CMD ["/bin/bash"]

# Expose ports for Gradio/web interfaces
EXPOSE 7860
