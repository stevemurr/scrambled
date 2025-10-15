"""Tests for CLAP embedding extractor."""

import numpy as np
import pytest
from pathlib import Path

from scrambled_hacks.embeddings import ClapEmbedder


@pytest.fixture
def embedder():
    """Create a ClapEmbedder instance for testing."""
    return ClapEmbedder(device="cpu")  # Force CPU for testing


@pytest.fixture
def sample_audio():
    """Generate sample audio for testing."""
    # Generate 1 second of random audio at 48kHz
    sr = 48000
    duration = 1.0
    audio = np.random.randn(int(sr * duration)).astype(np.float32)
    return audio, sr


def test_embedder_initialization(embedder):
    """Test that ClapEmbedder initializes correctly."""
    assert embedder is not None
    assert embedder.model is not None
    assert embedder.processor is not None
    assert embedder.device == "cpu"
    assert embedder.embedding_dim > 0
    assert embedder.model_name == "laion/clap-htsat-unfused"


def test_embed_audio(embedder, sample_audio):
    """Test embedding extraction from audio array."""
    audio, sr = sample_audio

    embedding = embedder.embed_audio(audio, sr=sr)

    # Check output shape and type
    assert isinstance(embedding, np.ndarray)
    assert embedding.ndim == 1
    assert len(embedding) == embedder.embedding_dim
    assert embedding.dtype == np.float32 or embedding.dtype == np.float64


def test_embed_audio_stereo(embedder):
    """Test embedding extraction from stereo audio."""
    # Generate stereo audio
    sr = 48000
    duration = 1.0
    audio_stereo = np.random.randn(int(sr * duration), 2).astype(np.float32)

    embedding = embedder.embed_audio(audio_stereo, sr=sr)

    # Should still work and produce same shape
    assert isinstance(embedding, np.ndarray)
    assert embedding.ndim == 1
    assert len(embedding) == embedder.embedding_dim


def test_embed_audio_empty():
    """Test that empty audio raises ValueError."""
    embedder = ClapEmbedder(device="cpu")
    empty_audio = np.array([])

    with pytest.raises(ValueError, match="Audio array is empty"):
        embedder.embed_audio(empty_audio, sr=48000)


def test_embed_file_nonexistent():
    """Test that non-existent file raises FileNotFoundError."""
    embedder = ClapEmbedder(device="cpu")

    with pytest.raises(FileNotFoundError):
        embedder.embed_file("nonexistent_file.wav")


def test_embed_batch(embedder, sample_audio):
    """Test batch embedding extraction."""
    audio, sr = sample_audio

    # Create batch of 3 audio samples
    audio_list = [audio, audio * 0.5, audio * 0.25]

    embeddings = embedder.embed_batch(audio_list, sr=sr)

    # Check output shape
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (3, embedder.embedding_dim)


def test_embed_batch_empty():
    """Test that empty batch raises ValueError."""
    embedder = ClapEmbedder(device="cpu")

    with pytest.raises(ValueError, match="audio_list is empty"):
        embedder.embed_batch([], sr=48000)


def test_embedder_repr(embedder):
    """Test string representation."""
    repr_str = repr(embedder)
    assert "ClapEmbedder" in repr_str
    assert "laion/clap-htsat-unfused" in repr_str
    assert "cpu" in repr_str


def test_device_auto_detection():
    """Test automatic device detection."""
    embedder = ClapEmbedder(device=None)  # Auto-detect

    # Should select one of cpu, cuda, or mps
    assert embedder.device in ["cpu", "cuda", "mps"]


def test_embedding_consistency(embedder, sample_audio):
    """Test that same audio produces consistent embeddings."""
    audio, sr = sample_audio

    embedding1 = embedder.embed_audio(audio, sr=sr)
    embedding2 = embedder.embed_audio(audio, sr=sr)

    # Embeddings should be very similar (allowing for small numerical differences)
    np.testing.assert_allclose(embedding1, embedding2, rtol=1e-5, atol=1e-7)


def test_different_audio_different_embeddings(embedder):
    """Test that different audio produces different embeddings."""
    sr = 48000

    # Generate two different audio samples
    audio1 = np.random.randn(sr).astype(np.float32)
    audio2 = np.random.randn(sr).astype(np.float32)

    embedding1 = embedder.embed_audio(audio1, sr=sr)
    embedding2 = embedder.embed_audio(audio2, sr=sr)

    # Embeddings should be different
    assert not np.allclose(embedding1, embedding2)
