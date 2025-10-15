"""CLAP audio embedding extractor.

This module provides functionality to extract semantic audio embeddings using
the CLAP (Contrastive Language-Audio Pretraining) model from HuggingFace.
"""

from typing import Optional, Union
from pathlib import Path

import numpy as np
import torch
import soundfile as sf
from transformers import ClapModel, ClapProcessor
from loguru import logger


class ClapEmbedder:
    """Extract CLAP embeddings from audio.

    Uses the pre-trained CLAP model to generate semantic audio embeddings
    that capture musical and acoustic properties like timbre, rhythm, energy,
    and genre characteristics.

    Attributes:
        model_name: HuggingFace model identifier
        device: Device to run model on ('cpu', 'cuda', or 'mps')
        model: Loaded CLAP model
        processor: CLAP audio processor
        embedding_dim: Dimensionality of output embeddings
    """

    def __init__(
        self,
        model_name: str = "laion/clap-htsat-unfused",
        device: Optional[str] = None
    ):
        """Initialize CLAP embedder.

        Args:
            model_name: HuggingFace model ID. Options:
                - "laion/clap-htsat-unfused" (general audio, recommended)
                - "laion/larger_clap_music_and_speech"
            device: Device to use. If None, auto-detects best available device.
        """
        self.model_name = model_name
        self.device = self._get_device(device)

        logger.info(f"Loading CLAP model: {model_name}")
        logger.info(f"Using device: {self.device}")

        # Load model and processor
        self.model = ClapModel.from_pretrained(model_name)
        self.processor = ClapProcessor.from_pretrained(model_name)

        # Move model to device
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode

        # Get embedding dimension
        self.embedding_dim = self.model.config.projection_dim

        logger.info(f"CLAP model loaded successfully (embedding_dim={self.embedding_dim})")

    def _get_device(self, device: Optional[str] = None) -> str:
        """Determine best available device.

        Args:
            device: Requested device or None for auto-detection

        Returns:
            Device string ('cpu', 'cuda', or 'mps')
        """
        if device is not None:
            return device

        # Auto-detect best available device
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            # Apple Silicon MPS
            return "mps"
        else:
            return "cpu"

    def embed_audio(
        self,
        audio: np.ndarray,
        sr: int = 48000
    ) -> np.ndarray:
        """Extract CLAP embeddings from audio array.

        Args:
            audio: Audio samples as numpy array (mono or stereo)
            sr: Sample rate of audio

        Returns:
            CLAP embedding as numpy array of shape (embedding_dim,)

        Raises:
            ValueError: If audio array is empty or invalid
        """
        if audio.size == 0:
            raise ValueError("Audio array is empty")

        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        # Process audio through CLAP processor
        inputs = self.processor(
            audio=audio,
            sampling_rate=sr,
            return_tensors="pt"
        )

        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Extract embeddings (no gradients needed)
        with torch.no_grad():
            embeddings = self.model.get_audio_features(**inputs)

        # Move back to CPU and convert to numpy
        embeddings = embeddings.cpu().numpy()

        # Return flattened embedding
        return embeddings.flatten()

    def embed_file(
        self,
        filepath: Union[str, Path],
        sr: Optional[int] = None
    ) -> np.ndarray:
        """Extract CLAP embeddings from audio file.

        Args:
            filepath: Path to audio file (supports .wav, .mp3, .flac, etc.)
            sr: Target sample rate. If None, uses file's native sample rate.

        Returns:
            CLAP embedding as numpy array of shape (embedding_dim,)

        Raises:
            FileNotFoundError: If audio file doesn't exist
            ValueError: If audio file is invalid or empty
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Audio file not found: {filepath}")

        # Load audio file
        audio, file_sr = sf.read(filepath, always_2d=False)

        # Use file's sample rate if not specified
        if sr is None:
            sr = file_sr
        elif sr != file_sr:
            # Resample if needed
            logger.debug(f"Resampling from {file_sr}Hz to {sr}Hz")
            import librosa
            audio = librosa.resample(audio, orig_sr=file_sr, target_sr=sr)

        # Extract embeddings
        return self.embed_audio(audio, sr=sr)

    def embed_batch(
        self,
        audio_list: list[np.ndarray],
        sr: int = 48000
    ) -> np.ndarray:
        """Extract CLAP embeddings from batch of audio arrays.

        More efficient than calling embed_audio() in a loop for multiple files.

        Args:
            audio_list: List of audio arrays
            sr: Sample rate of all audio arrays

        Returns:
            Batch of embeddings as numpy array of shape (batch_size, embedding_dim)

        Raises:
            ValueError: If audio_list is empty or contains invalid audio
        """
        if not audio_list:
            raise ValueError("audio_list is empty")

        # Convert all to mono if needed
        processed_audio = []
        for audio in audio_list:
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            processed_audio.append(audio)

        # Process batch through CLAP processor
        inputs = self.processor(
            audio=processed_audio,
            sampling_rate=sr,
            return_tensors="pt"
        )

        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Extract embeddings
        with torch.no_grad():
            embeddings = self.model.get_audio_features(**inputs)

        # Move back to CPU and convert to numpy
        embeddings = embeddings.cpu().numpy()

        return embeddings

    def __repr__(self) -> str:
        return (
            f"ClapEmbedder(model_name='{self.model_name}', "
            f"device='{self.device}', embedding_dim={self.embedding_dim})"
        )
