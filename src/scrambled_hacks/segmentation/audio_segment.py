"""AudioSegment dataclass for storing segmented audio with metadata."""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
import numpy as np


@dataclass
class AudioSegment:
    """Represents a single audio segment with embedding and metadata.

    Attributes:
        audio: Raw audio samples as numpy array
        start_sample: Start position in original audio (sample index)
        end_sample: End position in original audio (sample index)
        sr: Sample rate in Hz
        embedding: CLAP embedding vector (optional, computed later)
        metadata: Additional metadata dictionary
    """

    audio: np.ndarray
    start_sample: int
    end_sample: int
    sr: int
    embedding: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate segment data after initialization."""
        if self.audio.size == 0:
            raise ValueError("Audio array is empty")

        if self.start_sample < 0:
            raise ValueError(f"start_sample must be >= 0, got {self.start_sample}")

        if self.end_sample <= self.start_sample:
            raise ValueError(
                f"end_sample ({self.end_sample}) must be > start_sample ({self.start_sample})"
            )

        if self.sr <= 0:
            raise ValueError(f"Sample rate must be > 0, got {self.sr}")

    @property
    def duration_samples(self) -> int:
        """Duration in samples."""
        return self.end_sample - self.start_sample

    @property
    def duration_seconds(self) -> float:
        """Duration in seconds."""
        return self.duration_samples / self.sr

    @property
    def start_time(self) -> float:
        """Start time in seconds."""
        return self.start_sample / self.sr

    @property
    def end_time(self) -> float:
        """End time in seconds."""
        return self.end_sample / self.sr

    @property
    def has_embedding(self) -> bool:
        """Check if embedding has been computed."""
        return self.embedding is not None

    def __repr__(self) -> str:
        return (
            f"AudioSegment("
            f"start={self.start_time:.3f}s, "
            f"end={self.end_time:.3f}s, "
            f"duration={self.duration_seconds:.3f}s, "
            f"sr={self.sr}Hz, "
            f"has_embedding={self.has_embedding})"
        )
